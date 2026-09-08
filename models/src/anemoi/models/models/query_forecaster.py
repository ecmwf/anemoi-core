# (C) Copyright 2026 Anemoi contributors.

"""Metadata-conditioned direct forecasting on native or requested coordinates."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo, GraphShardInfo
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph_provider import (
    DynamicGraphProvider,
    create_graph_provider,
)
from anemoi.models.layers.query_adapter import QueryMetadataAdapter, QueryValueAdapter


class SphericalKNNGraphProvider(DynamicGraphProvider):
    """Build a small topology-only k-nearest-neighbour graph from radians."""

    def __init__(self, neighbours: int, chunk_size: int) -> None:
        super().__init__(edge_dim=1)
        self.neighbours = neighbours
        self.chunk_size = chunk_size

    def build_graph(
        self, src_nodes: Tensor, dst_nodes: Tensor, **_kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        if (
            src_nodes.ndim != 2
            or dst_nodes.ndim != 2
            or src_nodes.shape[1] != 2
            or dst_nodes.shape[1] != 2
        ):
            raise ValueError(
                "Dynamic output coordinates must have shape (nodes, 2) as latitude/longitude radians."
            )
        if not src_nodes.shape[0] or not dst_nodes.shape[0]:
            raise ValueError(
                "Dynamic graph construction requires non-empty source and output coordinates."
            )
        src_xyz = torch.stack(
            (
                torch.cos(src_nodes[:, 0]) * torch.cos(src_nodes[:, 1]),
                torch.cos(src_nodes[:, 0]) * torch.sin(src_nodes[:, 1]),
                torch.sin(src_nodes[:, 0]),
            ),
            dim=-1,
        )
        dst_xyz = torch.stack(
            (
                torch.cos(dst_nodes[:, 0]) * torch.cos(dst_nodes[:, 1]),
                torch.cos(dst_nodes[:, 0]) * torch.sin(dst_nodes[:, 1]),
                torch.sin(dst_nodes[:, 0]),
            ),
            dim=-1,
        )
        neighbours = min(self.neighbours, src_nodes.shape[0])
        selected_chunks = []
        source_chunks = []
        destination_chunks = []
        for start in range(0, dst_xyz.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, dst_xyz.shape[0])
            distances = torch.cdist(dst_xyz[start:stop], src_xyz)
            selected, source = torch.topk(distances, neighbours, largest=False, dim=1)
            destination = torch.arange(start, stop, device=dst_nodes.device)[
                :, None
            ].expand_as(source)
            selected_chunks.append(selected)
            source_chunks.append(source)
            destination_chunks.append(destination)
        selected = torch.cat(selected_chunks)
        source = torch.cat(source_chunks)
        destination = torch.cat(destination_chunks)
        edge_index = torch.stack((source.reshape(-1), destination.reshape(-1)))
        scale = selected.max().clamp_min(torch.finfo(selected.dtype).eps)
        return (selected.reshape(-1, 1) / scale), edge_index


class QueryForecaster(nn.Module):
    """One shared model for variable-sized input fields and scalar output queries."""

    def __init__(
        self,
        config: Any,
        graph_data: HeteroData,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays
        self._graph_data = graph_data
        self.dataset_names = list(metadata["metadata_inference"]["dataset_names"])
        catalogue = metadata["metadata_inference"]["query_catalogue"]
        self.query_catalogue = catalogue

        model_config = config.model
        hidden = model_config.num_channels
        adapter_hidden = model_config.query.metadata_hidden_dim
        self.node_attributes = NamedNodesAttributes(
            {
                name: 0
                for name in [*self.dataset_names, model_config.model.hidden_nodes_name]
            },
            graph_data,
        )
        self.hidden_name = model_config.model.hidden_nodes_name
        self.latent_skip = model_config.model.latent_skip
        self.value_adapter = QueryValueAdapter(
            len(catalogue["continuous_metadata"]),
            adapter_hidden,
            len(catalogue["variables"]),
            len(catalogue["provenances"]),
            len(catalogue.get("units", ["unknown"])),
            model_config.query.adapter_node_chunk_size,
        )
        self.query_adapter = QueryMetadataAdapter(
            len(catalogue["continuous_metadata"]),
            adapter_hidden,
            self.value_adapter.variable_embedding,
            self.value_adapter.provenance_embedding,
            self.value_adapter.unit_embedding,
        )
        self.query_to_latent = nn.Linear(adapter_hidden, hidden)

        self.encoder_graph_provider = nn.ModuleDict()
        for name in self.dataset_names:
            self.encoder_graph_provider[name] = create_graph_provider(
                graph=graph_data[(name, "to", self.hidden_name)],
                edge_attributes=model_config.encoders.shared.mapper.sub_graph_edge_attributes,
                src_size=self.node_attributes.num_nodes[name],
                dst_size=self.node_attributes.num_nodes[self.hidden_name],
                trainable_size=0,
            )
        edge_dims = {
            provider.edge_dim for provider in self.encoder_graph_provider.values()
        }
        if len(edge_dims) != 1:
            raise ValueError(
                f"Shared query encoder requires equal edge dimensions, got {edge_dims}."
            )
        self.encoder = instantiate(
            model_config.encoders.shared.mapper,
            _recursive_=False,
            in_channels_src=adapter_hidden
            + 1
            + self.node_attributes.attr_ndims[self.dataset_names[0]],
            in_channels_dst=self.node_attributes.attr_ndims[self.hidden_name],
            edge_dim=edge_dims.pop(),
        )

        self.processor_graph_provider = create_graph_provider(
            graph=graph_data[(self.hidden_name, "to", self.hidden_name)],
            edge_attributes=model_config.processor.sub_graph_edge_attributes,
            src_size=self.node_attributes.num_nodes[self.hidden_name],
            dst_size=self.node_attributes.num_nodes[self.hidden_name],
            trainable_size=0,
        )
        self.processor = instantiate(
            model_config.processor,
            _recursive_=False,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        self.decoder_graph_provider = nn.ModuleDict()
        for name in self.dataset_names:
            self.decoder_graph_provider[name] = create_graph_provider(
                graph=graph_data[(self.hidden_name, "to", name)],
                edge_attributes=model_config.decoders.shared.mapper.sub_graph_edge_attributes,
                src_size=self.node_attributes.num_nodes[self.hidden_name],
                dst_size=self.node_attributes.num_nodes[name],
                trainable_size=0,
            )
        self.dynamic_decoder_graph_provider = SphericalKNNGraphProvider(
            model_config.query.decoder_neighbours,
            model_config.query.decoder_chunk_size,
        )
        self.decoder = instantiate(
            model_config.decoders.shared.mapper,
            _recursive_=False,
            in_channels_src=hidden,
            in_channels_dst=adapter_hidden
            + self.node_attributes.attr_ndims[self.dataset_names[0]],
            out_channels_dst=1,
            edge_dim=next(iter(self.decoder_graph_provider.values())).edge_dim,
        )

    def _forward_tensors(
        self,
        inputs: dict[str, Any],
        query: dict[str, Any],
        output_coordinates: Tensor | None = None,
        model_comm_group: Any = None,
    ) -> Tensor:
        if model_comm_group is not None and model_comm_group.size() != 1:
            raise NotImplementedError(
                "The initial query model supports one GPU per model group."
            )
        batch_size = 1
        hidden_nodes = self.node_attributes(self.hidden_name, batch_size)
        source_latents = []
        source_coverages = []
        for name, source in inputs.items():
            if name not in self.encoder_graph_provider:
                raise KeyError(
                    f"Input source {name!r} has no registered training geometry."
                )
            pooled = self.value_adapter(
                source.values,
                source.metadata,
                source.variable_ids,
                source.provenance_ids,
                source.unit_ids,
                source.mask,
            )
            pooled = pooled.reshape(-1, pooled.shape[-1])
            source_coverage = pooled[:, -1]
            source_nodes = torch.cat(
                (
                    pooled,
                    self.node_attributes(name, batch_size) * source_coverage[:, None],
                ),
                dim=-1,
            )
            edge_attr, edge_index, edge_sizes = self.encoder_graph_provider[
                name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            _, latent = self.encoder(
                (source_nodes, hidden_nodes),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=None, dst_nodes=None, edges=edge_sizes
                ),
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )
            source_latents.append(latent)
            coverage = torch.zeros(
                self.node_attributes.num_nodes[self.hidden_name],
                dtype=latent.dtype,
                device=latent.device,
            )
            coverage.index_add_(0, edge_index[1], source_coverage[edge_index[0]])
            source_coverages.append((coverage > 0).to(latent.dtype)[:, None])
        if not source_latents:
            raise ValueError("At least one registered input source is required.")
        coverages = torch.stack(source_coverages)
        latent = (torch.stack(source_latents) * coverages).sum(dim=0) / coverages.sum(
            dim=0
        ).clamp_min(1)

        query_embedding = self.query_adapter(
            query["metadata"],
            query["variable_id"],
            query["provenance_id"],
            query["unit_id"],
        )
        latent = latent + self.query_to_latent(query_embedding).repeat_interleave(
            self.node_attributes.num_nodes[self.hidden_name], dim=0
        )
        edge_attr, edge_index, edge_sizes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        processed = self.processor(
            x=latent,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=None, edges=edge_sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )
        if self.latent_skip:
            processed = processed + latent

        target_name = query.get("grid")
        if output_coordinates is None:
            if target_name not in self.decoder_graph_provider:
                raise ValueError(
                    "Query grid must name a registered geometry or output_coordinates must be supplied."
                )
            target_coordinates = self.node_attributes.get_coordinates(target_name)
            target_nodes = self.node_attributes(target_name, batch_size)
            provider = self.decoder_graph_provider[target_name]
            edge_attr, edge_index, edge_sizes = provider.get_edges(
                batch_size=1, model_comm_group=model_comm_group
            )
        else:
            target_coordinates = output_coordinates.to(
                device=processed.device, dtype=processed.dtype
            )
            target_nodes = torch.cat(
                (torch.sin(target_coordinates), torch.cos(target_coordinates)), dim=-1
            )
            provider = self.dynamic_decoder_graph_provider
            edge_attr, edge_index, edge_sizes = provider.get_edges(
                batch_size=1,
                src_coords=self.node_attributes.get_coordinates(self.hidden_name),
                dst_coords=target_coordinates,
                model_comm_group=model_comm_group,
            )
        target_query = query_embedding.repeat_interleave(
            target_coordinates.shape[0], dim=0
        )
        destination = torch.cat((target_query, target_nodes), dim=-1)
        output = self.decoder(
            (processed, destination),
            batch_size=1,
            shard_info=BipartiteGraphShardInfo(
                src_nodes=None, dst_nodes=None, edges=edge_sizes
            ),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )
        return output.reshape(1, -1)

    def forward(
        self,
        inputs: dict[str, Any],
        query: dict[str, Any],
        output_coordinates: Tensor | None = None,
        model_comm_group: Any = None,
    ) -> Tensor:
        """Return normalized values for tensor queries or physical values for user dictionaries."""
        if "metadata" in query:
            return self._forward_tensors(
                inputs, query, output_coordinates, model_comm_group
            )
        return self.predict(inputs, query, output_coordinates)

    def predict(
        self,
        inputs: dict[str, dict[str, Any]],
        query: dict[str, Any],
        output_coordinates: Tensor | np.ndarray | None = None,
    ) -> Tensor:
        """Canonicalize physical-valued fields and evaluate ``model(inputs, query)``."""
        from anemoi.models.query import QueryMetadata

        if output_coordinates is None:
            output_coordinates = query.get("output_coordinates")
        if query.get("horizon") is not None or query.get("window") is not None:
            cadence = query.get(
                "output_frequency",
                query.get("frequency", query.get("output_cadence")),
            )
            if cadence is None:
                raise ValueError("A horizon/window query requires output_frequency.")
            start = timedelta(hours=QueryMetadata._hours(query["lead_time"]))
            end = (
                timedelta(hours=QueryMetadata._hours(query["horizon"]))
                if query.get("horizon") is not None
                else start + timedelta(hours=QueryMetadata._hours(query["window"]))
            )
            step = timedelta(hours=QueryMetadata._hours(cadence))
            if step.total_seconds() <= 0 or end < start:
                raise ValueError(
                    "output_frequency must be positive and horizon/window must end after lead_time."
                )
            predictions = []
            lead = start
            while lead <= end:
                expanded = {
                    key: value
                    for key, value in query.items()
                    if key not in {"horizon", "window"}
                }
                expanded["lead_time"] = f"{int(lead.total_seconds())}s"
                predictions.append(self.predict(inputs, expanded, output_coordinates))
                lead += step
            return torch.cat(predictions, dim=0)

        catalogue = QueryMetadata(
            self.query_catalogue,
            self.metadata["metadata_inference"]["query"]["reference_provenance"],
        )
        device = next(self.parameters()).device
        encoded_inputs = {}
        for source, source_input in inputs.items():
            values = torch.as_tensor(
                source_input["values"], dtype=torch.float32, device=device
            )
            if values.ndim != 2 or values.shape[1] != len(source_input["fields"]):
                raise ValueError(
                    f"Input {source!r} values must have shape (nodes, fields)."
                )
            if source not in self.dataset_names:
                raise KeyError(
                    f"Input source {source!r} has no geometry in this checkpoint."
                )
            expected_nodes = self.node_attributes.num_nodes[source]
            if values.shape[0] != expected_nodes:
                raise ValueError(
                    f"Input {source!r} has {values.shape[0]} nodes; its registered geometry requires {expected_nodes}."
                )
            metadata = []
            variable_ids = []
            provenance_ids = []
            unit_ids = []
            normalized = []
            for column, field_value in enumerate(source_input["fields"]):
                field = catalogue.resolve_input(field_value, source)
                if not field.get("time_invariant", False):
                    policy = self.metadata["metadata_inference"]["query"]
                    if "input_history_seconds" not in policy:
                        raise ValueError(
                            "Checkpoint metadata has no input-history bound for leakage-safe inference."
                        )
                    offset = field["time_offset_hours"]
                    earliest = -float(policy["input_history_seconds"]) / 3600
                    latest = (
                        -float(
                            policy.get("availability_lag_seconds", {}).get(source, 0)
                        )
                        / 3600
                    )
                    if offset < earliest or offset > latest:
                        raise ValueError(
                            f"Input {source!r}/{field['variable']} time_offset={offset}h is outside the "
                            f"checkpoint availability window [{earliest}h, {latest}h]."
                        )
                metadata.append(catalogue.encode(field, field["time_offset_hours"]))
                variable_ids.append(catalogue.variable_to_id[field["variable"]])
                provenance_ids.append(catalogue.provenance_to_id[field["provenance"]])
                unit_ids.append(catalogue.unit_to_id[field.get("units") or "unknown"])
                normalized.append((values[:, column] - field["mean"]) / field["stdev"])
            normalized_values = torch.stack(normalized, dim=-1)
            encoded_inputs[source] = SimpleNamespace(
                values=torch.nan_to_num(normalized_values)[None],
                metadata=torch.as_tensor(np.stack(metadata), device=device)[None],
                variable_ids=torch.tensor(variable_ids, device=device)[None],
                provenance_ids=torch.tensor(provenance_ids, device=device)[None],
                unit_ids=torch.tensor(unit_ids, device=device)[None],
                mask=torch.isfinite(normalized_values)[None],
            )

        query_value = dict(query)
        if output_coordinates is not None:
            query_value["output_coordinates"] = True
        canonical = catalogue.canonicalize(query_value)
        if canonical["grid"] is None and output_coordinates is None:
            raise ValueError(
                "Output geometry requires a registered grid or explicit output_coordinates."
            )
        output_validity_mask = query.get("output_validity_mask")
        below_ground_policy = query.get("below_ground_policy")
        if (
            canonical["level_type"] == "pressure"
            and output_validity_mask is None
            and below_ground_policy != "unmasked"
        ):
            raise ValueError(
                "Pressure output queries require output_validity_mask or explicit "
                "below_ground_policy='unmasked'; the checkpoint cannot infer future surface pressure."
            )
        query_tensor = {
            "metadata": torch.as_tensor(
                catalogue.encode(canonical, canonical["lead_time_hours"]), device=device
            )[None],
            "variable_id": torch.tensor(
                [catalogue.variable_to_id[canonical["variable"]]], device=device
            ),
            "provenance_id": torch.tensor(
                [catalogue.provenance_to_id[canonical["provenance"]]], device=device
            ),
            "unit_id": torch.tensor(
                [catalogue.unit_to_id[canonical.get("units") or "unknown"]],
                device=device,
            ),
            "grid": canonical["grid"],
        }
        coordinates = (
            None
            if output_coordinates is None
            else torch.deg2rad(
                torch.as_tensor(output_coordinates, dtype=torch.float32, device=device)
            )
        )
        area_mask = None
        if canonical["bbox"] is not None and coordinates is None:
            if canonical["grid"] not in self.decoder_graph_provider:
                raise ValueError(
                    f"Query grid {canonical['grid']!r} has no registered geometry in this checkpoint."
                )
            target_coordinates = self.node_attributes.get_coordinates(
                canonical["grid"]
            ).to(device)
            west, south, east, north = torch.deg2rad(
                torch.tensor(
                    canonical["bbox"], dtype=target_coordinates.dtype, device=device
                )
            )
            longitudes = torch.atan2(
                torch.sin(target_coordinates[:, 1]), torch.cos(target_coordinates[:, 1])
            )
            area_mask = (
                (longitudes >= west)
                & (longitudes <= east)
                & (target_coordinates[:, 0] >= south)
                & (target_coordinates[:, 0] <= north)
            )
            coordinates = target_coordinates[area_mask]
        normalized_output = self._forward_tensors(
            encoded_inputs, query_tensor, coordinates
        )
        if canonical["bbox"] is not None and output_coordinates is not None:
            target_coordinates = coordinates
            west, south, east, north = torch.deg2rad(
                torch.tensor(
                    canonical["bbox"], dtype=target_coordinates.dtype, device=device
                )
            )
            longitudes = torch.atan2(
                torch.sin(target_coordinates[:, 1]), torch.cos(target_coordinates[:, 1])
            )
            area_mask = (
                (longitudes >= west)
                & (longitudes <= east)
                & (target_coordinates[:, 0] >= south)
                & (target_coordinates[:, 0] <= north)
            )
            normalized_output = normalized_output[:, area_mask]
        if output_validity_mask is not None:
            validity = torch.as_tensor(
                output_validity_mask, dtype=torch.bool, device=device
            ).reshape(-1)
            if area_mask is not None and validity.numel() == area_mask.numel():
                validity = validity[area_mask]
            if validity.numel() != normalized_output.shape[1]:
                raise ValueError(
                    "output_validity_mask must match the requested coordinates before or after bbox selection."
                )
            normalized_output = normalized_output.masked_fill(
                ~validity[None], torch.nan
            )
        mean, stdev = catalogue.normalization(canonical)
        return normalized_output * stdev + mean
