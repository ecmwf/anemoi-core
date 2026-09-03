# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.graphs.create import HeteroData
from anemoi.models.data.batch import Batch
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph_fusion import FusableSource
from anemoi.models.distributed.graph_fusion import build_fused_source_index
from anemoi.models.distributed.graph_fusion import fuse_encoder_edges
from anemoi.models.distributed.graph_fusion import fuse_source_features
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.utils import model_is_distributed
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import BaseGraphModel
from anemoi.models.models.base import PROJECTING_FUSING_STRATEGIES
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from anemoi.models.data.flat import FlatView
    from anemoi.models.data.views import SourceView

LOGGER = logging.getLogger(__name__)


def latlons_to_sincos(latlon: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sin(latlon), torch.cos(latlon)], dim=-1)


def _format_dims(dims: dict[str, int]) -> str:
    """Helper function. Prints one dataset per line, widths aligned, widest first so the odd one out stands out."""
    pad = max(len(name) for name in dims)
    ordered = sorted(dims.items(), key=lambda item: (-item[1], item[0]))
    return "\n".join(f"      {name:<{pad}}  {width}" for name, width in ordered)


def _mismatched_input_dims_message(encoder_name: str, in_dims: dict[str, int], strategy: str) -> str:
    """Explain why an unprojected encoder cannot take source datasets of differing widths."""
    return (
        f"Encoder '{encoder_name}' has dataset_fusing_strategy '{strategy}', which passes every source "
        f"dataset through the same encoder weights but builds no per-dataset projections, so all "
        f"{len(in_dims)} of its source datasets must have the same input width. Got:\n"
        f"{_format_dims(in_dims)}\n"
        "    These are assembled encoder input widths (variables + coordinates + trainable node "
        "parameters + any timedelta node features), not raw variable counts.\n"
        "    To share one encoder across datasets of differing widths, set\n"
        f"        model.encoders.{encoder_name}.dataset_fusing_strategy: 'sequential'\n"
        f"    which inserts a thin linear projection per dataset onto a common width "
        f"(default max = {max(in_dims.values())}, override with "
        f"model.encoders.{encoder_name}.fusion_projection_dim).\n"
        "    Use 'joint' instead to encode all sources in a single pass, or give these datasets "
        "separate encoders if they should not share weights."
    )


def _mismatched_edge_dims_message(encoder_name: str, edge_dims: dict[str, int]) -> str:
    """Explain a per-source edge-dimension mismatch, which is a graph-config problem."""
    return (
        f"Encoder '{encoder_name}' has source datasets whose encoder graphs supply different edge "
        f"feature widths, so they cannot share one mapper. Got:\n"
        f"{_format_dims(edge_dims)}\n"
        "    Edge width comes from the graph, not the model: a static (gridded) source contributes "
        "the attributes named in the mapper's sub_graph_edge_attributes plus trainable_size, while a "
        "dynamic (observation) source contributes every attribute declared on its edges and no "
        "trainable parameters.\n"
        "    Align the 'attributes' of each source's '<dataset> -> hidden' edge in the graph config, "
        "or give these datasets separate encoders."
    )


@dataclass(frozen=True)
class EncoderSource:
    """One source dataset, assembled and paired with its encoder edges.

    Produced by _prepare_encoder_source and consumed by the fusing strategies, so that
    assembling a source is done identically however it is later encoded.
    """

    dataset_name: str
    x_data_latent: Tensor
    x_skip: Tensor | None
    coordinates: Tensor
    batch_sizes: tuple[int, ...] | None
    shard_sizes: ShardSizes
    edge_attr: Tensor
    edge_index: Tensor
    edge_shard_sizes: ShardSizes
    shard_info: BipartiteGraphShardInfo


class AnemoiModelEncProcDec(BaseGraphModel):
    """Message passing graph neural network."""

    def _configure_dynamic_node_attributes(self, dynamic_node_config: DotDict) -> None:
        """Instantiate runtime timedelta encoders for dynamic dataset nodes."""
        timedelta_target = "anemoi.graphs.nodes.attributes.Timedeltas"
        for dataset_name, node_config in dynamic_node_config.items():
            runtime_attributes = {}
            for attribute_name, attribute_config in (node_config.get("attributes") or {}).items():
                if attribute_config.get("_target_") != timedelta_target:
                    continue
                runtime_attributes[attribute_name] = instantiate(attribute_config)

            if runtime_attributes:
                self.dynamic_node_attributes[dataset_name] = runtime_attributes
                self.dynamic_node_attribute_dims[dataset_name] = sum(
                    attribute.ndim for attribute in runtime_attributes.values()
                )

    def _encode_dynamic_node_attributes(self, dataset_name: str, x_flat: "FlatView") -> torch.Tensor | None:
        """Encode configured per-node runtime attributes."""
        attribute_builders = self.dynamic_node_attributes.get(dataset_name)
        if not attribute_builders:
            return None
        if x_flat.timedeltas is None:
            raise ValueError(
                f"Dataset {dataset_name!r} configures timedelta node attributes, "
                "but the batch does not provide timedeltas."
            )
        if x_flat.timedeltas.shape[0] != x_flat.coordinates.shape[0]:
            raise ValueError(
                f"Dataset {dataset_name!r} has {x_flat.timedeltas.shape[0]} timedeltas "
                f"for {x_flat.coordinates.shape[0]} coordinates."
            )

        features = [attribute.compute(x_flat.timedeltas) for attribute in attribute_builders.values()]
        return torch.cat(features, dim=-1)

    def _encoder_projects_sources(self, encoder_name: str) -> bool:
        """Whether this encoder maps its source datasets through per-dataset projections.

        Only multi-source encoders using a real fusing strategy do; single-source encoders keep
        feeding the mapper their assembled input directly, so their weights are unchanged.
        """
        return (
            len(self.encoder2datasets[encoder_name]) > 1
            and self.encoder_fusing_strategy[encoder_name] in PROJECTING_FUSING_STRATEGIES
        )

    def _project_source(self, encoder_name: str, dataset_name: str, x_data_latent: Tensor) -> Tensor:
        """Apply this encoder's thin projection for one source dataset, if it has one."""
        if encoder_name not in self.encoder_src_projection:
            return x_data_latent
        return self.encoder_src_projection[encoder_name][dataset_name](x_data_latent)

    def _build_encoding_networks(self, encoders_config: DotDict) -> None:
        """Instantiate one mapper per encoder, plus thin source projections where needed.

        Requires self.encoder_graph_provider to be populated, since an encoder's edge
        dimension comes from its source datasets' graph providers.
        """
        # Per-dataset "thin" projections onto a shared width, for multi-source encoders whose
        # source datasets differ in input dimension. Keyed [encoder_name][dataset_name].
        self.encoder_src_projection = torch.nn.ModuleDict()

        self.encoder = torch.nn.ModuleDict()
        for encoder_name, encoder_config in encoders_config.items():
            in_dims = {d: self.input_dim[d] for d in self.encoder2datasets[encoder_name]}
            edge_dims = {d: self.encoder_graph_provider[d].edge_dim for d in self.encoder2datasets[encoder_name]}
            if len(set(edge_dims.values())) > 1:
                raise ValueError(_mismatched_edge_dims_message(encoder_name, edge_dims))

            projects = self._encoder_projects_sources(encoder_name)
            if projects:
                in_channels_src = encoder_config.get("fusion_projection_dim") or max(in_dims.values())
            else:
                if len(set(in_dims.values())) > 1:
                    raise ValueError(
                        _mismatched_input_dims_message(
                            encoder_name, in_dims, self.encoder_fusing_strategy[encoder_name]
                        )
                    )
                in_channels_src = next(iter(in_dims.values()))

            self.encoder[encoder_name] = instantiate(
                encoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=in_channels_src,
                in_channels_dst=self.input_dim_latent,
                edge_dim=next(iter(edge_dims.values())),
            )

            if projects:
                Linear = self.encoder[encoder_name].layer_factory.Linear
                self.encoder_src_projection[encoder_name] = torch.nn.ModuleDict(
                    {d: Linear(dim, in_channels_src) for d, dim in in_dims.items()}
                )

    def _build_encoding_graphproviders(
        self,
        encoders_config: DotDict,
        static_graph: HeteroData,
        dynamic_graph_config: DotDict
    ) -> None:
        """Builds the graph providers for the encoding networks."""

        self.encoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.input_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the input as it doesn't have a corresponding encoder."
                )
                continue

            encoder_config = encoders_config[self.dataset2encoder[dataset_name]]

            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=static_graph[(dataset_name, "to", self._graph_name_hidden)],
                edge_attribute_names=encoder_config.mapper.get("sub_graph_edge_attributes"),
                **dynamic_graph_config[(dataset_name, "to", self._graph_name_hidden)],
                src_size=static_graph[dataset_name].num_nodes,
                dst_size=static_graph[self._graph_name_hidden].num_nodes,
                trainable_size=encoder_config.mapper.get("trainable_size", 0),
            )

    def _build_processing_graphproviders(
        self,
        processor_config: DotDict,
        static_graph: HeteroData,
        dynamic_graph_config: DotDict
    ) -> None:
        """Builds the graph providers for the processor network."""

        self.processor_graph_provider = create_graph_provider(
            graph=static_graph[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            edge_attribute_names=processor_config.get("sub_graph_edge_attributes"),
            **dynamic_graph_config[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_size=static_graph[self._graph_name_hidden].num_nodes,
            dst_size=static_graph[self._graph_name_hidden].num_nodes,
            trainable_size=processor_config.get("trainable_size", 0),
        )
    
    def _build_processing_networks(self, processor_config: DotDict) -> None:
        self.processor = instantiate(
            processor_config,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        assert self.processor.num_channels == self.latent_aggregator.hidden_dim, (
            f"Processor number of channels ({self.processor.num_channels}) must match latent aggregator output channels"
            f" ({self.latent_aggregator.hidden_dim})."
        )

    def _build_decoding_graphproviders(
        self,
        decoders_config: DotDict,
        static_graph: HeteroData,
        dynamic_graph_config: DotDict
    ) -> None:
        """Builds the graph providers for the decoding network."""
        self.decoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.target_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the output as it doesn't have a corresponding decoder."
                )
                continue

            decoder_config = decoders_config[self.dataset2decoder[dataset_name]]
            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=static_graph[(self._graph_name_hidden, "to", dataset_name)],
                edge_attribute_names=decoder_config.mapper.get("sub_graph_edge_attributes"),
                **dynamic_graph_config[(self._graph_name_hidden, "to", dataset_name)],
                src_size=static_graph[self._graph_name_hidden].num_nodes,
                dst_size=static_graph[dataset_name].num_nodes,
                trainable_size=decoder_config.mapper.get("trainable_size", 0),
            )

    def _build_decoding_networks(self, decoders_config: DotDict) -> None:
        """Builds the decoding networks."""
        self.decoder = torch.nn.ModuleDict()
        for decoder_name, decoder_config in decoders_config.items():
            decoder_in_channels_dst = [self.target_dim[d] for d in self.decoder2datasets[decoder_name]]
            assert all(ch == decoder_in_channels_dst[0] for ch in decoder_in_channels_dst), (
                f"All datasets for decoder {decoder_name} must have the same target dimension, "
                f"but got {decoder_in_channels_dst}."
            )
            decoder_output_channels_dst = [self.output_dim[d] for d in self.decoder2datasets[decoder_name]]
            assert all(ch == decoder_output_channels_dst[0] for ch in decoder_output_channels_dst), (
                f"All datasets for decoder {decoder_name} must have the same output dimension, "
                f"but got {decoder_output_channels_dst}."
            )

            self.decoder[decoder_name] = instantiate(
                decoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.processor.num_channels,
                in_channels_dst=decoder_in_channels_dst[0],
                out_channels_dst=decoder_output_channels_dst[0],
                edge_dim=self.decoder_graph_provider[decoder_config.target_datasets[0]].edge_dim,
            )

    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, dynamic_graph_config: DotDict) -> None:
        """Builds the model components."""
        self._build_encoding_graphproviders(model_config.encoders, static_graph, dynamic_graph_config)
        self._build_encoding_networks(model_config.encoders)
        
        self._build_latent_aggregator(model_config.latent_aggregator)
        self._build_processing_graphproviders(model_config.processor, static_graph, dynamic_graph_config)
        self._build_processing_networks(model_config.processor)

        self._build_decoding_graphproviders(model_config.decoders, static_graph, dynamic_graph_config)
        self._build_decoding_networks(model_config.decoders)

    def _assemble_input(
        self,
        x: "SourceView",
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, "SourceView", ShardSizes, tuple[int, ...] | None, torch.Tensor | None]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        x_flat: "FlatView" = x.flatten()  # flatten data to (nodes, features)
        grid_shard_sizes = x_flat.shard_sizes

        if dataset_name in self.residual:
            x_skip = self.residual[dataset_name](
                x.data,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                n_step_output=self.n_step_output,
            )
        else:
            x_skip = None

        inputs = [x_flat.data, latlons_to_sincos(x_flat.coordinates)]
        dynamic_node_attributes = self._encode_dynamic_node_attributes(dataset_name, x_flat)
        if dynamic_node_attributes is not None:
            inputs.append(dynamic_node_attributes.to(device=x_flat.data.device, dtype=x_flat.data.dtype))

        if dataset_name in self.node_attributes:
            trainable_parameters = self.node_attributes(dataset_name, batch_size=batch_size).to(x_flat.data.device)
            if grid_shard_sizes is not None:
                trainable_parameters = shard_tensor(trainable_parameters, 0, grid_shard_sizes, model_comm_group)

            inputs.append(trainable_parameters)

        x_data_latent = torch.cat(inputs, dim=-1)

        # gather full coordinates for correct graph building in the encoder
        coordinates = x_flat.coordinates
        timedeltas = x_flat.timedeltas
        if grid_shard_sizes is not None:
            coordinates = gather_tensor(coordinates, dim=0, sizes=grid_shard_sizes, mgroup=model_comm_group)
            if timedeltas is not None:
                timedeltas = gather_tensor(timedeltas, dim=0, sizes=grid_shard_sizes, mgroup=model_comm_group)

        return coordinates, x_data_latent, x_skip, grid_shard_sizes, x_flat.batch_sizes, timedeltas

    def _assemble_target(
        self,
        x_input_data: "SourceView",
        x_encoded_data: Tensor | None,
        x_target: "SourceView",
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Assemble the decoder destination features for a single dataset.

        Concatenates the feature blocks listed in ``decoders_target_input`` for this dataset's
        decoder into the per-node vector fed to the decoder as ``x_dst``.

        Returns
        -------
        target_coords : Tensor
            Coordinates of the target nodes, shape (N, 2) for lat/lon.
        x_target_latent : Tensor
            Latent features for the target nodes, shape (N, F) where F is the concatenated feature dimension.
        grid_shard_sizes : ShardSizes
            Shard sizes for the target nodes, or None if the dataset is not sharded.
        data_batch_sizes : tuple[int, ...] | None
            Batch sizes for the target nodes, or None if the dataset is not batched.
        target_timedeltas : Tensor | None
            Timedeltas for the target nodes, or None if the dataset does not have timedeltas (gridded).
        """
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        x_target_flat: "FlatView" = x_target.flatten()
        grid_shard_sizes = x_target_flat.shard_sizes

        target_features = self.decoders_target_input[self.dataset2decoder[dataset_name]]
        x_target_latent = target_features.tensor(
            x_input_data,
            x_encoded_data,
            x_target_flat,
            batch_size=batch_size,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
        )

        target_coords = x_target_flat.coordinates
        target_timedeltas = x_target_flat.timedeltas
        if grid_shard_sizes is not None:
            target_coords = gather_tensor(target_coords, dim=0, sizes=grid_shard_sizes, mgroup=model_comm_group)
            if target_timedeltas is not None:
                target_timedeltas = gather_tensor(
                    target_timedeltas, dim=0, sizes=grid_shard_sizes, mgroup=model_comm_group
                )

        # Fail fast with a clear message if the decoder destination features do not line up with the
        # target nodes. Only valid when unsharded (under sharding the composite gathers
        # target_coords to full size while x_target_latent stays local).
        if grid_shard_sizes is None:
            assert x_target_latent.shape[0] == target_coords.shape[0], (
                f"Decoder x_dst rows ({x_target_latent.shape[0]}) != target node count "
                f"({target_coords.shape[0]}) for dataset '{dataset_name}'. This usually means an "
                f"'encoded_data' target feature is used for a dataset whose input and target node "
                f"sets differ (e.g. tabular observations); use ['coordinates', 'target_forcings'] "
                f"instead."
            )

        return target_coords, x_target_latent, grid_shard_sizes, x_target.flatten().batch_sizes, target_timedeltas

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor | None,
        target: "SourceView",
        dtype: torch.dtype,
        dataset_name: str,
    ) -> "SourceView":
        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"

        # clone to make sure we return a copy, not a view
        # a view cannot be modified in-place by the residual add below without breaking autograd!
        pred = target.unflatten(x_out)
        output_names = self.data_indices[dataset_name].model.output.ordered_names
        output_positions = [self.data_indices[dataset_name].name_to_index[name] for name in output_names]
        output_statistics = {name: values[output_positions] for name, values in self.statistics[dataset_name].items()}
        pred = pred.clone(variables=output_names, statistics=output_statistics)

        if x_skip is not None:
            assert (
                x_skip.ndim == 5
            ), f"Residual must be (batch, time, ensemble, grid, variables), but got shape {x_skip.shape}"
            assert (
                x_skip.shape[1] == pred.data.shape[1]
            ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({pred.data.shape[1]})."
            new_data = pred.data.clone()
            new_data[..., self._internal_output_idx[dataset_name]] += x_skip[
                ..., self._internal_input_idx[dataset_name]
            ]
            pred = pred.clone(data=new_data)

        pred = self.boundings[dataset_name](pred)

        return pred

    def _assert_valid_sharding(
        self,
        batch_size: int,
        ensemble_size: int,
        in_out_sharded: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        assert not (
            in_out_sharded and model_comm_group is None
        ), "If input is sharded, model_comm_group must be provided."

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

            assert (
                model_comm_group.size() == 1 or ensemble_size == 1
            ), "Ensemble size per device must be 1 when model is sharded across GPUs"

    def _prepare_encoder_source(
        self,
        x: "SourceView",
        *,
        dataset_name: str,
        batch_size: int,
        hidden_coordinates: Tensor,
        hidden_coordinates_batched: Tensor,
        hidden_batch_sizes: tuple[int, ...],
        shard_sizes_hidden: ShardSizes,
        model_comm_group: ProcessGroup | None = None,
    ) -> EncoderSource | None:
        """Assemble one source dataset and its encoder edges.

        Returns ``None`` when the dataset contributes no nodes to this batch, in which case the
        caller skips it entirely (no skip connection, no latent).
        """
        data_coords, x_data_latent, x_skip, shard_sizes_data, data_batch_sizes, data_timedeltas = self._assemble_input(
            x,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
        )
        if data_coords.shape[0] == 0:
            return None

        graph_batch_kwargs = (
            {"src_batch_sizes": data_batch_sizes, "dst_batch_sizes": hidden_batch_sizes}
            if data_batch_sizes is not None
            else {}
        )
        edge_attr, edge_index, edge_shard_sizes = self.encoder_graph_provider[dataset_name].get_edges(
            batch_size=batch_size,
            src_coords=data_coords,
            dst_coords=hidden_coordinates_batched if data_batch_sizes is not None else hidden_coordinates,
            src_timedeltas=data_timedeltas,
            model_comm_group=model_comm_group,
            **graph_batch_kwargs,
        )
        edge_attr = edge_attr.to(device=x_data_latent.device, dtype=x_data_latent.dtype)
        edge_index = edge_index.to(x_data_latent.device)
        assert edge_index.shape[1] == edge_attr.shape[0], (
            f"Encoder edge_index shape {list(edge_index.shape)} does not match "
            f"edge_attr shape {list(edge_attr.shape)} for dataset {dataset_name}."
        )

        return EncoderSource(
            dataset_name=dataset_name,
            x_data_latent=x_data_latent,
            x_skip=x_skip,
            coordinates=data_coords,
            batch_sizes=data_batch_sizes,
            shard_sizes=shard_sizes_data,
            edge_attr=edge_attr,
            edge_index=edge_index,
            edge_shard_sizes=edge_shard_sizes,
            shard_info=BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data,  # None if not sharded (in_out_sharded=False)
                dst_nodes=shard_sizes_hidden,
                edges=edge_shard_sizes,
            ),
        )

    def _encode_sources(
        self,
        encoder_name: str,
        sources: list[EncoderSource],
        *,
        x_hidden_latent: Tensor,
        x_data_latent_dict: dict[str, Tensor],
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
    ) -> dict[str, Tensor]:
        """Encode one encoder's source datasets, returning its latents keyed by latent key.

        With ``sequential`` (and with ``none``) every source dataset gets its own
        pass through the shared encoder against the same unmodified hidden latent, and yields its
        own latent; the latent aggregator combines them.

        With ``joint`` and more than one source dataset present, all sources are merged into one
        bipartite graph and encoded in a single pass, yielding one already-fused latent (per encoder).
        """
        if self.encoder_fusing_strategy[encoder_name] == "joint" and len(sources) > 1:
            return self._encode_joint(
                encoder_name,
                sources,
                x_hidden_latent=x_hidden_latent,
                x_data_latent_dict=x_data_latent_dict,
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

        projects = self._encoder_projects_sources(encoder_name)
        latents: dict[str, Tensor] = {}

        for source in sources:
            x_src = self._project_source(encoder_name, source.dataset_name, source.x_data_latent)
            x_data_latent, x_latent = self.encoder[encoder_name](
                (x_src, x_hidden_latent),
                batch_size=batch_size,
                shard_info=source.shard_info,
                edge_attr=source.edge_attr,
                edge_index=source.edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )
            # Decoder target features expect the *assembled* (pre-projection) width, so a
            # projecting encoder reports its input rather than the mapper's src passthrough.
            x_data_latent_dict[source.dataset_name] = source.x_data_latent if projects else x_data_latent
            latents[self._latent_key(encoder_name, source.dataset_name)] = x_latent

        return latents

    def _encode_joint(
        self,
        encoder_name: str,
        sources: list[EncoderSource],
        *,
        x_hidden_latent: Tensor,
        x_data_latent_dict: dict[str, Tensor],
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
    ) -> dict[str, Tensor]:
        """Encode all of one encoder's source datasets in a single fused pass.

        The source nodes of every dataset are merged into one index space and their edge sets are
        concatenated and re-sorted by destination, so the mapper sees a single bipartite graph whose
        source side is the union of all the datasets.
        """
        tabular = {source.dataset_name: source.batch_sizes is not None for source in sources}
        if len(set(tabular.values())) > 1:
            msg = (
                f"Encoder '{encoder_name}' cannot jointly encode gridded and tabular sources "
                f"({tabular}): they are built against different destination index spaces. "
                "Use dataset_fusing_strategy: 'sequential' instead."
            )
            raise ValueError(msg)

        rank = torch.distributed.get_rank(group=model_comm_group) if model_is_distributed(model_comm_group) else 0
        world_size = model_comm_group.size() if model_is_distributed(model_comm_group) else 1

        fusable = [
            FusableSource(
                name=source.dataset_name,
                features=self._project_source(encoder_name, source.dataset_name, source.x_data_latent),
                shard_sizes=source.shard_sizes,
                batch_sizes=source.batch_sizes,
                edge_attr=source.edge_attr,
                edge_index=source.edge_index,
                edge_shard_sizes=source.edge_shard_sizes,
            )
            for source in sources
        ]

        index = build_fused_source_index(fusable, batch_size=batch_size, rank=rank, world_size=world_size)

        # The heads shard strategy reshapes nodes as "(batch grid) -> batch heads grid vars", which
        # requires the same node count in every batch element.
        if getattr(self.encoder[encoder_name], "shard_strategy", None) == "heads":
            if len(set(index.merged_batch_sizes)) > 1:
                msg = (
                    f"Encoder '{encoder_name}' uses shard_strategy 'heads', which needs a uniform "
                    f"node count per batch element, but the merged sources give "
                    f"{index.merged_batch_sizes}. Use shard_strategy 'edges' for joint fusion of "
                    "ragged sources."
                )
                raise ValueError(msg)

        x_src = fuse_source_features(fusable, index)

        # Every source of an encoder is built against the same hidden destination nodes, so the
        # destination shard metadata is shared and can be taken from any of them.
        shard_sizes_hidden = sources[0].shard_info.dst_nodes
        num_dst = sum(shard_sizes_hidden) if shard_sizes_hidden is not None else x_hidden_latent.shape[0]

        edge_attr, edge_index, edge_shard_sizes = fuse_encoder_edges(fusable, index, num_dst=num_dst)

        _, x_latent = self.encoder[encoder_name](
            (x_src, x_hidden_latent),
            batch_size=batch_size,
            shard_info=BipartiteGraphShardInfo(
                src_nodes=index.merged_shard_sizes,
                dst_nodes=shard_sizes_hidden,
                edges=edge_shard_sizes,
            ),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        # Report each dataset's assembled (pre-projection) source tensor, which is the width the
        # decoder target features expect
        for source in sources:
            x_data_latent_dict[source.dataset_name] = source.x_data_latent

        return {self._latent_key(encoder_name, sources[0].dataset_name): x_latent}

    def forward(
        self,
        batch: Batch,
        target: Optional[Batch] = None,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        batch : Batch
            Typed batch envelope. ``batch.data`` carries the per-dataset input
            tensors; ``batch.coordinates`` carries the per-dataset coordinate
            tensors used by dynamic graph providers / node attributes. Per-dataset
            grid sharding is carried by the batch and read through the source
            views (``view.flatten().shard_sizes``).
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        dataset_names = list(batch.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(batch, 0)
        ensemble_size = self._get_consistent_dim(batch, 2)

        in_out_sharded = self._resolve_in_out_sharded(batch)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Latents produced by the encoders, keyed by latent key (dataset name today; a joint
        # encoder will contribute a single entry under its own name)
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}

        # Prepare hidden latent
        hidden_coordinates = self._hidden_coordinates().to(batch.device)
        hidden_coordinates_batched = einops.repeat(hidden_coordinates, "n f -> (repeat n) f", repeat=batch_size)
        hidden_batch_sizes = (hidden_coordinates.shape[0],) * batch_size
        x_hidden_latent = latlons_to_sincos(hidden_coordinates)
        x_hidden_latent = einops.repeat(x_hidden_latent, "n f -> (repeat n) f", repeat=batch_size)

        hidden_trainable_parameters = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        if hidden_trainable_parameters is not None:
            hidden_trainable_parameters = hidden_trainable_parameters.to(x_hidden_latent.device)
            x_hidden_latent = torch.cat([x_hidden_latent, hidden_trainable_parameters], dim=-1)

        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        # Encoders are iterated in config order, and their source datasets in the order listed
        # under `source_datasets`, so the fusing order is the one the user wrote down.
        for encoder_name, source_datasets in self.encoder2datasets.items():
            sources = []
            for dataset_name in source_datasets:
                if dataset_name not in batch:
                    continue

                source = self._prepare_encoder_source(
                    batch[dataset_name],
                    dataset_name=dataset_name,
                    batch_size=batch_size,
                    hidden_coordinates=hidden_coordinates,
                    hidden_coordinates_batched=hidden_coordinates_batched,
                    hidden_batch_sizes=hidden_batch_sizes,
                    shard_sizes_hidden=shard_sizes_hidden,
                    model_comm_group=model_comm_group,
                )
                if source is None:  # no data points for this dataset in this batch
                    continue

                x_skip_dict[dataset_name] = source.x_skip
                sources.append(source)

            if not sources:
                continue

            dataset_latents.update(
                self._encode_sources(
                    encoder_name,
                    sources,
                    x_hidden_latent=x_hidden_latent,
                    x_data_latent_dict=x_data_latent_dict,
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )
            )

        # Combine all encoded latents
        x_latent = self.latent_aggregator(x_hidden_latent, dataset_latents)

        # Processor
        processor_edge_attr, processor_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            src_coords=hidden_coordinates,
            dst_coords=hidden_coordinates,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        processor_edge_attr = processor_edge_attr.to(x_latent.device)
        processor_edge_index = processor_edge_index.to(x_latent.device)

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
        )

        # Latent skip connection
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        # Decoder
        x_out_dict = {}
        for dataset_name in self.target_datasets:
            target_coords, target_data_latent, shard_sizes_data, data_batch_sizes, data_timedeltas = (
                self._assemble_target(
                    batch[dataset_name],
                    x_data_latent_dict.get(dataset_name, None),
                    target[dataset_name],
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                    dataset_name=dataset_name,
                )
            )

            if target_coords.numel() == 0:
                LOGGER.debug(
                    "No data points for dataset %s in the batch (data_coords.shape = %s), "
                    + "will decode to a size-zero tensor ...",
                    dataset_name,
                    list(target_coords.shape),
                )

            graph_batch_kwargs = (
                {"src_batch_sizes": hidden_batch_sizes, "dst_batch_sizes": data_batch_sizes}
                if data_batch_sizes is not None
                else {}
            )
            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                src_coords=hidden_coordinates_batched if data_batch_sizes is not None else hidden_coordinates,
                dst_coords=target_coords,
                dst_timedeltas=data_timedeltas,
                model_comm_group=model_comm_group,
                **graph_batch_kwargs,
            )
            decoder_edge_attr = decoder_edge_attr.to(device=x_latent.device, dtype=x_latent.dtype)
            decoder_edge_index = decoder_edge_index.to(x_latent.device)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            decoder_name = self.dataset2decoder[dataset_name]
            x_out = self.decoder[decoder_name](
                (x_latent_proc, target_data_latent),
                batch_size=batch_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip_dict.get(dataset_name, None),
                target[dataset_name],
                dtype=x_out.dtype,
                dataset_name=dataset_name,
            )

        # Preserve the reconstructed output metadata rather than the decoder
        # conditioning metadata carried by target.
        output = target
        for dataset_name in x_out_dict.keys():
            do_coords_match = target[dataset_name].coordinates == x_out_dict[dataset_name].coordinates
            assert (
                do_coords_match if isinstance(do_coords_match, bool) else torch.all(do_coords_match)
            ), "Target and output coordinates must match."
            output = output.update_source(dataset_name, x_out_dict[dataset_name])

        return output

    def _latent_key(self, encoder_name: str, dataset_name: str) -> str:
        """Key under which one encoded latent is handed to the latent aggregator.

        Strategies that encode each source dataset separately contribute one latent per dataset
        and so key by dataset name. ``joint`` encodes all its sources in a single pass and
        contributes one already-fused latent, keyed by the encoder name.
        """
        if self.encoder_fusing_strategy[encoder_name] == "joint":
            return encoder_name
        return dataset_name

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes
