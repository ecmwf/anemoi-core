# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
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
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from anemoi.models.data.views import FlatView
    from anemoi.models.data.views import SourceView

LOGGER = logging.getLogger(__name__)


def latlons_to_sincos(latlon: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sin(latlon), torch.cos(latlon)], dim=-1)


def sum_dataset_latents(
    dataset_latents: dict[str, torch.Tensor],
    hidden_latent: torch.Tensor,
    num_channels: int,
) -> torch.Tensor:
    """Sum encoded dataset contributions, preserving a tensor zero for empty inputs."""
    zero_latent = hidden_latent.new_zeros((hidden_latent.shape[0], num_channels))
    return sum(dataset_latents.values(), start=zero_latent)


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

    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, dynamic_graph_config: DotDict) -> None:
        """Builds the model components."""
        # Encoder data -> hidden
        self.encoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.input_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the input as it doesn't have a corresponding encoder."
                )
                continue

            encoder_config = model_config.encoders[self.dataset2encoder[dataset_name]]

            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden)],
                edge_attribute_names=encoder_config.mapper.get("sub_graph_edge_attributes"),
                **dynamic_graph_config[(dataset_name, "to", self._graph_name_hidden)],
                # src_size=self.node_attributes.num_nodes.get(dataset_name, None),
                # dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                src_size=self._graph_data[dataset_name].num_nodes,
                dst_size=self._graph_data[self._graph_name_hidden].num_nodes,
                trainable_size=encoder_config.mapper.get("trainable_size", 0),
            )

        self.encoder = torch.nn.ModuleDict()
        for encoder_name, encoder_config in model_config.encoders.items():
            encoder_in_channels_src = [self.input_dim[d] for d in self.encoder2datasets[encoder_name]]
            assert all(ch == encoder_in_channels_src[0] for ch in encoder_in_channels_src), (
                f"All datasets for encoder {encoder_name} must have the same input dimension, "
                f"but got {encoder_in_channels_src}."
            )

            self.encoder[encoder_name] = instantiate(
                encoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=encoder_in_channels_src[0],
                in_channels_dst=self.input_dim_latent,
                edge_dim=self.encoder_graph_provider[encoder_config.source_datasets[0]].edge_dim,
            )

        # Latent aggregator: combines encoder outputs before the processor
        self.latent_aggregator = instantiate(
            model_config.latent_aggregator,
            num_channels=self._get_latent_aggregator_channels(),
        )

        # Processor hidden -> hidden
        self.processor_graph_provider = create_graph_provider(
            graph=static_graph[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            edge_attribute_names=model_config.processor.get("sub_graph_edge_attributes"),
            src_size=self._graph_data[self._graph_name_hidden].num_nodes,
            dst_size=self._graph_data[self._graph_name_hidden].num_nodes,
            trainable_size=model_config.processor.get("trainable_size", 0),
            # graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            # edge_attributes=model_config.processor.get("sub_graph_edge_attributes"),
            # src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            # dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            # trainable_size=model_config.processor.get("trainable_size", 0),
        )

        self.processor = instantiate(
            model_config.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        assert self.processor.num_channels == self.latent_aggregator.hidden_dim, (
            f"Processor number of channels ({self.processor.num_channels}) must match latent aggregator output channels"
            f" ({self.latent_aggregator.hidden_dim})."
        )

        # Decoder hidden -> data
        self.decoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.target_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the output as it doesn't have a corresponding decoder."
                )
                continue

            decoder_config = model_config.decoders[self.dataset2decoder[dataset_name]]
            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(self._graph_name_hidden, "to", dataset_name)],
                edge_attribute_names=decoder_config.mapper.get("sub_graph_edge_attributes"),
                **dynamic_graph_config[(self._graph_name_hidden, "to", dataset_name)],
                # src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                # dst_size=self.node_attributes.num_nodes.get(dataset_name, None),
                src_size=self._graph_data[self._graph_name_hidden].num_nodes,
                dst_size=self._graph_data[dataset_name].num_nodes,
                trainable_size=decoder_config.mapper.get("trainable_size", 0),
            )

        self.decoder = torch.nn.ModuleDict()
        for decoder_name, decoder_config in model_config.decoders.items():
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

        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        target_features = self.decoders_target_input[self.dataset2decoder[dataset_name]]
        target_coords, target_timedeltas, x_target_latent = target_features.tensor(
            x_input_data,
            x_encoded_data,
            x_target,
            batch_size=batch_size,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
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

        # Process each dataset through its corresponding encoder
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

        for dataset_name in dataset_names:
            if dataset_name not in self.input_datasets:
                continue

            data_coords, x_data_latent, x_skip, shard_sizes_data, data_batch_sizes, data_timedeltas = (
                self._assemble_input(
                    batch[dataset_name],
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                    dataset_name=dataset_name,
                )
            )
            if data_coords.shape[0] == 0:
                continue

            x_skip_dict[dataset_name] = x_skip

            graph_batch_kwargs = (
                {"src_batch_sizes": data_batch_sizes, "dst_batch_sizes": hidden_batch_sizes}
                if data_batch_sizes is not None
                else {}
            )
            encoder_edge_attr, encoder_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                src_coords=data_coords,
                dst_coords=hidden_coordinates_batched if data_batch_sizes is not None else hidden_coordinates,
                src_timedeltas=data_timedeltas,
                model_comm_group=model_comm_group,
                **graph_batch_kwargs,
            )
            encoder_edge_attr = encoder_edge_attr.to(device=x_data_latent.device, dtype=x_data_latent.dtype)
            encoder_edge_index = encoder_edge_index.to(x_data_latent.device)
            assert encoder_edge_index.shape[1] == encoder_edge_attr.shape[0], (
                f"Encoder edge_index shape {list(encoder_edge_index.shape)} does not match "
                f"edge_attr shape {list(encoder_edge_attr.shape)} for dataset {dataset_name}."
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data,  # None if not sharded (in_out_sharded=False)
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            encoder_name = self.dataset2encoder[dataset_name]
            x_data_latent, x_latent = self.encoder[encoder_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )
            x_data_latent_dict[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents
        # x_latent = sum_dataset_latents(dataset_latents, x_hidden_latent, self.num_channels)
        x_latent = self.latent_aggregator(dataset_latents)

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

    def _get_latent_aggregator_channels(self) -> dict[str, int]:
        """Return encoder output widths by dataset."""
        return {
            dataset_name: self.encoder[self.dataset2encoder[dataset_name]].hidden_dim
            for dataset_name in self.input_datasets
        }

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes
