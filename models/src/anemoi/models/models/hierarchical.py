# (C) Copyright 2025-2026 Anemoi contributors.
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
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.graphs.create import HeteroData
from anemoi.models.data.batch import Batch
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import BaseGraphProvider
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.models.encoder_processor_decoder import _edges_or_none
from anemoi.models.models.encoder_processor_decoder import latlons_to_sincos
from anemoi.models.utils.config import COORDS_DIM
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from anemoi.models.data.sources.base import Template

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecHierarchical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network.

    The encoders map the datasets onto the first hidden level (``hidden_nodes_name[0]``). The
    latent then goes up the hidden levels through the upscale mappers, is processed at the last
    level, and comes back down through the downscale mappers, with a skip connection at every
    level, before the decoders map it onto the datasets. The latent width doubles at each level up.
    With ``enable_hierarchical_level_processing``, every level but the last also has a processor
    on the way up and on the way down.
    """

    @property
    def _hidden_names(self) -> list[str]:
        """Names of the hidden levels, from the first (connected to the datasets) to the last."""
        return self._as_hidden_node_names(self._graph_name_hidden)

    def _build_graph_provider(
        self,
        src_name: str,
        dst_name: str,
        layer_config: DotDict,
        static_graph: HeteroData,
        dynamic_graph_config: DotDict,
    ) -> BaseGraphProvider:
        """Build the graph provider for the edges from ``src_name`` to ``dst_name``."""
        edge_type = (src_name, "to", dst_name)
        return create_graph_provider(
            graph=_edges_or_none(static_graph, edge_type),
            edge_attribute_names=layer_config.get("sub_graph_edge_attributes"),
            **dynamic_graph_config.get(edge_type, {}),
            src_size=static_graph[src_name].num_nodes,
            dst_size=static_graph[dst_name].num_nodes,
            trainable_size=layer_config.get("trainable_size", 0),
        )

    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, dynamic_graph_config: DotDict) -> None:
        """Builds the model components."""
        hidden_names = self._hidden_names
        self.num_hidden = len(hidden_names)

        # Encoders: data -> first hidden level
        self.encoder_graph_provider = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.input_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the input as it doesn't have a corresponding encoder."
                )
                continue

            encoder_config = model_config.encoders[self.dataset2encoder[dataset_name]]
            self.encoder_graph_provider[dataset_name] = self._build_graph_provider(
                dataset_name, hidden_names[0], encoder_config.mapper, static_graph, dynamic_graph_config
            )
        self._build_encoding_networks(model_config.encoders)

        # Latent aggregator: combines encoder outputs at the first hidden level
        self._build_latent_aggregator(model_config.latent_aggregator)

        # self.hidden_dims is the dimensionality of features at each depth
        self.hidden_dims = {hidden: self.latent_aggregator.hidden_dim * (2**i) for i, hidden in enumerate(hidden_names)}

        # Level processors
        self.level_process = model_config.enable_hierarchical_level_processing
        if self.level_process:
            self.down_level_processor = nn.ModuleDict()
            self.down_level_processor_graph_providers = nn.ModuleDict()
            self.up_level_processor = nn.ModuleDict()
            self.up_level_processor_graph_providers = nn.ModuleDict()
            for nodes_name in hidden_names[:-1]:
                self.down_level_processor_graph_providers[nodes_name] = self._build_graph_provider(
                    nodes_name, nodes_name, model_config.processor, static_graph, dynamic_graph_config
                )
                self.down_level_processor[nodes_name] = instantiate(
                    model_config.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_name],
                    edge_dim=self.down_level_processor_graph_providers[nodes_name].edge_dim,
                    num_layers=model_config.level_process_num_layers,
                )

                self.up_level_processor_graph_providers[nodes_name] = self._build_graph_provider(
                    nodes_name, nodes_name, model_config.processor, static_graph, dynamic_graph_config
                )
                self.up_level_processor[nodes_name] = instantiate(
                    model_config.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_name],
                    edge_dim=self.up_level_processor_graph_providers[nodes_name].edge_dim,
                    num_layers=model_config.level_process_num_layers,
                )

        # Main processor at the last hidden level
        self.processor_graph_provider = self._build_graph_provider(
            hidden_names[-1], hidden_names[-1], model_config.processor, static_graph, dynamic_graph_config
        )
        self.processor = instantiate(
            model_config.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.hidden_dims[hidden_names[-1]],
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        # Upscale: hidden level i -> i + 1, keyed by the source level
        self.upscale = nn.ModuleDict()
        self.upscale_graph_providers = nn.ModuleDict()
        for src_nodes_name, dst_nodes_name in zip(hidden_names[:-1], hidden_names[1:]):
            self.upscale_graph_providers[src_nodes_name] = self._build_graph_provider(
                src_nodes_name, dst_nodes_name, model_config.upscale_mapper, static_graph, dynamic_graph_config
            )
            self.upscale[src_nodes_name] = instantiate(
                model_config.upscale_mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=COORDS_DIM + self.node_attributes.num_trainable_parameters.get(dst_nodes_name, 0),
                num_channels=self.hidden_dims[dst_nodes_name],
                edge_dim=self.upscale_graph_providers[src_nodes_name].edge_dim,
            )

        # Downscale: hidden level i + 1 -> i, keyed by the destination level
        self.downscale = nn.ModuleDict()
        self.downscale_graph_providers = nn.ModuleDict()
        for dst_nodes_name, src_nodes_name in zip(hidden_names[:-1], hidden_names[1:]):
            self.downscale_graph_providers[dst_nodes_name] = self._build_graph_provider(
                src_nodes_name, dst_nodes_name, model_config.downscale_mapper, static_graph, dynamic_graph_config
            )
            self.downscale[dst_nodes_name] = instantiate(
                model_config.downscale_mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.hidden_dims[dst_nodes_name],
                num_channels=self.hidden_dims[src_nodes_name],
                out_channels_dst=self.hidden_dims[dst_nodes_name],
                edge_dim=self.downscale_graph_providers[dst_nodes_name].edge_dim,
            )

        # Decoders: first hidden level -> data
        self.decoder_graph_provider = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.target_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the output as it doesn't have a corresponding decoder."
                )
                continue

            decoder_config = model_config.decoders[self.dataset2decoder[dataset_name]]
            self.decoder_graph_provider[dataset_name] = self._build_graph_provider(
                hidden_names[0], dataset_name, decoder_config.mapper, static_graph, dynamic_graph_config
            )

        self.decoder = nn.ModuleDict()
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
                in_channels_src=self.hidden_dims[hidden_names[0]],
                in_channels_dst=decoder_in_channels_dst[0],
                out_channels_dst=decoder_output_channels_dst[0],
                edge_dim=self.decoder_graph_provider[decoder_config.target_datasets[0]].edge_dim,
            )

    def _hidden_coordinates(self, hidden_name: str | None = None) -> Tensor:
        """Coordinates of a hidden level, by default the first one (the one the datasets connect to)."""
        return self._graph_data[hidden_name or self._hidden_names[0]].x

    def _hidden_edges(
        self,
        graph_provider: BaseGraphProvider,
        src_coordinates: Tensor,
        dst_coordinates: Tensor,
        x_latent: Tensor,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, ShardSizes]:
        """Edges between two hidden levels (or within one), on the device and in the dtype of ``x_latent``."""
        edge_attr, edge_index, edge_shard_sizes = graph_provider.get_edges(
            batch_size=batch_size,
            src_coords=src_coordinates,
            dst_coords=dst_coordinates,
            model_comm_group=model_comm_group,
        )
        return (
            edge_attr.to(device=x_latent.device, dtype=x_latent.dtype),
            edge_index.to(x_latent.device),
            edge_shard_sizes,
        )

    def _process_level(
        self,
        processor: nn.Module,
        graph_provider: BaseGraphProvider,
        x_latent: Tensor,
        coordinates: Tensor,
        shard_sizes: ShardSizes,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        """Run a processor on the latent of one hidden level."""
        edge_attr, edge_index, edge_shard_sizes = self._hidden_edges(
            graph_provider, coordinates, coordinates, x_latent, batch_size, model_comm_group
        )
        return processor(
            x_latent,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes, edges=edge_shard_sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )

    def forward(
        self,
        batch: Batch,
        target_forcings: Batch,
        target_template: dict[str, "Template"],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> Batch:
        """Forward pass of the model.

        Parameters
        ----------
        batch : Batch
            Input sources per dataset. Per-dataset grid sharding is carried by the sources.
        target_forcings : Batch
            Decoder conditioning: the forcing variables at the output valid times.
        target_template : dict[str, Template]
            What to predict per decoded dataset: the output variables and target nodes.
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None.
        **kwargs
            Additional model arguments, unused.

        Returns
        -------
        Batch
            Output of the model per decoded dataset (sharded if the input is sharded).
        """
        dataset_names = list(batch.keys())
        hidden_names = self._hidden_names

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = batch.batch_size
        ensemble_sizes = {dataset_name: batch[dataset_name].ensemble_size for dataset_name in dataset_names}
        if len(set(ensemble_sizes.values())) != 1:
            raise ValueError("All datasets must have the same ensemble size")
        ensemble_size = next(iter(ensemble_sizes.values()))

        in_out_sharded = self._resolve_in_out_sharded(batch)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # The graph and mappers operate on one node copy per sample and member.
        batch_size *= ensemble_size

        # Initial latent of every hidden level: its coordinates and trainable parameters, pre-sharded
        hidden_coordinates = {}
        x_hidden_latents = {}
        shard_sizes_hidden = {}
        for hidden in hidden_names:
            hidden_coordinates[hidden] = self._hidden_coordinates(hidden).to(batch.device)
            x_hidden_latent = einops.repeat(
                latlons_to_sincos(hidden_coordinates[hidden]), "n f -> (repeat n) f", repeat=batch_size
            )
            hidden_trainable_parameters = self.node_attributes(hidden, batch_size=batch_size)
            if hidden_trainable_parameters is not None:
                x_hidden_latent = torch.cat([x_hidden_latent, hidden_trainable_parameters], dim=-1)

            shard_sizes_hidden[hidden] = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
            x_hidden_latents[hidden] = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden[hidden], model_comm_group)

        # Encoders, onto the first hidden level
        first_hidden = hidden_names[0]
        first_hidden_coordinates_batched = einops.repeat(
            hidden_coordinates[first_hidden], "n f -> (repeat n) f", repeat=batch_size
        )
        first_hidden_batch_sizes = (hidden_coordinates[first_hidden].shape[0],) * batch_size

        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        for encoder_name, source_datasets in self.encoder2datasets.items():
            sources = []
            for dataset_name in source_datasets:
                if dataset_name not in batch:
                    continue

                source = self._prepare_encoder_source(
                    batch[dataset_name],
                    dataset_name=dataset_name,
                    batch_size=batch_size,
                    hidden_coordinates=hidden_coordinates[first_hidden],
                    hidden_coordinates_batched=first_hidden_coordinates_batched,
                    hidden_batch_sizes=first_hidden_batch_sizes,
                    shard_sizes_hidden=shard_sizes_hidden[first_hidden],
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
                    x_hidden_latent=x_hidden_latents[first_hidden],
                    x_data_latent_dict=x_data_latent_dict,
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )
            )

        # Combine all encoded latents at the first hidden level
        x_latent = self.latent_aggregator(x_hidden_latents[first_hidden], dataset_latents)

        # Up the hidden levels
        x_level_skip = {}
        x_encoded_latents = {}
        for src_hidden_name, dst_hidden_name in zip(hidden_names[:-1], hidden_names[1:]):
            if self.level_process:
                x_latent = self._process_level(
                    self.down_level_processor[src_hidden_name],
                    self.down_level_processor_graph_providers[src_hidden_name],
                    x_latent,
                    hidden_coordinates[src_hidden_name],
                    shard_sizes_hidden[src_hidden_name],
                    batch_size,
                    model_comm_group,
                )

            # store latents for skip connections
            x_level_skip[src_hidden_name] = x_latent

            upscale_edge_attr, upscale_edge_index, upscale_edge_shard_sizes = self._hidden_edges(
                self.upscale_graph_providers[src_hidden_name],
                hidden_coordinates[src_hidden_name],
                hidden_coordinates[dst_hidden_name],
                x_latent,
                batch_size,
                model_comm_group,
            )
            x_encoded_latents[src_hidden_name], x_latent = self.upscale[src_hidden_name](
                (x_latent, x_hidden_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden[src_hidden_name],
                    dst_nodes=shard_sizes_hidden[dst_hidden_name],
                    edges=upscale_edge_shard_sizes,
                ),
                edge_attr=upscale_edge_attr,
                edge_index=upscale_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )

        # Processor at the last hidden level
        last_hidden = hidden_names[-1]
        x_latent_proc = self._process_level(
            self.processor,
            self.processor_graph_provider,
            x_latent,
            hidden_coordinates[last_hidden],
            shard_sizes_hidden[last_hidden],
            batch_size,
            model_comm_group,
        )

        # Latent skip connection
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent
        x_latent = x_latent_proc

        # Down the hidden levels
        for dst_hidden_name, src_hidden_name in reversed(list(zip(hidden_names[:-1], hidden_names[1:]))):
            downscale_edge_attr, downscale_edge_index, downscale_edge_shard_sizes = self._hidden_edges(
                self.downscale_graph_providers[dst_hidden_name],
                hidden_coordinates[src_hidden_name],
                hidden_coordinates[dst_hidden_name],
                x_latent,
                batch_size,
                model_comm_group,
            )
            x_latent = self.downscale[dst_hidden_name](
                (x_latent, x_encoded_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden[src_hidden_name],
                    dst_nodes=shard_sizes_hidden[dst_hidden_name],
                    edges=downscale_edge_shard_sizes,
                ),
                edge_attr=downscale_edge_attr,
                edge_index=downscale_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )

            # Add skip connections
            x_latent = x_latent + x_level_skip[dst_hidden_name]

            if self.level_process:
                x_latent = self._process_level(
                    self.up_level_processor[dst_hidden_name],
                    self.up_level_processor_graph_providers[dst_hidden_name],
                    x_latent,
                    hidden_coordinates[dst_hidden_name],
                    shard_sizes_hidden[dst_hidden_name],
                    batch_size,
                    model_comm_group,
                )

        # Decoders, from the first hidden level
        x_out_dict = {}
        for dataset_name, target_dataset_template in target_template.items():
            if dataset_name not in self.target_datasets:
                continue

            target_coords, target_data_latent, shard_sizes_data, data_batch_sizes, data_timedeltas = (
                self._assemble_target(
                    batch[dataset_name],
                    x_data_latent_dict.get(dataset_name, None),
                    target_forcings.get(dataset_name, None),
                    target_dataset_template,
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                    dataset_name=dataset_name,
                )
            )

            graph_batch_kwargs = (
                {"src_batch_sizes": first_hidden_batch_sizes, "dst_batch_sizes": data_batch_sizes}
                if data_batch_sizes is not None
                else {}
            )
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                src_coords=(
                    first_hidden_coordinates_batched
                    if data_batch_sizes is not None
                    else hidden_coordinates[first_hidden]
                ),
                dst_coords=target_coords,
                dst_timedeltas=data_timedeltas,
                model_comm_group=model_comm_group,
                **graph_batch_kwargs,
            )
            decoder_edge_attr = decoder_edge_attr.to(dtype=x_latent.dtype)

            decoder_name = self.dataset2decoder[dataset_name]
            x_out = self.decoder[decoder_name](
                (x_latent, target_data_latent),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden[first_hidden],
                    dst_nodes=shard_sizes_data,  # None if not sharded
                    edges=dec_edge_shard_sizes,
                ),
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip_dict.get(dataset_name, None),
                target_dataset_template,
                dtype=x_out.dtype,
                dataset_name=dataset_name,
            )

        return Batch(x_out_dict)
