# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from hydra.utils import get_class
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.graphs.create import HeteroData
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.layers.graph_provider import BaseGraphProvider
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.utils.config import COORDS_DIM
from anemoi.utils.config import DotDict

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

    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, dynamic_graph_config: DotDict) -> None:
        """Builds the model components."""
        hidden_names = self.hidden_names

        # Encoders: data -> first hidden level
        self._build_encoding_graphproviders(model_config.encoders, static_graph, dynamic_graph_config)
        self._build_encoding_networks(model_config.encoders)

        # Latent aggregator: combines encoder outputs at the first hidden level
        self._build_latent_aggregator(model_config.latent_aggregator)

        # Width of the latent at each level, doubling from one level to the next
        self.hidden_dims = {hidden: self.latent_aggregator.hidden_dim * (2**i) for i, hidden in enumerate(hidden_names)}

        # Level processors
        self.level_process = model_config.enable_hierarchical_level_processing
        if self.level_process:
            self.down_level_processor = nn.ModuleDict()
            self.down_level_processor_graph_providers = nn.ModuleDict()
            self.up_level_processor = nn.ModuleDict()
            self.up_level_processor_graph_providers = nn.ModuleDict()
            for nodes_name in hidden_names[:-1]:
                self.down_level_processor_graph_providers[nodes_name] = self._create_graph_provider(
                    nodes_name, nodes_name, model_config.processor, static_graph, dynamic_graph_config
                )
                self.down_level_processor[nodes_name] = instantiate(
                    model_config.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_name],
                    edge_dim=self.down_level_processor_graph_providers[nodes_name].edge_dim,
                    num_layers=model_config.level_process_num_layers,
                )

                self.up_level_processor_graph_providers[nodes_name] = self._create_graph_provider(
                    nodes_name, nodes_name, model_config.processor, static_graph, dynamic_graph_config
                )
                self.up_level_processor[nodes_name] = instantiate(
                    model_config.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_name],
                    edge_dim=self.up_level_processor_graph_providers[nodes_name].edge_dim,
                    num_layers=model_config.level_process_num_layers,
                )

        # Main processor at the last hidden level, the only mesh in processor_graph_provider
        last_level = hidden_names[-1]
        self.processor_graph_provider = nn.ModuleDict(
            {
                last_level: self._create_graph_provider(
                    last_level, last_level, model_config.processor, static_graph, dynamic_graph_config
                )
            }
        )
        self.processor = instantiate(
            model_config.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.hidden_dims[last_level],
            edge_dim=self.processor_graph_provider[last_level].edge_dim,
        )

        # Upscale: hidden level i -> i + 1, keyed by the source level
        self.upscale = nn.ModuleDict()
        self.upscale_graph_providers = nn.ModuleDict()
        for src_nodes_name, dst_nodes_name in zip(hidden_names[:-1], hidden_names[1:]):
            self.upscale_graph_providers[src_nodes_name] = self._create_graph_provider(
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

        # Downscale: hidden level i + 1 -> i, keyed by the destination level. Its destination input is what
        # upscale[i] returned for level i: the level latent itself, or, for a mapper that updates its source
        # nodes (returns_src_latent), that updated latent at level i + 1's width.
        upscale_returns_src_latent = getattr(
            get_class(model_config.upscale_mapper._target_), "returns_src_latent", False
        )
        self.downscale = nn.ModuleDict()
        self.downscale_graph_providers = nn.ModuleDict()
        for dst_nodes_name, src_nodes_name in zip(hidden_names[:-1], hidden_names[1:]):
            self.downscale_graph_providers[dst_nodes_name] = self._create_graph_provider(
                src_nodes_name, dst_nodes_name, model_config.downscale_mapper, static_graph, dynamic_graph_config
            )
            self.downscale[dst_nodes_name] = instantiate(
                model_config.downscale_mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.hidden_dims[src_nodes_name if upscale_returns_src_latent else dst_nodes_name],
                num_channels=self.hidden_dims[src_nodes_name],
                out_channels_dst=self.hidden_dims[dst_nodes_name],
                edge_dim=self.downscale_graph_providers[dst_nodes_name].edge_dim,
            )

        # Decoders: first hidden level -> data
        self._build_decoding_graphproviders(model_config.decoders, static_graph, dynamic_graph_config)
        self._build_decoding_networks(model_config.decoders)

    def _map_level(
        self,
        mapper: nn.Module,
        graph_provider: BaseGraphProvider,
        x_src: Tensor,
        x_dst: Tensor,
        *,
        src_name: str,
        dst_name: str,
        hidden_coordinates: dict[str, Tensor],
        shard_sizes_hidden: dict[str, ShardSizes],
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Map the latent of hidden level ``src_name`` onto hidden level ``dst_name``, keeping it sharded."""
        edge_attr, edge_index, edge_shard_sizes = self._hidden_edges(
            graph_provider,
            hidden_coordinates[src_name],
            hidden_coordinates[dst_name],
            x_src,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        return mapper(
            (x_src, x_dst),
            batch_size=batch_size,
            shard_info=BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden[src_name],
                dst_nodes=shard_sizes_hidden[dst_name],
                edges=edge_shard_sizes,
            ),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,
        )

    def _process(
        self,
        x_latent: Tensor,
        *,
        hidden_name: str,
        hidden_coordinates: Tensor,
        shard_sizes_hidden: ShardSizes,
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
    ) -> Tensor:
        """Take the latent of the first hidden level up the levels, process it at the last, and bring it back down."""
        hidden_names = self.hidden_names

        # Coordinates, starting latent and shard sizes of every level; the first is the one the encoders wrote to.
        coordinates = {hidden_name: hidden_coordinates}
        x_hidden_latents = {}
        shard_sizes = {hidden_name: shard_sizes_hidden}
        for level in hidden_names[1:]:
            coordinates[level], x_hidden_latents[level], shard_sizes[level] = self._initial_hidden_latent(
                level, batch_size=batch_size, device=x_latent.device, model_comm_group=model_comm_group
            )
        level_kwargs = {
            "hidden_coordinates": coordinates,
            "shard_sizes_hidden": shard_sizes,
            "batch_size": batch_size,
            "model_comm_group": model_comm_group,
        }

        # Up the hidden levels
        x_level_skip = {}
        x_encoded_latents = {}
        for src_name, dst_name in zip(hidden_names[:-1], hidden_names[1:]):
            if self.level_process:
                x_latent = self._run_processor(
                    self.down_level_processor[src_name],
                    self.down_level_processor_graph_providers[src_name],
                    x_latent,
                    hidden_coordinates=coordinates[src_name],
                    shard_sizes_hidden=shard_sizes[src_name],
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

            x_level_skip[src_name] = x_latent

            # The source output is the level latent, or the mapper's updated source latent at the next level's
            # width (returns_src_latent); it is the destination input of downscale[src_name].
            x_encoded_latents[src_name], x_latent = self._map_level(
                self.upscale[src_name],
                self.upscale_graph_providers[src_name],
                x_latent,
                x_hidden_latents[dst_name],
                src_name=src_name,
                dst_name=dst_name,
                **level_kwargs,
            )

        # Processor at the last hidden level, with the latent skip connection
        last_name = hidden_names[-1]
        x_latent = super()._process(
            x_latent,
            hidden_name=last_name,
            hidden_coordinates=coordinates[last_name],
            shard_sizes_hidden=shard_sizes[last_name],
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        # Down the hidden levels, adding each level's skip connection
        for dst_name, src_name in reversed(list(zip(hidden_names[:-1], hidden_names[1:]))):
            x_latent = self._map_level(
                self.downscale[dst_name],
                self.downscale_graph_providers[dst_name],
                x_latent,
                x_encoded_latents[dst_name],
                src_name=src_name,
                dst_name=dst_name,
                **level_kwargs,
            )
            x_latent = x_latent + x_level_skip[dst_name]

            if self.level_process:
                x_latent = self._run_processor(
                    self.up_level_processor[dst_name],
                    self.up_level_processor_graph_providers[dst_name],
                    x_latent,
                    hidden_coordinates=coordinates[dst_name],
                    shard_sizes_hidden=shard_sizes[dst_name],
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

        return x_latent
