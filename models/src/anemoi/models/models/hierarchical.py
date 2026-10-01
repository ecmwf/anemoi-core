# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from hydra.utils import instantiate
from torch import Tensor
from torch import nn

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.encoder_processor_decoder import ForwardContext
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecHierarchical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network.

    Encoders and decoders are those of :class:`AnemoiModelEncProcDec`, attached to the first hidden
    level. The process stage is a U-Net over the hidden levels: the latent is mapped level by level
    down to the deepest one (``upscale`` mappers, optional per-level ``down_level_processor``),
    processed there, and mapped back up (``downscale`` mappers, per-level skip connections, optional
    ``up_level_processor``). The number of latent channels doubles at every level.
    """

    def _calculate_hidden_dims(self) -> dict[str, int]:
        """Latent channels at each hidden level: ``2**level`` times the aggregator output."""
        return {
            hidden_name: self.latent_aggregator.hidden_dim * (2**level)
            for level, hidden_name in enumerate(self._hidden_names)
        }

    def _build_level_processor(self, nodes_name: str, model_config: DotDict) -> tuple[nn.Module, nn.Module]:
        """Build the graph provider and processor acting within one hidden level."""
        graph_provider = self._build_graph_provider(nodes_name, nodes_name, model_config.processor)
        processor = instantiate(
            model_config.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.hidden_dims[nodes_name],
            edge_dim=graph_provider.edge_dim,
            num_layers=model_config.level_process_num_layers,
        )
        return graph_provider, processor

    def _build_processor(self, model_config: DotDict) -> None:
        """Build the hierarchy: level processors, the deepest processor and the inter-level mappers."""
        self.num_hidden = len(self._hidden_names)
        # Pairs of (shallower, deeper) adjacent hidden levels, from the data side to the deepest level.
        level_pairs = list(zip(self._hidden_names[:-1], self._hidden_names[1:]))

        # Level processors
        self.level_process = model_config.enable_hierarchical_level_processing
        if self.level_process:
            self.down_level_processor = nn.ModuleDict()
            self.down_level_processor_graph_providers = nn.ModuleDict()
            self.up_level_processor = nn.ModuleDict()
            self.up_level_processor_graph_providers = nn.ModuleDict()
            for nodes_name in self._hidden_names[:-1]:
                (
                    self.down_level_processor_graph_providers[nodes_name],
                    self.down_level_processor[nodes_name],
                ) = self._build_level_processor(nodes_name, model_config)
                (
                    self.up_level_processor_graph_providers[nodes_name],
                    self.up_level_processor[nodes_name],
                ) = self._build_level_processor(nodes_name, model_config)

        # Main processor at deepest level
        super()._build_processor(model_config, num_channels=self.hidden_dims[self._hidden_names[-1]])

        # Upscale: shallower -> deeper level (forward mappers)
        self.upscale = nn.ModuleDict()
        self.upscale_graph_providers = nn.ModuleDict()
        for src_nodes_name, dst_nodes_name in level_pairs:
            self.upscale_graph_providers[src_nodes_name] = self._build_graph_provider(
                src_nodes_name, dst_nodes_name, model_config.upscale_mapper
            )
            self.upscale[src_nodes_name] = instantiate(
                model_config.upscale_mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.node_attributes.attr_ndims[dst_nodes_name],
                num_channels=self.hidden_dims[dst_nodes_name],
                edge_dim=self.upscale_graph_providers[src_nodes_name].edge_dim,
            )

        # Downscale: deeper -> shallower level (backward mappers), keyed by the destination level
        self.downscale = nn.ModuleDict()
        self.downscale_graph_providers = nn.ModuleDict()
        for dst_nodes_name, src_nodes_name in level_pairs:
            self.downscale_graph_providers[dst_nodes_name] = self._build_graph_provider(
                src_nodes_name, dst_nodes_name, model_config.downscale_mapper
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

    def _process(self, x_latent: Tensor, ctx: ForwardContext) -> Tensor:
        """U-Net over the hidden levels: ``hidden_1 -> ... -> hidden_n -> processor -> ... -> hidden_1``."""
        level_pairs = list(zip(self._hidden_names[:-1], self._hidden_names[1:]))

        # Down the hierarchy, towards the deepest level
        x_level_skip: dict[str, Tensor] = {}
        x_level_encoded: dict[str, Tensor] = {}
        for src_nodes_name, dst_nodes_name in level_pairs:
            if self.level_process:
                x_latent = self._run_processor(
                    self.down_level_processor[src_nodes_name],
                    self.down_level_processor_graph_providers[src_nodes_name],
                    x_latent,
                    ctx.shard_sizes_hidden[src_nodes_name],
                    ctx,
                )

            # store latents for skip connections
            x_level_skip[src_nodes_name] = x_latent

            # Encode to next hidden level
            x_level_encoded[src_nodes_name], x_latent = self._run_mapper(
                self.upscale[src_nodes_name],
                self.upscale_graph_providers[src_nodes_name],
                (x_latent, ctx.x_hidden[dst_nodes_name]),
                src_shard_sizes=ctx.shard_sizes_hidden[src_nodes_name],
                dst_shard_sizes=ctx.shard_sizes_hidden[dst_nodes_name],
                ctx=ctx,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )

        # Deepest level: main processor with latent skip connection
        x_latent = super()._process(x_latent, ctx)

        # Back up the hierarchy, towards the data nodes
        for dst_nodes_name, src_nodes_name in reversed(level_pairs):
            # Decode to previous level
            x_latent = self._run_mapper(
                self.downscale[dst_nodes_name],
                self.downscale_graph_providers[dst_nodes_name],
                (x_latent, x_level_encoded[dst_nodes_name]),
                src_shard_sizes=ctx.shard_sizes_hidden[src_nodes_name],
                dst_shard_sizes=ctx.shard_sizes_hidden[dst_nodes_name],
                ctx=ctx,
                keep_x_dst_sharded=True,
            )

            # Add skip connections
            x_latent = x_latent + x_level_skip[dst_nodes_name]

            if self.level_process:
                x_latent = self._run_processor(
                    self.up_level_processor[dst_nodes_name],
                    self.up_level_processor_graph_providers[dst_nodes_name],
                    x_latent,
                    ctx.shard_sizes_hidden[dst_nodes_name],
                    ctx,
                )

        return x_latent
