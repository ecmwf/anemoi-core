# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os
from abc import abstractmethod
from typing import Optional

import torch
from anemoi.training.utils.debug_hydra import instantiate_debug
from anemoi.utils.config import DotDict
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards

LOGGER = logging.getLogger(__name__)

# Used by the VAE reconstruction models


class EncoderBlock(nn.Module):
    "Consists of one mapping layer and N processing layers"

    def __init__(
        self,
        mapper: DotDict,
        processor: DotDict,
        in_channels_src: int,
        in_channels_dst: int,
        out_channels_dst: int,
        hidden_dim: int,
        noise_injector: Optional[nn.Module] = None,
        emb_nodes_src_bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        # TODO (rilwan-ade): (currently its mapper -> processor)
        # NOTE: (rilwan-ade) should it be processor -> mapper --> processor? In the literature VQ-VAE / KL-VAE,
        # there is processing before the mapping and then a (single/few) processing layer after the mapping
        self.noise_injector = noise_injector

        # processing before passing through the mapping layer
        if in_channels_src != hidden_dim:
            self.emb_nodes_src = nn.Linear(in_channels_src, hidden_dim, bias=False)
        else:
            self.emb_nodes_src = nn.Identity()

        if in_channels_dst != hidden_dim:
            self.emb_nodes_dst = nn.Linear(in_channels_dst, hidden_dim, bias=False)
        else:
            self.emb_nodes_dst = nn.Identity()

        self.initialise_processor_mapper(
            mapper=mapper,
            processor=processor,
            out_channels_dst=out_channels_dst,
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def initialise_processor_mapper(
        self,
        mapper: DotDict,
        processor: DotDict,
        out_channels_dst: int,
        hidden_dim: int,
        sub_graph_mapper: HeteroData = None,
        sub_graph_edge_attributes: list[str] = None,
        src_grid_size: int = None,
        dst_grid_size: int = None,
        sub_graph_processor: HeteroData = None,
        ln_autocast: bool = False,
        noise_injector: nn.Module = None,
        attention_span: int = 512,
        emb_nodes_src_bias: bool = True,
    ):

        self.processor = instantiate_debug(
            processor,
            num_channels=hidden_dim,
            sub_graph=sub_graph_processor,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=src_grid_size,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            ln_autocast=ln_autocast,
            attention_span=attention_span,
        )

        # mapping layer
        # self.mapper = instantiate(
        self.mapper = instantiate_debug(
            mapper,
            in_channels_src=hidden_dim,
            in_channels_dst=hidden_dim,
            out_channels_dst=out_channels_dst,
            hidden_dim=hidden_dim,
            sub_graph=sub_graph_mapper,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            ln_autocast=ln_autocast,
        )

    def forward(
        self, x_src_latent, x_dst_latent, batch_size, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        x_src_latent = self.emb_nodes_src(x_src_latent)
        x_dst_latent = self.emb_nodes_dst(x_dst_latent)

        # get shard shapes
        shard_shapes_src = get_shape_shards(x_src_latent, 0, model_comm_group)
        shard_shapes_dst = get_shape_shards(x_dst_latent, 0, model_comm_group)

        # TODO (rilwan-ade): currently noise is sampled even if it is not used in later conditional layer norms in the processor/mapper
        if self.noise_injector:
            noise_src = self.noise_injector(x_src_latent, shard_shapes_src, model_comm_group)
            noise_dst = self.noise_injector(x_dst_latent, shard_shapes_dst, model_comm_group)
            shape_noise_src = get_shape_shards(noise_src, 0, model_comm_group)
            shape_noise_dst = get_shape_shards(noise_dst, 0, model_comm_group)
        else:
            noise_src = None
            noise_dst = None
            shape_noise_src = None
            shape_noise_dst = None

        # Transformer based processor
        x_src_latent = self.processor(
            x_src_latent,
            batch_size=batch_size,
            shard_shapes=(shard_shapes_src, shape_noise_src),
            model_comm_group=model_comm_group,
            noise_levels=(noise_src),
        )

        if self.mapper.checkpoint:
            x_src_latent, x_dst_latent = checkpoint(
                self.mapper,
                x_src_latent,
                x_dst_latent,
                batch_size,
                (noise_src, noise_dst),
                (shard_shapes_src, shard_shapes_dst, shape_noise_src, shape_noise_dst),
                model_comm_group,
                use_reentrant=False,
            )
        else:
            x_src_latent, x_dst_latent = self.mapper(
                x_src_latent,
                x_dst_latent,
                noise_levels=(noise_src, noise_dst),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_src, shard_shapes_dst, shape_noise_src, shape_noise_dst),
                model_comm_group=model_comm_group,
            )

        output = (x_src_latent, x_dst_latent)

        return output


class DecoderBlock(nn.Module):
    "Consists of one mapping layer and N processing layers"

    def __init__(
        self,
        mapper: DotDict,
        processor: DotDict,
        hidden_dim: int,
        in_channels_src: int,
        in_channels_dst: int,
        out_channels_dst: int,
        noise_injector: Optional[nn.Module] = None,
        emb_nodes_src_bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.noise_injector = noise_injector

        # processing before passing through the mapping layer
        # self.processor = instantiate(
        if in_channels_src != hidden_dim:
            self.emb_nodes_src = nn.Linear(in_channels_src, hidden_dim, bias=False)
        else:
            self.emb_nodes_src = nn.Identity()

        if in_channels_dst != hidden_dim:
            self.emb_nodes_dst = nn.Linear(in_channels_dst, hidden_dim, bias=False)
        else:
            self.emb_nodes_dst = nn.Identity()

        self.initialise_processor_mapper(
            processor=processor, mapper=mapper, hidden_dim=hidden_dim, out_channels_dst=out_channels_dst, **kwargs
        )

    def initialise_processor_mapper(
        self,
        processor: DotDict,
        mapper: DotDict,
        hidden_dim: int,
        out_channels_dst: int,
        sub_graph_processor: HeteroData,
        sub_graph_edge_attributes: list[str],
        sub_graph_mapper: HeteroData,
        src_grid_size: int,
        dst_grid_size: int,
        ln_autocast: bool,
        attention_span: int,
    ):

        self.processor = instantiate_debug(
            processor,
            num_channels=hidden_dim,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            ln_autocast=ln_autocast,
            sub_graph=sub_graph_processor,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=src_grid_size,
            attention_span=attention_span,
        )

        # mapping layer
        self.mapper = instantiate_debug(
            mapper,
            in_channels_src=hidden_dim,
            in_channels_dst=hidden_dim,
            out_channels_dst=out_channels_dst,
            hidden_dim=hidden_dim,
            sub_graph=sub_graph_mapper,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            ln_autocast=ln_autocast,
        )

    def forward(
        self, x_src_latent, x_dst_latent, batch_size, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        x_src_latent = self.emb_nodes_src(x_src_latent)

        # get shard shapes
        shard_shapes_src = get_shape_shards(x_src_latent, 0, model_comm_group)
        shard_shapes_dst = get_shape_shards(x_dst_latent, 0, model_comm_group)

        # TODO (rilwan-ade): currently noise is sampled even if it is not used in later conditional layer norms in the processor/mapper
        if self.noise_injector:
            noise_src = self.noise_injector(x_src_latent, shard_shapes_src, model_comm_group)
            noise_dst = self.noise_injector(x_dst_latent, shard_shapes_dst, model_comm_group)
            shape_noise_src = get_shape_shards(noise_src, 0, model_comm_group)
            shape_noise_dst = get_shape_shards(noise_dst, 0, model_comm_group)
        else:
            noise_src = None
            noise_dst = None
            shape_noise_src = None
            shape_noise_dst = None

        x_src_latent_proc = self.processor(
            x_src_latent,
            noise_levels=(noise_src),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_src, shape_noise_src),
            model_comm_group=model_comm_group,
        )

        # Run mapper
        if self.mapper.checkpoint:
            _, x_dst_latent = checkpoint(
                self.mapper,
                x_src_latent_proc,
                x_dst_latent,
                batch_size,
                (noise_src, noise_dst),
                (shard_shapes_src, shard_shapes_dst, shape_noise_src, shape_noise_dst),
                model_comm_group,
                use_reentrant=False,
            )
        else:
            _, x_dst_latent = self.mapper(
                x_src_latent_proc,
                x_dst_latent,
                noise_levels=(noise_src, noise_dst),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_src, shard_shapes_dst, shape_noise_src, shape_noise_dst),
                model_comm_group=model_comm_group,
            )

        return x_dst_latent


class EncoderBlockRopeTransformerFlex(EncoderBlock):
    "Consists of one mapping layer and N processing layers"

    def initialise_processor_mapper(
        self,
        mapper: DotDict,
        processor: DotDict,
        out_channels_dst: int,
        hidden_dim: int,
        processor_block_mask: Tensor,
        mapper_block_mask: Tensor,
        src_rope_embedding: nn.Module,
        dst_rope_embedding: nn.Module,
    ):

        self.processor = instantiate_debug(
            processor,
            num_channels=hidden_dim,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            block_mask=processor_block_mask,
            rope_embedding=src_rope_embedding,
        )

        # mapping layer
        # self.mapper = instantiate(
        self.mapper = instantiate_debug(
            mapper,
            in_channels_src=hidden_dim,
            in_channels_dst=hidden_dim,
            out_channels_dst=out_channels_dst,
            hidden_dim=hidden_dim,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            block_mask=mapper_block_mask,
            src_rope_embedding=src_rope_embedding,
            dst_rope_embedding=dst_rope_embedding,
        )


class DecoderBlockRopeTransformerFlex(DecoderBlock):
    "Consists of one mapping layer and N processing layers"

    def initialise_processor_mapper(
        self,
        processor: DotDict,
        mapper: DotDict,
        hidden_dim: int,
        out_channels_dst: int,
        processor_block_mask: Tensor,
        mapper_block_mask: Tensor,
        src_rope_embedding: nn.Module,
        dst_rope_embedding: nn.Module,
    ):

        self.processor = instantiate_debug(
            processor,
            num_channels=hidden_dim,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            block_mask=processor_block_mask,
            rope_embedding=src_rope_embedding,
        )

        self.mapper = instantiate_debug(
            mapper,
            in_channels_src=hidden_dim,
            in_channels_dst=hidden_dim,
            out_channels_dst=out_channels_dst,
            hidden_dim=hidden_dim,
            cln_noise_dim=self.noise_injector.outp_channels if self.noise_injector else None,
            block_mask=mapper_block_mask,
            src_rope_embedding=src_rope_embedding,
            dst_rope_embedding=dst_rope_embedding,
        )
