# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.utils.config import DotDict
from anemoi.training.utils.debug_hydra import instantiate_debug



LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()

        self._graph_data = graph_data
        self._graph_name_data = config.graph.data
        self._graph_name_hidden = config.graph.hidden

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices

        self.multi_step = config.training.multistep_input
        self.num_channels = config.model.num_channels

        self.node_attributes = NamedNodesAttributes(config.model.trainable_parameters.hidden, self._graph_data)

        self.initialise_encoder_decoder(config)

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.internal_model.output.name_to_index)
                for cfg in getattr(config.model, "bounding", [])
            ]
        )

    def initialise_encoder_processor_decoder(self, config: DotDict):  
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        # Encoder data -> hidden
        self.encoder = instantiate(
            config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.internal_model.output.full) - len(
            data_indices.internal_model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x.dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out


class AnemoiModelEncProcDec_GraphTransformerFlexAttn(AnemoiModelEncProcDec):
    """
    GraphTransformerFlexAttn Implies:
    - TransformerProcessor
    - GraphTransformerForwardMapper
    - GraphTransformerBackwardMapper
    """

    def initialise_encoder_processor_decoder(self, config: DotDict):
        
        self.initialise_block_masks(config)

        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        self.processor_attention_span = config.processor_block_mask.attention_span

        self.encoder = instantiate_debug(
                    config.model.encoder,
                    in_channels_src=input_dim,
                    in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
                    hidden_dim=self.num_channels,
                    sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
                    src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
                    dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden]
        )

        processor_grid_name = self._graph_name_hidden
        self.processor = instantiate_debug(
            config.model.processor,
            num_channels=self.num_channels,
            processor_block_mask=self.map_spanSrcTgtBasegrid_blockmask[
                (self.processor_attention_span, processor_grid_name, processor_grid_name, False)
            ],
                )

        # Decoder hidden -> data
        self.decoder = instantiate(
            config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

    def initialise_block_masks(self, config: DotDict):
        from anemoi.models.layers.flex_attention import BlockMaskCreator
        

        # NOTE: improve this disbursement across devices strategy
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        self.map_spanSrcTgtBasegrid_blockmask_creator = {}
        self.map_spanSrcTgtBasegrid_blockmask = {}
        
        # Setting up block mask for processor
        processor_grid_name = self._graph_name_hidden
        self.processor_attention_span = config.processor_block_mask.attention_span

        bmc = BlockMaskCreator(
            graph=self._graph_data,
            query_grid_name=processor_grid_name,
            keyvalue_grid_name=processor_grid_name,
            devices=devices,
            **config.model.processor_block_mask
            # base_grid="query",
            # method="knn_haversine",
        )

        self.map_spanSrcTgtBasegrid_blockmask[(self.processor_attention_span, processor_grid_name, processor_grid_name, False)] = (
            bmc.setup_block_mask()
        )
        # NOTE: this line below is to make sure the BMC class persists in the memory (otherwise (i think) it may get garbage collected)
        # - stopping it being gc'd was useful when I had a different strategy for moving the block mask to the correct device. Not sure if this is necessary
        # now
        self.map_spanSrcTgtBasegrid_blockmask_creator[(self.processor_attention_span, processor_grid_name, processor_grid_name, False)] = bmc


class AnemoiModelEncProcDec_TransformerFlexAttn(AnemoiModelEncProcDec):
    """Vector Quantized Variational Autoencoder.

    TransformerFlexAttn Implies:
    - TransformerProcessor
    - TransformerForwardMapper
    - TransformerBackwardMapper

    """

    def initialise_encoder_decoder(self, config: DotDict):
        self.initialise_block_masks(config)

        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        
        # Initiate encoder
        self.mapper_attention_span = config.model.mapper_attention_span
        encoder_src_grid_name = self._graph_name_data
        encoder_dst_grid_name = self._graph_name_hidden
        encoder_base_grid = config.model.encoder_block_mask.base_grid

        self.encoder = instantiate_debug(
                config.model.encoder,
                in_channels_src=input_dim,
                in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                block_mask=self.map_spanSrcTgtBasegrid_blockmask[
                    (self.mapper_attention_span, encoder_src_grid_name, encoder_dst_grid_name, encoder_base_grid)
                ],
        )

        # Initiate processor
        self.processor_attention_span = config.base_processor_attention_span
        processor_src_grid_name = self._graph_name_hidden
        processor_dst_grid_name = self._graph_name_hidden
        processor_base_grid = config.model.processor_block_mask.base_grid
        self.processor = instantiate_debug(
            config.model.processor,
            num_channels=self.num_channels,
            processor_block_mask=self.map_spanSrcTgtBasegrid_blockmask[
                (self.processor_attention_span, processor_src_grid_name, processor_dst_grid_name, processor_base_grid)
            ],
        )

        # Initiate decoder
        decoder_src_grid_name = self._graph_name_hidden
        decoder_dst_grid_name = self._graph_name_data
        decoder_base_grid = config.model.decoder_block_mask.base_grid

        self.decoder = instantiate_debug(
            config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            block_mask=self.map_spanSrcTgtBasegrid_blockmask[
                (self.decoder_attention_span, decoder_src_grid_name, decoder_dst_grid_name, decoder_base_grid)
            ],
        )


    def initialise_block_masks(self, config: DotDict):
        from anemoi.models.layers.flex_attention import BlockMaskCreator
        

        # NOTE: improve this disbursement across devices strategy
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        self.map_spanSrcTgtBasegrid_blockmask_creator = {}
        self.map_spanSrcTgtBasegrid_blockmask = {}
        
        # region: Setting up block mask for processor
        processor_grid_name = self._graph_name_hidden
        attention_span = config.processor_block_mask.attention_span

        bmc = BlockMaskCreator(
            graph=self._graph_data,
            query_grid_name=processor_grid_name,
            keyvalue_grid_name=processor_grid_name,
            devices=devices,
            **config.model.processor_block_mask
            # base_grid="query",
            # method="knn_haversine",
        )

        self.map_spanSrcTgtBasegrid_blockmask[(attention_span, processor_grid_name, processor_grid_name, config.model.processor_block_mask.base_grid)] = (
            bmc.setup_block_mask()
        )
        # NOTE: this line below is to make sure the BMC class persists in the memory (otherwise (i think) it may get garbage collected)
        # - stopping it being gc'd was useful when I had a different strategy for moving the block mask to the correct device. Not sure if this is necessary
        # now
        self.map_spanSrcTgtBasegrid_blockmask_creator[(attention_span, processor_grid_name, processor_grid_name, config.model.processor_block_mask.base_grid)] = bmc

        # endregion: Setting up block masks for processor
        
        # region: Setting up block masks for encoder
        encoder_grid_name_src = self._graph_name_data
        encoder_grid_name_dst = self._graph_name_hidden

        bmc = BlockMaskCreator(
            graph=self._graph_data,
            query_grid_name=encoder_grid_name_dst,
            keyvalue_grid_name=encoder_grid_name_src,
            devices=devices,
            **config.model.encoder_block_mask
        )

        self.map_spanSrcTgtBasegrid_blockmask[(attention_span, encoder_grid_name_src, encoder_grid_name_dst, config.model.encoder_block_mask.base_grid)] = (
            bmc.setup_block_mask()
        )
        self.map_spanSrcTgtBasegrid_blockmask_creator[(attention_span, encoder_grid_name_src, encoder_grid_name_dst, config.model.encoder_block_mask.base_grid)] = bmc

        # endregion: Setting up block masks for encoder

        # region: Setting up block masks for decoder
        decoder_grid_name_src = self._graph_name_hidden
        decoder_grid_name_dst = self._graph_name_data

        bmc = BlockMaskCreator(
            graph=self._graph_data,
            query_grid_name=decoder_grid_name_dst,
            keyvalue_grid_name=decoder_grid_name_src,
            devices=devices,
            **config.model.decoder_block_mask
        )

        self.map_spanSrcTgtBasegrid_blockmask[(attention_span, decoder_grid_name_src, decoder_grid_name_dst, config.model.decoder_block_mask.base_grid)] = (
            bmc.setup_block_mask()
        )
        self.map_spanSrcTgtBasegrid_blockmask_creator[(attention_span, decoder_grid_name_src, decoder_grid_name_dst, config.model.decoder_block_mask.base_grid)] = bmc

        # endregion: Setting up block masks for decoder
