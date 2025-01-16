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



class AnemoiVQVAES4_GraphTransformerFlexAttn_Haversine(AnemoiModelEncProcDec):
    """Vector Quantized Variational Autoencoder.
    This is version s1 of the S4 model:

    """

    def initialise_encoder_decoder(self, config: DotDict):

        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.map_spanSrcTgtInv_blockmask_creator = {}
        self.map_spanSrcTgtInv_blockmask = {}

        # setup block masks for encoder Transformer processors
        for source_grid_name in self.list_graph_name_encoder:
            scaled_haversine_distance = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                source_grid_name,
                scaling_method="inverse_scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )
            bmc = BlockMaskCreator(
                self._graph_data,
                scaled_haversine_distance,
                query_grid_name=source_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[
                (scaled_haversine_distance, source_grid_name, source_grid_name, False)
            ] = bmc.setup_block_mask()
            self.map_spanSrcTgtInv_blockmask_creator[
                (scaled_haversine_distance, source_grid_name, source_grid_name, False)
            ] = bmc

        # setup block masks for decoder transformer processors
        for source_grid_name in self.list_graph_name_decoder:
            scaled_haversine_distance = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                source_grid_name,
                scaling_method="inverse_scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )
            # Processor
            bmc = BlockMaskCreator(
                self._graph_data,
                scaled_haversine_distance,
                query_grid_name=source_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[
                (scaled_haversine_distance, source_grid_name, source_grid_name, False)
            ] = bmc.setup_block_mask()
            self.map_spanSrcTgtInv_blockmask_creator[
                (scaled_haversine_distance, source_grid_name, source_grid_name, False)
            ] = bmc

        # setup encoder modules
        encoder_modules = []
        for i in range(self.no_levels_encoder):

            dst_grid_name = self.list_graph_name_encoder[i + 1]
            src_grid_name = self.list_graph_name_encoder[i]

            scaled_haversine_distance = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="inverse_scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )

            # Local KNN based Flex Attention
            encoder_modules.append(
                # NOTE: Below method used for debugging purposes
                # instantiate(
                instantiate_debug(
                    config.model.encoder,
                    in_channels_src_weather=(
                        config.model.weather_state_embedding_out_features
                        if i == 0
                        else self.num_channels_encoder[i - 1]
                    ),
                    in_channels_src_latlon=config.model.latlon_embedding_out_features,
                    in_channels_dst_latlon=config.model.latlon_embedding_out_features,
                    hidden_dim=self.hidden_dim_encoder[i],  # Hidden dimension of the mapper and processor
                    out_channels_dst=self.num_channels_encoder[i],  # Output channels of the mapper
                    sub_graph_mapper=(self._graph_data[(src_grid_name, "to", dst_grid_name)]),
                    sub_graph_edge_attributes=config.model.attributes.edges,
                    src_grid_size=self._list_hidden_grid_size_encoder[i],
                    dst_grid_size=self._list_hidden_grid_size_encoder[i + 1],
                    ln_autocast=config.model.ln_autocast,
                    noise_injector=self.noise_injector,
                    emb_nodes_src_bias=False if i == 0 else True,
                    # cln_noise_dim=self.noise_injector.outp_channels,
                    processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (scaled_haversine_distance, src_grid_name, src_grid_name, False)
                    ],
                )
            )

        self.encoder = nn.ModuleList(encoder_modules)

        # setup decoder modules
        decoder_modules = []
        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]
            src_grid_name = self.list_graph_name_decoder[i]

            scaled_haversine_distance = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="inverse_scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )

            # NOTE () : Matt coreolis force is inherently positional encoding
            decoder_modules.append(
                # instantiate(
                instantiate_debug(
                    config.model.decoder,
                    in_channels_src_weather=(
                        self.num_channels_decoder[i - 1] if i != 0 else self.num_channels_encoder[-1]
                    ),
                    in_channels_src_latlon=config.model.latlon_embedding_out_features,
                    in_channels_dst_latlon=config.model.latlon_embedding_out_features,
                    out_channels_dst=self.num_channels_decoder[i],
                    hidden_dim=self.hidden_dim_decoder[i],
                    sub_graph_mapper=(
                        self._graph_data[
                            (self.list_graph_name_decoder[i], "to", self.list_graph_name_decoder[i + 1])
                        ]  # For some reaosn edge_length is all zeros here
                    ),
                    sub_graph_edge_attributes=config.model.attributes.edges,
                    src_grid_size=self._list_hidden_grid_size_decoder[i],
                    dst_grid_size=(self._list_hidden_grid_size_decoder[i + 1]),
                    ln_autocast=config.model.ln_autocast,
                    noise_injector=self.noise_injector,
                    processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (scaled_haversine_distance, src_grid_name, src_grid_name, False)
                    ],
                )
            )

        self.decoder = nn.ModuleList(decoder_modules)


class AnemoiModelEncProcDec_GraphTransformerFlexAttn_Knn(AnemoiModelEncProcDec):
    """Vector Quantized Variational Autoencoder."""

    def initialise_encoder_processor_decoder(self, config: DotDict):
        

        self.initialise_block_masks(config)

        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        processor_attention_span = config.base_processor_attention_span


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
            processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                (processor_attention_span, processor_grid_name, processor_grid_name, False)
            ],
                )

        self.decoder = instantiate_debug(



    def initialise_block_masks(self, config: DotDict):
        from anemoi.models.layers.flex_attention import BlockMaskCreator
        from anemoi.models.layers.flex_attention import calculate_scaled_attention_attention_spans

        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.map_spanSrcTgtInv_blockmask_creator = {}
        self.map_spanSrcTgtInv_blockmask = {}
        
        # Setting up block mask for processor
        processor_grid_name = self._graph_name_hidden
        attention_span = calculate_scaled_attention_attention_spans(
            self.base_processor_attention_span,
            self._input_grid_name,
            processor_grid_name,
            scaling_method="scale_span_relative_to_grid_size",
            _graph_data=self._graph_data,
        )  # NOTE: need to sort out what the attention span should be on the mapper edge

        bmc = BlockMaskCreator(
            self._graph_data,
            attention_span,
            query_grid_name=processor_grid_name,
            keyvalue_grid_name=processor_grid_name,
            devices=devices,
            method="knn_haversine",
            base_grid="query",
        )

        self.map_spanSrcTgtInv_blockmask[(attention_span, processor_grid_name, processor_grid_name, False)] = (
            bmc.setup_block_mask()
        )
        # NOTE: this line below is to make sure the BMC class persists in the memory (otherwise (i think) it may get garbage collected)
        # - stopping it being gc'd was useful when I had a different strategy for moving the block mask to the correct device. Not sure if this is necessary
        # now
        self.map_spanSrcTgtInv_blockmask_creator[(attention_span, processor_grid_name, processor_grid_name, False)] = bmc


class AnemoiVQVAES4_TransformerFlexAttn_Haversine(AnemoiVQVAES4):
    """Vector Quantized Variational Autoencoder.

    This is version s1 of the S4 model:

    """

    def initialise_encoder_decoder(self, config: DotDict):

        mapper_attention_span = config.model.mapper_attention_span

        # Setup block masks
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.map_spanSrcTgtInv_blockmask_creator = {}
        self.map_spanSrcTgtInv_blockmask = {}

        for source_grid_name, target_grid_name in zip(self.list_graph_name_encoder, self.list_graph_name_encoder[1:]):

            # Mapper
            bmc = BlockMaskCreator(
                self._graph_data,
                mapper_attention_span,
                query_grid_name=target_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="haversine",
                base_grid="keyvalue",
            )
            self.map_spanSrcTgtInv_blockmask[(mapper_attention_span, target_grid_name, source_grid_name, True)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[
                (mapper_attention_span, target_grid_name, source_grid_name, True)
            ] = bmc

            attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                source_grid_name,
                scaling_method="constant_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )  # NOTE: need to sort out what the attention span should be on the mapper edge

            bmc = BlockMaskCreator(
                self._graph_data,
                attention_span,
                query_grid_name=source_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[(attention_span, source_grid_name, source_grid_name, False)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[(attention_span, source_grid_name, source_grid_name, False)] = bmc

        for source_grid_name, target_grid_name in zip(self.list_graph_name_decoder, self.list_graph_name_decoder[1:]):

            # TODO later: if the reverse exists in the encoder than we simply use the same block mask but transposed
            # TODO later: add checks to see if block mask is already created and then use that
            # TODO later: set up a way to ensure that reverse=True/False share the underlying knn_index_matrix or connection matrix, but just that the k/v are swapped

            # Mapper
            bmc = BlockMaskCreator(
                self._graph_data,
                mapper_attention_span,
                query_grid_name=target_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[(mapper_attention_span, target_grid_name, source_grid_name, False)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[
                (mapper_attention_span, target_grid_name, source_grid_name, False)
            ] = bmc

            # Processor
            attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                source_grid_name,
                scaling_method="constant_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )
            bmc = BlockMaskCreator(
                self._graph_data,
                attention_span,
                query_grid_name=source_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[(attention_span, source_grid_name, source_grid_name, False)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[(attention_span, source_grid_name, source_grid_name, False)] = bmc

        encoder_modules = []
        for i in range(self.no_levels_encoder):

            dst_grid_name = self.list_graph_name_encoder[i + 1]
            src_grid_name = self.list_graph_name_encoder[i]

            processor_attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="constant_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )

            # Local KNN based Flex Attention
            encoder_modules.append(
                # NOTE: Below method used for debugging purposes
                # instantiate(
                instantiate_debug(
                    config.model.encoder,
                    #     _recursive_=False,
                    #     _convert_=None,
                    in_channels_src_weather=(
                        config.model.weather_state_embedding_out_features
                        if i == 0
                        else self.num_channels_encoder[i - 1]
                    ),
                    in_channels_src_latlon=config.model.latlon_embedding_out_features,
                    in_channels_dst_latlon=config.model.latlon_embedding_out_features,
                    hidden_dim=self.hidden_dim_encoder[i],  # Hidden dimension of the mapper and processor
                    out_channels_dst=self.num_channels_encoder[i],  # Output channels of the mapper
                    ln_autocast=config.model.ln_autocast,
                    noise_injector=self.noise_injector,
                    emb_nodes_src_bias=False if i == 0 else True,
                    # cln_noise_dim=self.noise_injector.outp_channels,
                    mapper_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (mapper_attention_span, dst_grid_name, src_grid_name, True)
                    ],
                    processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (processor_attention_span, src_grid_name, src_grid_name, False)
                    ],
                )
            )

        self.encoder = nn.ModuleList(encoder_modules)

        decoder_modules = []
        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]
            src_grid_name = self.list_graph_name_decoder[i]

            processor_attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                measure="distance",
                scaling_method="constant_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )

            # NOTE () : Matt coreolis force is inherently positional encoding
            decoder_modules.append(
                # instantiate(
                instantiate_debug(
                    config.model.decoder,
                    in_channels_src_weather=(
                        self.num_channels_decoder[i - 1] if i != 0 else self.num_channels_encoder[-1]
                    ),
                    in_channels_src_latlon=config.model.latlon_embedding_out_features,
                    in_channels_dst_latlon=config.model.latlon_embedding_out_features,
                    out_channels_dst=self.num_channels_decoder[i],
                    hidden_dim=self.hidden_dim_decoder[i],
                    ln_autocast=config.model.ln_autocast,
                    noise_injector=self.noise_injector,
                    mapper_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (mapper_attention_span, dst_grid_name, src_grid_name, False)
                    ],
                    processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (processor_attention_span, src_grid_name, src_grid_name, False)
                    ],
                )
            )

        self.decoder = nn.ModuleList(decoder_modules)


class AnemoiVQVAES4_TransformerFlexKnn(AnemoiModelEncProcDec):
    """Vector Quantized Variational Autoencoder.

    """

    def initialise_encoder_decoder(self, config: DotDict):

        mapper_attention_span = config.model.mapper_attention_span

        # Setup block masks
        # NOTE: This device allocation strategy needs to be improved
        # TODO: Implement strategy that hooks onto the .to() calls such that 
        # if this is moved to a device that does  not already have a copy of necessary tensors, then a new copies of required tensors are made on this device 
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.map_spanSrcTgtInv_blockmask_creator = {}
        self.map_spanSrcTgtInv_blockmask = {}

        # setup block masks for encoder transformer mapper and processors
        for source_grid_name, target_grid_name in zip(self.list_graph_name_encoder, self.list_graph_name_encoder[1:]):

            # Mapper
            bmc = BlockMaskCreator(
                self._graph_data,
                mapper_attention_span,
                query_grid_name=target_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="knn_haversine",
                base_grid="keyvalue",
            )
            self.map_spanSrcTgtInv_blockmask[(mapper_attention_span, target_grid_name, source_grid_name, True)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[
                (mapper_attention_span, target_grid_name, source_grid_name, True)
            ] = bmc

            attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                source_grid_name,
                scaling_method="scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )  # NOTE: need to sort out what the attention span should be on the mapper edge

            bmc = BlockMaskCreator(
                self._graph_data,
                attention_span,
                query_grid_name=source_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="knn_haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[(attention_span, source_grid_name, source_grid_name, False)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[(attention_span, source_grid_name, source_grid_name, False)] = bmc

        # setup block masks for decoder transformer mapper and processors
        for source_grid_name, target_grid_name in zip(self.list_graph_name_decoder, self.list_graph_name_decoder[1:]):

            # TODO later: if the reverse exists in the encoder than we simply use the same block mask but transposed
            # TODO later: add checks to see if block mask is already created and then use that
            # TODO later: set up a way to ensure that reverse=True/False share the underlying knn_index_matrix or connection matrix, but just that the k/v are swapped

            # Mapper
            bmc = BlockMaskCreator(
                self._graph_data,
                mapper_attention_span,
                query_grid_name=target_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="knn_haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[(mapper_attention_span, target_grid_name, source_grid_name, False)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[
                (mapper_attention_span, target_grid_name, source_grid_name, False)
            ] = bmc

            # Processor
            attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                source_grid_name,
                scaling_method="scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )
            bmc = BlockMaskCreator(
                self._graph_data,
                attention_span,
                query_grid_name=source_grid_name,
                keyvalue_grid_name=source_grid_name,
                devices=devices,
                method="knn_haversine",
                base_grid="query",
            )
            self.map_spanSrcTgtInv_blockmask[(attention_span, source_grid_name, source_grid_name, False)] = (
                bmc.setup_block_mask()
            )
            self.map_spanSrcTgtInv_blockmask_creator[(attention_span, source_grid_name, source_grid_name, False)] = bmc

        encoder_modules = []
        for i in range(self.no_levels_encoder):

            dst_grid_name = self.list_graph_name_encoder[i + 1]
            src_grid_name = self.list_graph_name_encoder[i]

            processor_attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )

            # Local KNN based Flex Attention
            encoder_modules.append(
                # NOTE: Below method used for debugging purposes
                # instantiate(
                instantiate_debug(
                    config.model.encoder,
                    #     _recursive_=False,
                    #     _convert_=None,
                    in_channels_src_weather=(
                        config.model.weather_state_embedding_out_features
                        if i == 0
                        else self.num_channels_encoder[i - 1]
                    ),
                    in_channels_src_latlon=config.model.latlon_embedding_out_features,
                    in_channels_dst_latlon=config.model.latlon_embedding_out_features,
                    hidden_dim=self.hidden_dim_encoder[i],  # Hidden dimension of the mapper and processor
                    out_channels_dst=self.num_channels_encoder[i],  # Output channels of the mapper
                    ln_autocast=config.model.ln_autocast,
                    noise_injector=self.noise_injector,
                    emb_nodes_src_bias=False if i == 0 else True,
                    # cln_noise_dim=self.noise_injector.outp_channels,
                    mapper_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (mapper_attention_span, dst_grid_name, src_grid_name, True)
                    ],
                    processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (processor_attention_span, src_grid_name, src_grid_name, False)
                    ],
                )
            )

        self.encoder = nn.ModuleList(encoder_modules)

        decoder_modules = []
        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]
            src_grid_name = self.list_graph_name_decoder[i]

            processor_attention_span = calculate_scaled_attention_attention_spans(
                self.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )

            # NOTE () : Matt coreolis force is inherently positional encoding
            decoder_modules.append(
                # instantiate(
                instantiate_debug(
                    config.model.decoder,
                    in_channels_src_weather=(
                        self.num_channels_decoder[i - 1] if i != 0 else self.num_channels_encoder[-1]
                    ),
                    in_channels_src_latlon=config.model.latlon_embedding_out_features,
                    in_channels_dst_latlon=config.model.latlon_embedding_out_features,
                    out_channels_dst=self.num_channels_decoder[i],
                    hidden_dim=self.hidden_dim_decoder[i],
                    ln_autocast=config.model.ln_autocast,
                    noise_injector=self.noise_injector,
                    mapper_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (mapper_attention_span, dst_grid_name, src_grid_name, False)
                    ],
                    processor_block_mask=self.map_spanSrcTgtInv_blockmask[
                        (processor_attention_span, src_grid_name, src_grid_name, False)
                    ],
                )
            )

        self.decoder = nn.ModuleList(decoder_modules)

