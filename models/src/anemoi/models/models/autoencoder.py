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
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelAutoEncoder(AnemoiModelEncProcDec):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        nn.Module.__init__(self)

        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics
        self._truncation_data = truncation_data

        model_config = DotDict(model_config)
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden
        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        # we can't register these as buffers because DDP does not support sparse tensors
        # these will be moved to the GPU when first used via sefl.interpolate_down/interpolate_up
        self.A_down, self.A_up = None, None
        if "down" in self._truncation_data:
            self.A_down = self._make_truncation_matrix(self._truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if "up" in self._truncation_data:
            self.A_up = self._make_truncation_matrix(self._truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

        self.supports_sharded_input = True

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.num_channels,
            in_channels_dst=self.target_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=self.data_indices.model.output.name_to_index,
                    statistics=self.statistics,
                    name_to_index_stats=self.data_indices.data.input.name_to_index,
                )
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        super()._calculate_shapes_and_indices(data_indices)
        forcing_names = data_indices.model._forcing
        self._forcing_input_idx = [data_indices.name_to_index[name] for name in forcing_names]
        self.num_input_channels_forcings = len(self._forcing_input_idx)
        self.target_dim = self.num_input_channels_forcings + self.node_attributes.attr_ndims[self._graph_name_data]

    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):

        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        return x_data_latent, shard_shapes_data

    def _assemble_output(self, x_out, batch_size, ensemble_size, dtype):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out

    def _assemble_forcings(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        node_attributes_target = self.node_attributes(self._graph_name_data, batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_target, 0, grid_shard_shapes, model_comm_group)
            node_attributes_target = shard_tensor(node_attributes_target, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_target_latent = torch.cat(
            (
                einops.rearrange(
                    x[..., self._forcing_input_idx],
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                node_attributes_target,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_target = self._get_shard_shapes(x_target_latent, 0, grid_shard_shapes, model_comm_group)
        return x_target_latent, shard_shapes_target

    def forward(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        in_out_sharded = grid_shard_shapes is not None

        assert not (
            in_out_sharded and (grid_shard_shapes is None or model_comm_group is None)
        ), "If input is sharded, grid_shard_shapes and model_comm_group must be provided."

        x_data_latent, shard_shapes_data = self._assemble_input(x, batch_size, grid_shard_shapes, model_comm_group)

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        # Encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        # Do not pass x_data_latent to the decoder
        # In autoencoder training this would cause the model to discard everything else and just keep the values they were before
        # Only pass data and forcing coordinates to the decoder
        x_target_latent, shard_shapes_target = self._assemble_forcings(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        # Decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent, x_target_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_target),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, batch_size, ensemble_size, x.dtype)

        return x_out


class AnemoiModelHierarchicalAutoEncoder(AnemoiModelAutoEncoder):

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        nn.Module.__init__(self)
        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics
        self._truncation_data = truncation_data

        model_config = DotDict(model_config)
        self._graph_name_data = model_config.graph.data
        self._graph_hidden_names = model_config.graph.hidden
        self.num_hidden = len(self._graph_hidden_names)
        self.multi_step = model_config.training.multistep_input
        num_channels = model_config.model.num_channels
        self.hidden_dims = {hidden: num_channels * (2**i) for i, hidden in enumerate(self._graph_hidden_names)}
        self.level_process = model_config.model.enable_hierarchical_level_processing

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.supports_sharded_input = True

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_hidden_names[0]],
            hidden_dim=self.hidden_dims[self._graph_hidden_names[0]],
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_hidden_names[0])],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_hidden_names[0]],
        )

        # Level processors
        if self.level_process:
            self.down_level_processor = nn.ModuleDict()
            self.up_level_processor = nn.ModuleDict()

            for i in range(0, self.num_hidden - 1):
                nodes_names = self._graph_hidden_names[i]

                self.down_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_names],
                    sub_graph=self._graph_data[(nodes_names, "to", nodes_names)],
                    src_grid_size=self.node_attributes.num_nodes[nodes_names],
                    dst_grid_size=self.node_attributes.num_nodes[nodes_names],
                    num_layers=model_config.model.level_process_num_layers,
                )

                self.up_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_names],
                    sub_graph=self._graph_data[(nodes_names, "to", nodes_names)],
                    src_grid_size=self.node_attributes.num_nodes[nodes_names],
                    dst_grid_size=self.node_attributes.num_nodes[nodes_names],
                    num_layers=model_config.model.level_process_num_layers,
                )

        # Downscale
        self.downscale = nn.ModuleDict()

        for i in range(0, self.num_hidden - 1):
            src_nodes_name = self._graph_hidden_names[i]
            dst_nodes_name = self._graph_hidden_names[i + 1]

            self.downscale[src_nodes_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.node_attributes.attr_ndims[dst_nodes_name],
                hidden_dim=self.hidden_dims[dst_nodes_name],
                sub_graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
                src_grid_size=self.node_attributes.num_nodes[src_nodes_name],
                dst_grid_size=self.node_attributes.num_nodes[dst_nodes_name],
            )

        # Upscale
        self.upscale = nn.ModuleDict()

        for i in range(1, self.num_hidden):
            src_nodes_name = self._graph_hidden_names[i]
            dst_nodes_name = self._graph_hidden_names[i - 1]

            self.upscale[src_nodes_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.node_attributes.attr_ndims[dst_nodes_name],
                hidden_dim=self.hidden_dims[src_nodes_name],
                out_channels_dst=self.hidden_dims[dst_nodes_name],
                sub_graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
                src_grid_size=self.node_attributes.num_nodes[src_nodes_name],
                dst_grid_size=self.node_attributes.num_nodes[dst_nodes_name],
            )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.hidden_dims[self._graph_hidden_names[0]],
            in_channels_dst=self.target_dim,
            hidden_dim=self.hidden_dims[self._graph_hidden_names[0]],
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_hidden_names[0], "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_hidden_names[0]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=self.data_indices.model.output.name_to_index,
                    statistics=self.statistics,
                    name_to_index_stats=self.data_indices.data.input.name_to_index,
                )
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )


    def forward(
        self,
        x: Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        in_out_sharded = grid_shard_shapes is not None

        assert not (
            in_out_sharded and (grid_shard_shapes is None or model_comm_group is None)
        ), "If input is sharded, grid_shard_shapes and model_comm_group must be provided."

        # Prepare input
        x_data_latent, shard_shapes_data = self._assemble_input(x, batch_size, grid_shard_shapes, model_comm_group)

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias
        x_hidden_latents = {}
        for hidden in self._graph_hidden_names:
            x_hidden_latents[hidden] = self.node_attributes(hidden, batch_size=batch_size)

        # Get data and hidden shapes for sharding
        shard_shapes_hiddens = {}
        for hidden, x_latent in x_hidden_latents.items():
            shard_shapes_hiddens[hidden] = get_shard_shapes(x_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, curr_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latents[self._graph_hidden_names[0]]),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hiddens[self._graph_hidden_names[0]]),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        x_encoded_latents = {}

        ## Downscale
        for i in range(0, self.num_hidden - 1):
            src_hidden_name = self._graph_hidden_names[i]
            dst_hidden_name = self._graph_hidden_names[i + 1]

            # Processing at same level
            if self.level_process:
                curr_latent = self.down_level_processor[src_hidden_name](
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hiddens[src_hidden_name],
                    model_comm_group=model_comm_group,
                )

            # Encode to next hidden level
            x_encoded_latents[src_hidden_name], curr_latent = self._run_mapper(
                self.downscale[src_hidden_name],
                (curr_latent, x_hidden_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )

        ## Upscale
        for i in range(self.num_hidden - 1, 0, -1):
            src_hidden_name = self._graph_hidden_names[i]
            dst_hidden_name = self._graph_hidden_names[i - 1]

            # Decode to next level
            curr_latent = self._run_mapper(
                self.upscale[src_hidden_name],
                (curr_latent, x_hidden_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
            )

            # Processing at same level
            if self.level_process:
                curr_latent = self.up_level_processor[dst_hidden_name](
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hiddens[dst_hidden_name],
                    model_comm_group=model_comm_group,
                )

        # Do not pass x_data_latent to the decoder
        # In autoencoder training this would cause the model to discard everything else and just keep the values they were before
        # Only pass data and forcing coordinates to the decoder
        x_target_latent, shard_shapes_target = self._assemble_forcings(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (curr_latent, x_target_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hiddens[self._graph_hidden_names[0]], shard_shapes_target),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, batch_size, ensemble_size, x.dtype)

        return x_out
