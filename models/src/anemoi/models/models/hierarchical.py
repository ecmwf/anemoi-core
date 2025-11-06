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

from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.truncation import BaseTruncation
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecHierarchical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network."""

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

        # hidden_dims is the dimentionality of features at each depth
        self.hidden_dims = {hidden: num_channels * (2**i) for i, hidden in enumerate(self._graph_hidden_names)}

        # Unpack config for hierarchical graph
        self.level_process = model_config.model.enable_hierarchical_level_processing
        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        # build networks
        self._build_networks(model_config)

        # build truncation
        self.truncation = BaseTruncation(self._truncation_data)

        # build boundings
        self.boundings = build_boundings(model_config, self.data_indices, self.statistics)

    def _calculate_input_dim_latent(self):
        return self.node_attributes.attr_ndims[self._graph_hidden_names[0]]

    def _build_networks(self, model_config):
        """Builds the model components."""

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.input_dim,
            in_channels_dst=self.input_dim_latent,
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

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.hidden_dims[self._graph_hidden_names[self.num_hidden - 1]],
            sub_graph=self._graph_data[
                (self._graph_hidden_names[self.num_hidden - 1], "to", self._graph_hidden_names[self.num_hidden - 1])
            ],
            src_grid_size=self.node_attributes.num_nodes[self._graph_hidden_names[self.num_hidden - 1]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_hidden_names[self.num_hidden - 1]],
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
                in_channels_dst=self.hidden_dims[dst_nodes_name],
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
            in_channels_dst=self.input_dim,
            hidden_dim=self.hidden_dims[self._graph_hidden_names[0]],
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_hidden_names[0], "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_hidden_names[0]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
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
        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias
        x_hidden_latents = {}
        for hidden in self._graph_hidden_names:
            x_hidden_latents[hidden] = self.node_attributes(hidden, batch_size=batch_size)

        # Get data and hidden shapes for sharding
        shard_shapes_hiddens = {}
        for hidden, x_latent in x_hidden_latents.items():
            shard_shapes_hiddens[hidden] = get_shard_shapes(x_latent, 0, model_comm_group=model_comm_group)

        # Run encoder
        x_data_latent, curr_latent = self.encoder(
            (x_data_latent, x_hidden_latents[self._graph_hidden_names[0]]),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hiddens[self._graph_hidden_names[0]]),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        x_encoded_latents = {}
        skip_connections = {}

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

            # store latents for skip connections
            skip_connections[src_hidden_name] = curr_latent

            # Encode to next hidden level
            x_encoded_latents[src_hidden_name], curr_latent = self.downscale[src_hidden_name](
                (curr_latent, x_hidden_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )

        # Processing hidden-most level
        curr_latent = self.processor(
            curr_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hiddens[dst_hidden_name],
            model_comm_group=model_comm_group,
        )

        ## Upscale
        for i in range(self.num_hidden - 1, 0, -1):
            src_hidden_name = self._graph_hidden_names[i]
            dst_hidden_name = self._graph_hidden_names[i - 1]

            # Decode to next level
            curr_latent = self.upscale[src_hidden_name](
                (curr_latent, x_encoded_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
            )

            # Add skip connections
            curr_latent = curr_latent + skip_connections[dst_hidden_name]

            # Processing at same level
            if self.level_process:
                curr_latent = self.up_level_processor[dst_hidden_name](
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hiddens[dst_hidden_name],
                    model_comm_group=model_comm_group,
                )

        # Run decoder
        x_out = self.decoder(
            (curr_latent, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hiddens[self._graph_hidden_names[0]], shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out
