# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph import TrainableTensor
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
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        nn.Module.__init__(self)

        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_hidden_names = model_config.graph.hidden
        self.num_hidden = len(self._graph_hidden_names)

        # Unpack config for hierarchical graph
        self.level_process = model_config.model.enable_hierarchical_level_processing

        # hidden_dims is the dimentionality of features at each depth
        self.hidden_dims = {
            hidden: model_config.model.num_channels * (2**i) for i, hidden in enumerate(self._graph_hidden_names)
        }

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices

        self.multi_step = model_config.training.multistep_input

        # self.node_attributes = {hidden_name: NamedNodesAttributes(model_config.model.trainable_parameters[hidden_name], self._graph_data)
        #                         for hidden_name in self._graph_hidden_names}
        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=input_dim,
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
                    num_channels=self.hidden_dims[nodes_names],
                    sub_graph=self._graph_data[(nodes_names, "to", nodes_names)],
                    src_grid_size=self.node_attributes.num_nodes[nodes_names],
                    dst_grid_size=self.node_attributes.num_nodes[nodes_names],
                    num_layers=model_config.model.level_process_num_layers,
                )

                self.up_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    num_channels=self.hidden_dims[nodes_names],
                    sub_graph=self._graph_data[(nodes_names, "to", nodes_names)],
                    src_grid_size=self.node_attributes.num_nodes[nodes_names],
                    dst_grid_size=self.node_attributes.num_nodes[nodes_names],
                    num_layers=model_config.model.level_process_num_layers,
                )

        self.processor = instantiate(
            model_config.model.processor,
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
            in_channels_src=self.hidden_dims[self._graph_hidden_names[0]],
            in_channels_dst=input_dim,
            hidden_dim=self.hidden_dims[self._graph_hidden_names[0]],
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_hidden_names[0], "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_hidden_names[0]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.internal_model.output.name_to_index)
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.trainable_data = TrainableTensor(trainable_size=self.trainable_data_size, tensor_size=self._data_grid_size)
        self.trainable_hidden = nn.ModuleDict()

        for hidden in self._graph_hidden_names:
            self.trainable_hidden[hidden] = TrainableTensor(
                trainable_size=self.trainable_hidden_size, tensor_size=self._hidden_grid_sizes[hidden]
            )

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None, **kwargs) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        x_trainable_data = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias
        x_trainable_hiddens = {}
        for hidden in self._graph_hidden_names:
            x_trainable_hiddens[hidden] = self.node_attributes(hidden, batch_size=batch_size)

        # Get data and hidden shapes for sharding
        shard_shapes_data = get_shape_shards(x_trainable_data, 0, model_comm_group)
        shard_shapes_hiddens = {}
        for hidden, x_latent in x_trainable_hiddens.items():
            shard_shapes_hiddens[hidden] = get_shape_shards(x_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, curr_latent = self._run_mapper(
            self.encoder,
            (x_trainable_data, x_trainable_hiddens[self._graph_hidden_names[0]]),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hiddens[self._graph_hidden_names[0]]),
            model_comm_group=model_comm_group,
        )

        # Run processor
        x_encoded_latents = {}
        x_skip = {}

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
            x_skip[src_hidden_name] = curr_latent

            # Encode to next hidden level
            x_encoded_latents[src_hidden_name], curr_latent = self._run_mapper(
                self.downscale[src_hidden_name],
                (curr_latent, x_trainable_hiddens[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
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
            curr_latent = self._run_mapper(
                self.upscale[src_hidden_name],
                (curr_latent, x_encoded_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
            )

            # Add skip connections
            curr_latent = curr_latent + x_skip[dst_hidden_name]

            # Processing at same level
            if self.level_process:
                curr_latent = self.up_level_processor[dst_hidden_name](
                    curr_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hiddens[dst_hidden_name],
                    model_comm_group=model_comm_group,
                )

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (curr_latent, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hiddens[self._graph_hidden_names[0]], shard_shapes_data),
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
