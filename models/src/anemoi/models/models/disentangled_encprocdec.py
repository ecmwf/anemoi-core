# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.utils.config import DotDict
from .autoencoder import AnemoiModelAutoEncoder, AnemoiModelHierarchicalAutoEncoder

LOGGER = logging.getLogger(__name__)

class AnemoiModelDisentangledEncProcDec(AnemoiModelAutoEncoder):
    """Disnentangled graph network."""

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

        # Monkey-patch _calculate_shapes_and_indices on self before init
        self._calculate_shapes_and_indices = self._calculate_shapes_and_indices.__get__(self)

        super(AnemoiModelAutoEncoder, self).__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )
    
    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        return AnemoiModelAutoEncoder._assemble_input(self, x, batch_size, grid_shard_shapes, model_comm_group)

    def _assemble_output(self, x_out, batch_size, ensemble_size, dtype):
        return AnemoiModelAutoEncoder._assemble_output(self, x_out, batch_size, ensemble_size, dtype)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        AnemoiModelAutoEncoder._calculate_shapes_and_indices(self, data_indices)

        # only 1 timestep per time to the encoder
        self.input_dim = (
            self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        )

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
            Input data (batch time ensemble grid vars)
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
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Encode each time step separately and then accumulate them
        x_accum = None
        for i in range(self.multi_step):
            x_t = x[:, i:i+1, ...]  # shape: [B, 1, E, G, D]

            x_data_latent, shard_shapes_data = self._assemble_input(
                x_t, batch_size, grid_shard_shapes, model_comm_group
            )

            x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

            # Encode timestep 
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
            if x_accum is None:
                x_accum = x_latent
            else:
                x_accum += x_latent  # accumulates in-place
            
        # Store the last latent for the residual connection
        x_skip = x_latent

        # Processor
        x_latent_proc = self.processor(
            x_accum,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # Residual learning over the latent space
        x_latent_proc = x_latent_proc + x_skip

        # Decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, batch_size, ensemble_size, x.dtype)

        return x_out


class AnemoiModelDisentangledEncProcDecHierarchical(AnemoiModelHierarchicalAutoEncoder):
    """Disnentangled hierarchical graph network."""

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

        # Monkey-patch _calculate_shapes_and_indices on self before init
        self._calculate_shapes_and_indices = self._calculate_shapes_and_indices.__get__(self)

        super(AnemoiModelHierarchicalAutoEncoder, self).__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )
    
    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        return AnemoiModelHierarchicalAutoEncoder._assemble_input(self, x, batch_size, grid_shard_shapes, model_comm_group)

    def _assemble_output(self, x_out, batch_size, ensemble_size, dtype):
        return AnemoiModelHierarchicalAutoEncoder._assemble_output(self, x_out, batch_size, ensemble_size, dtype)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        AnemoiModelHierarchicalAutoEncoder._calculate_shapes_and_indices(self, data_indices)

        # only 1 timestep per time to the encoder
        self.input_dim = (
            self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        )

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
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Encode each time step separately and then accumulate them
        x_accum = None
        for i in range(self.multi_step):
            x_t = x[:, i:i+1, ...]  # shape: [B, 1, E, G, D]

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
                shard_shapes_hiddens[hidden] = get_shard_shapes(x_latent, 0, model_comm_group)
            
            # Encode timestep 
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

                if x_accum is None:
                    x_accum = curr_latent
                else:
                    x_accum += curr_latent  # accumulates in-place
        
        # Store the last latent for the residual connection
        x_skip = curr_latent

        # Processing hidden-most level
        curr_latent = self.processor(
            curr_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hiddens[dst_hidden_name],
            model_comm_group=model_comm_group,
        )

        # Residual learning over the latent space
        curr_latent = curr_latent + x_skip

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

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
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