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
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.utils.config import DotDict
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.models import AnemoiModelEncProcDecHierarchical

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
        super().__init__(model_config=model_config, data_indices=data_indices, statistics=statistics, graph_data=graph_data)

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

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent, x_data_latent),
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

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out


class AnemoiModelHierarchicalAutoEncoder(AnemoiModelEncProcDecHierarchical):

    def __init__(
            self,
            *,
            model_config: DotDict,
            data_indices: dict,
            statistics: dict,
            graph_data: HeteroData,
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
            super().__init__(model_config=model_config, data_indices=data_indices, statistics=statistics, graph_data=graph_data)

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
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

        print(shard_shapes_data, shard_shapes_hiddens)

        print("Encoder shapes: ", (shard_shapes_data, shard_shapes_hiddens[self._graph_hidden_names[0]]))
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
            print(f"Downscale {i} shapes: ", (shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]))

            x_encoded_latents[src_hidden_name], curr_latent = self._run_mapper(
                self.downscale[src_hidden_name],
                (curr_latent, x_trainable_hiddens[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
            )

        ## Upscale
        for i in range(self.num_hidden - 1, 0, -1):
            src_hidden_name = self._graph_hidden_names[i]
            dst_hidden_name = self._graph_hidden_names[i - 1]

            # Decode to next level
            print(f"Upscale {i} shapes: ", (shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]))

            curr_latent = self._run_mapper(
                self.upscale[src_hidden_name],
                (curr_latent, x_encoded_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hiddens[src_hidden_name], shard_shapes_hiddens[dst_hidden_name]),
                model_comm_group=model_comm_group,
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
        print("Decoder shapes: ", (shard_shapes_hiddens[self._graph_hidden_names[0]], shard_shapes_data))
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