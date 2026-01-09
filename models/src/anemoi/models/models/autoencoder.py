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

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelAutoEncoder(BaseGraphModel):

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""

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

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        super()._calculate_shapes_and_indices(data_indices)
        forcing_names = data_indices.model._forcing
        self._forcing_input_idx = [data_indices.name_to_index[name] for name in forcing_names]
        self.num_input_channels_forcings = len(self._forcing_input_idx)
        self.target_dim = self.num_input_channels_forcings + self.node_attributes.attr_ndims[self._graph_name_data]

    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

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
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_target, 0, grid_shard_shapes, model_comm_group
            )
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
        shard_shapes_target = get_or_apply_shard_shapes(x_target_latent, 0, grid_shard_shapes, model_comm_group)
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

        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_data_latent, shard_shapes_data = self._assemble_input(x, batch_size, grid_shard_shapes, model_comm_group)

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group=model_comm_group)

        # Encoder
        x_data_latent, x_latent = self.encoder(
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
        )

        # Do not pass x_data_latent to the decoder
        # In autoencoder training this would cause the model to discard everything else and just keep the values they were before
        # Only pass data and forcing coordinates to the decoder
        x_target_latent, shard_shapes_target = self._assemble_forcings(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        # Decoder
        x_out = self.decoder(
            (x_latent, x_target_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_target),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_target_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, batch_size, ensemble_size, x.dtype)

        return x_out
