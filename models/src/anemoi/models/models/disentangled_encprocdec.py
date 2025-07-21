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
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.utils.config import DotDict
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncoderProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiModelDisentangledEncProcDec(AnemoiModelEncoderProcDec):
    """Message passing graph neural network."""


    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        x_skip = x[:, -1, ...]
        x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
        x_skip = self._apply_truncation(x_skip, grid_shard_shapes, model_comm_group)
        x_skip = einops.rearrange(x_skip, "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size)

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

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
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

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

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

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

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

        # Processor
        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # Skip
        x_latent_proc = x_latent_proc + x_latent

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

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out
