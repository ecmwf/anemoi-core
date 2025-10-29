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
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelAutoEncoder
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

import torch.nn as nn

class TinyAEHalf(nn.Module):
    """
    Expects input [B, C1, C2, T, F] with F=67 by default.
    Encodes last dim F -> target_dim (default 33), decodes back to F.
    """
    def __init__(self, feat_dim: int = 67, target_dim: int | None = None):
        super().__init__()
        self.feat_dim = feat_dim
        self.target_dim = target_dim if target_dim is not None else feat_dim

        # self.enc_norm = nn.LayerNorm(self.feat_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.target_dim, bias=True),
            nn.ReLU(inplace=True),
        )
        # self.dec_norm = nn.LayerNorm(self.target_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.target_dim, self.feat_dim, bias=True)
        )

    def encode(self, x):  
        T, F = x.shape
        # x2d = self.enc_norm(x)
        z2d = self.encoder(x)                              
        return z2d.reshape(T, self.target_dim)

    def decode(self, z):
        T, H = z.shape
        # z2d = self.dec_norm(z)
        xhat2d = self.decoder(z)
        return xhat2d.reshape(T, self.feat_dim)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z

class AnemoiTinyAE(AnemoiModelAutoEncoder):

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""

        self.tiny_compressor = TinyAEHalf(len(self.full_idx), int(len(self.full_idx)*0.8))

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        super()._calculate_shapes_and_indices(data_indices)
        self.full_idx = data_indices.data.output.full

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

        x_data_latent, _ = self._assemble_input(x, batch_size, grid_shard_shapes, model_comm_group)

        x_out, z = self.tiny_compressor(x_data_latent[:, self.full_idx])

        print(x_out.shape, z.shape)
        
        x_out = self._assemble_output(x_out, batch_size, ensemble_size, x.dtype)

        return x_out
        
        
