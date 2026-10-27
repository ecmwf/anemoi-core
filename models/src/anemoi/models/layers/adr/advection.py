# (C) Copyright 2024-2026 Anemoi contributors.
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
from torch import nn

from anemoi.models.layers.adr.layers import LowRankBias
from anemoi.models.layers.adr.layers import SpatialMixer
from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.semi_lagrangian import SemiLagrangianLayer
from anemoi.models.layers.semi_lagrangian import _at_least_float32
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class SemiLagrangianAdvection(SemiLagrangianLayer):
    """Neural semi-Lagrangian advection, as in PARADIS (Pereira et al., arXiv:2601.21151).

    A spatial mixer projects the latent state down to the transported channels. A velocity network
    (layer norm, linear layer, learned bias field, activation, spatial mixer) predicts one
    velocity per head from the latent state. The transported channels are moved along their
    velocities and projected back up to the full latent state, which is returned.

    Velocities are scaled by ``time_step``, the length of one layer's step in units of the inverse
    of the Earth's rotation rate, to give the angles travelled. By default they are given in each
    point's local east and north directions, as in the paper. These turn round across a pole, so
    the stencils that predict them see reversed directions in the rows padded in from the other
    side of a pole. ``cartesian_displacement`` predicts 3D velocities instead (see
    ``SemiLagrangianLayer``).

    Each head has its own velocity and moves ``advection_channels // num_heads`` channels.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        advection_channels: int,
        num_heads: int,
        velocity_hidden_dim: int,
        nlat: int,
        nlon: int,
        time_step: float,
        layer_kernels: DotDict,
        kernel_size: int = 5,
        interpolation: str = "bicubic",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        cartesian_displacement: bool = False,
        grid_index: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            nlat=nlat,
            nlon=nlon,
            num_heads=num_heads,
            interpolation=interpolation,
            cartesian_displacement=cartesian_displacement,
            grid_index=grid_index,
        )
        if advection_channels % num_heads != 0:
            raise ValueError(f"advection_channels ({advection_channels}) must be divisible by num_heads ({num_heads}).")
        self.advection_channels = advection_channels
        self.time_step = time_step

        self.down = SpatialMixer(num_channels, advection_channels, layer_kernels, kernel_size)
        self.up = layer_kernels.Linear(advection_channels, num_channels)

        self.velocity_norm = layer_kernels.LayerNorm(normalized_shape=num_channels)
        self.velocity_in = layer_kernels.Linear(num_channels, velocity_hidden_dim)
        self.velocity_bias = LowRankBias(
            velocity_hidden_dim, nlat, nlon, layer_kernels, rank=bias_rank, num_base_maps=bias_base_maps
        )
        self.velocity_activation = layer_kernels.Activation()
        self.velocity_out = SpatialMixer(
            velocity_hidden_dim, self.displacement_dim * num_heads, layer_kernels, kernel_size
        )

        init_branch(self.down.stencil.conv, self.down.mix)
        init_branch(self.up)
        init_branch(self.velocity_in, self.velocity_out.stencil.conv, self.velocity_out.mix)

    def forward(self, x: Tensor, layout: LatLonLayout, cond: Optional[Tensor] = None) -> Tensor:
        cond_kwargs = {"cond": cond} if cond is not None else {}

        velocity = self.velocity_in(self.velocity_norm(x, **cond_kwargs))
        velocity = self.velocity_activation(self.velocity_bias(velocity, layout))
        velocity = self.velocity_out(velocity, layout).view(-1, self.num_heads, self.displacement_dim)

        transported = self.down(x, layout).view(-1, self.num_heads, self.advection_channels // self.num_heads)
        moved = self.sample(transported, _at_least_float32(velocity) * self.time_step, layout)
        return self.up(moved.view(-1, self.advection_channels))


class NeuralSemiLagrangian(nn.Module):
    """Neural Semi-Lagrangian module.

    The state moves towards its advected version by a learned share per channel.

    Attributes
    ----------
    transport : SemiLagrangianAdvection
        The semi-Lagrangian advection module.
    share : nn.Parameter
        The learned share per channel, passed through a sigmoid. It determines how much of the
        advected state is applied to the current state.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        nlat: int,
        nlon: int,
        time_step: float,
        layer_kernels: DotDict,
        grid_index: Optional[Tensor] = None,
        advection_channels: int = 768,
        num_heads: Optional[int] = None,
        velocity_hidden_dim: int = 384,
        kernel_size: int = 5,
        interpolation: str = "bicubic",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        cartesian_displacement: bool = False,
    ) -> None:
        super().__init__()
        if num_heads is None:
            LOGGER.debug(
                f"{self.__class__.__name__}: num_heads not specified, using {advection_channels} (advection channels)"
            )
            num_heads = advection_channels

        self.transport = SemiLagrangianAdvection(
            num_channels=num_channels,
            advection_channels=advection_channels,
            num_heads=num_heads,
            velocity_hidden_dim=velocity_hidden_dim,
            nlat=nlat,
            nlon=nlon,
            time_step=time_step,
            layer_kernels=layer_kernels,
            kernel_size=kernel_size,
            interpolation=interpolation,
            bias_rank=bias_rank,
            bias_base_maps=bias_base_maps,
            cartesian_displacement=cartesian_displacement,
            grid_index=grid_index,
        )
        # This determines how much of the advected state is applied to the current state.
        self.share = nn.Parameter(torch.full((num_channels,), -1.0))

    def forward(self, x: Tensor, layout: LatLonLayout, cond: Optional[Tensor] = None) -> Tensor:
        share = torch.sigmoid(self.share).to(x.dtype)
        return share * (self.transport(x, layout, cond) - x)
