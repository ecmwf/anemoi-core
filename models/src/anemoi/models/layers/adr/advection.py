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

from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import Layout
from anemoi.models.layers.semi_lagrangian import _at_least_float32
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class SemiLagrangianAdvection(nn.Module):
    """Neural semi-Lagrangian advection, as in PARADIS (Pereira et al., arXiv:2601.21151).

    - A spatial mixer projects the latent state down to the transported channels.
    - A velocity network (norm -> linear -> bias -> activation -> spatial mixer) predicts one
      velocity per velocity field.
    - The transported channels are moved along their velocities by the sampler and projected back.

    Velocities are scaled by ``time_step``, the length of one layer's step in units of the inverse
    of the Earth's rotation rate, to give the angles travelled. By default they are given in each
    point's local east and north directions. These turn round across a pole, so the mixers that
    predict them see reversed directions on the other side of a pole.

    Each velocity field moves ``advection_channels // num_velocity_fields`` channels.

    The geometry gives the spatial mixers, the bias field and the sampler, so this step runs
    unchanged on a regular latitude-longitude grid and on a mesh.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        advection_channels: int,
        num_velocity_fields: int,
        velocity_hidden_dim: int,
        time_step: float,
        layer_kernels: DotDict,
        geometry,
        cartesian_displacement: bool = False,
    ) -> None:
        super().__init__()
        if advection_channels % num_velocity_fields != 0:
            raise ValueError(
                f"advection_channels ({advection_channels}) must be divisible by "
                f"num_velocity_fields ({num_velocity_fields})."
            )
        self.num_velocity_fields = num_velocity_fields
        self.advection_channels = advection_channels
        self.time_step = time_step

        self.sampler = geometry.sampler(num_velocity_fields, cartesian_displacement=cartesian_displacement)
        self.displacement_dim = self.sampler.displacement_dim

        self.down = geometry.spatial_mixer(num_channels, advection_channels, layer_kernels)
        self.up = layer_kernels.Linear(advection_channels, num_channels)

        self.velocity_norm = layer_kernels.LayerNorm(normalized_shape=num_channels)
        self.velocity_in = layer_kernels.Linear(num_channels, velocity_hidden_dim)
        self.velocity_bias = geometry.position_bias(velocity_hidden_dim, layer_kernels)
        self.velocity_activation = layer_kernels.Activation()
        self.velocity_out = geometry.spatial_mixer(
            velocity_hidden_dim, self.displacement_dim * num_velocity_fields, layer_kernels
        )

        init_branch(*self.down.branch_layers())
        init_branch(self.up)
        init_branch(self.velocity_in, *self.velocity_out.branch_layers())

    def forward(self, x: Tensor, layout: Layout, cond: Optional[Tensor] = None) -> Tensor:
        cond_kwargs = {"cond": cond} if cond is not None else {}

        velocity = self.velocity_in(self.velocity_norm(x, **cond_kwargs))
        velocity = self.velocity_activation(self.velocity_bias(velocity, layout))
        velocity = self.velocity_out(velocity, layout, cond).unflatten(
            -1, (self.num_velocity_fields, self.displacement_dim)
        )

        transported = self.down(x, layout, cond).unflatten(-1, (self.num_velocity_fields, -1))
        moved = self.sampler.sample(transported, _at_least_float32(velocity) * self.time_step, layout)
        return self.up(moved.flatten(-2))


class Advection(nn.Module):
    """Advection step: the state moves towards its advected version by a learned share per channel.

    On a mesh the number of neighbour searches is ``batch * num_velocity_fields * nodes``, so a
    velocity field per advected channel, the default on a grid, is slow with the brute-force
    search of ``KNNSampler``.

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
        layer_kernels: DotDict,
        geometry,
        time_step: float,
        advection_channels: int = 768,
        num_velocity_fields: Optional[int] = None,
        velocity_hidden_dim: int = 384,
        cartesian_displacement: bool = False,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        if num_velocity_fields is None:
            LOGGER.debug(
                f"{type(self).__name__}: num_velocity_fields not specified, using {advection_channels}, "
                "one velocity field per advected channel"
            )
            num_velocity_fields = advection_channels

        self.transport = SemiLagrangianAdvection(
            num_channels=num_channels,
            advection_channels=advection_channels,
            num_velocity_fields=num_velocity_fields,
            velocity_hidden_dim=velocity_hidden_dim,
            time_step=time_step,
            layer_kernels=layer_kernels,
            geometry=geometry,
            cartesian_displacement=cartesian_displacement,
        )
        # This determines how much of the advected state is applied to the current state.
        self.share = nn.Parameter(torch.full((num_channels,), -1.0))

    def forward(self, x: Tensor, layout: Layout, cond: Optional[Tensor] = None) -> Tensor:
        share = torch.sigmoid(self.share).to(x.dtype)
        return share * (self.transport(x, layout, cond) - x)
