# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.adr.layers import GraphLowRankBias
from anemoi.models.layers.adr.layers import GraphSpatialMixer
from anemoi.models.layers.adr.layers import LatLonLowRankBias
from anemoi.models.layers.adr.layers import LatLonSpatialMixer
from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import GraphLayout
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.semi_lagrangian import GraphSemiLagrangianLayer
from anemoi.models.layers.semi_lagrangian import SemiLagrangianLayer
from anemoi.models.layers.semi_lagrangian import _at_least_float32
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

Layout = Union[LatLonLayout, GraphLayout]


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

    The geometry comes in through the sampler (``SemiLagrangianLayer`` on a latitude-longitude
    grid, ``GraphSemiLagrangianLayer`` on a graph) and through ``mixer(in_channels, out_channels)``
    and ``bias(num_channels)``, which build the spatial mixers and the bias field.
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
        sampler: Union[SemiLagrangianLayer, GraphSemiLagrangianLayer],
        mixer: Callable[[int, int], nn.Module],
        bias: Callable[[int], nn.Module],
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
        self.sampler = sampler
        self.displacement_dim = sampler.displacement_dim

        self.down = mixer(num_channels, advection_channels)
        self.up = layer_kernels.Linear(advection_channels, num_channels)

        self.velocity_norm = layer_kernels.LayerNorm(normalized_shape=num_channels)
        self.velocity_in = layer_kernels.Linear(num_channels, velocity_hidden_dim)
        self.velocity_bias = bias(velocity_hidden_dim)
        self.velocity_activation = layer_kernels.Activation()
        self.velocity_out = mixer(velocity_hidden_dim, self.displacement_dim * num_velocity_fields)

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

    The geometry-specific subclasses (``LatLonAdvection``, ``GraphAdvection``) build the
    transport.

    Attributes
    ----------
    transport : SemiLagrangianAdvection
        The semi-Lagrangian advection module.
    share : nn.Parameter
        The learned share per channel, passed through a sigmoid. It determines how much of the
        advected state is applied to the current state.
    """

    def __init__(self, *, num_channels: int, transport: SemiLagrangianAdvection) -> None:
        super().__init__()
        self.transport = transport
        # This determines how much of the advected state is applied to the current state.
        self.share = nn.Parameter(torch.full((num_channels,), -1.0))

    def forward(self, x: Tensor, layout: Layout, cond: Optional[Tensor] = None) -> Tensor:
        share = torch.sigmoid(self.share).to(x.dtype)
        return share * (self.transport(x, layout, cond) - x)


def _default_velocity_fields(cls: type, num_velocity_fields: Optional[int], advection_channels: int) -> int:
    if num_velocity_fields is None:
        LOGGER.debug(
            f"{cls.__name__}: num_velocity_fields not specified, using {advection_channels}, "
            "one velocity field per advected channel"
        )
        return advection_channels
    return num_velocity_fields


class LatLonAdvection(Advection):
    """Advection step on a regular latitude-longitude grid.

    The spatial mixers are a stencil per channel and a channel mix, and the values are read at
    the departure points with ``F.grid_sample`` (``SemiLagrangianLayer``).
    """

    def __init__(
        self,
        *,
        num_channels: int,
        nlat: int,
        nlon: int,
        time_step: float,
        layer_kernels: DotDict,
        advection_channels: int = 768,
        num_velocity_fields: Optional[int] = None,
        velocity_hidden_dim: int = 384,
        kernel_size: int = 5,
        interpolation: str = "bicubic",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        cartesian_displacement: bool = False,
    ) -> None:
        num_velocity_fields = _default_velocity_fields(self.__class__, num_velocity_fields, advection_channels)
        transport = SemiLagrangianAdvection(
            num_channels=num_channels,
            advection_channels=advection_channels,
            num_velocity_fields=num_velocity_fields,
            velocity_hidden_dim=velocity_hidden_dim,
            time_step=time_step,
            layer_kernels=layer_kernels,
            sampler=SemiLagrangianLayer(
                nlat=nlat,
                nlon=nlon,
                num_heads=num_velocity_fields,
                interpolation=interpolation,
                cartesian_displacement=cartesian_displacement,
            ),
            mixer=lambda i, o: LatLonSpatialMixer(i, o, layer_kernels, kernel_size),
            bias=lambda c: LatLonLowRankBias(
                c, nlat, nlon, layer_kernels, rank=bias_rank, num_base_maps=bias_base_maps
            ),
        )
        super().__init__(num_channels=num_channels, transport=transport)


class GraphAdvection(Advection):
    """Advection step on the nodes of a graph.

    The spatial mixers are a block over the hidden-to-hidden edges and a channel mix (see
    ``GraphSpatialMixer``; ``mixer_block`` chooses ``"edge_conv"``, ``"transformer"`` or
    ``"gnn"``, and ``mixer_kwargs`` go to the block), and the values are read at the departure
    points from their nearest nodes (``GraphSemiLagrangianLayer``, with ``num_neighbours`` and
    ``smoothing``).

    The number of neighbour searches is ``batch * num_velocity_fields * nodes``, so a velocity
    field per advected channel, as on a grid, is slow with the brute-force search.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        edge_dim: int,
        time_step: float,
        layer_kernels: DotDict,
        advection_channels: int = 768,
        num_velocity_fields: Optional[int] = None,
        velocity_hidden_dim: int = 384,
        mixer_block: str = "edge_conv",
        mixer_kwargs: Optional[dict] = None,
        num_neighbours: int = 4,
        smoothing: float = 0.01,
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        bias_lat_degree: int = 32,
        bias_lon_wavenumber: int = 32,
        cartesian_displacement: bool = False,
    ) -> None:
        num_velocity_fields = _default_velocity_fields(self.__class__, num_velocity_fields, advection_channels)
        mixer_kwargs = dict(mixer_kwargs or {})
        transport = SemiLagrangianAdvection(
            num_channels=num_channels,
            advection_channels=advection_channels,
            num_velocity_fields=num_velocity_fields,
            velocity_hidden_dim=velocity_hidden_dim,
            time_step=time_step,
            layer_kernels=layer_kernels,
            sampler=GraphSemiLagrangianLayer(
                num_heads=num_velocity_fields,
                num_neighbours=num_neighbours,
                smoothing=smoothing,
                cartesian_displacement=cartesian_displacement,
            ),
            mixer=lambda i, o: GraphSpatialMixer(i, o, edge_dim, layer_kernels, block=mixer_block, **mixer_kwargs),
            bias=lambda c: GraphLowRankBias(
                c,
                layer_kernels,
                rank=bias_rank,
                num_base_maps=bias_base_maps,
                lat_degree=bias_lat_degree,
                lon_wavenumber=bias_lon_wavenumber,
            ),
        )
        super().__init__(num_channels=num_channels, transport=transport)
