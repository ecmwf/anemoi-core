# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Building blocks for learned semi-Lagrangian processors on a regular latitude-longitude grid.

The hidden grid has ``nlat x nlon`` points at cell centres. Latitudes run from north to south,
``90 - (i + 0.5) * 180 / nlat`` degrees, and longitudes run east from zero, ``j * 360 / nlon``
degrees. No point lies on a pole. The hidden nodes carry their row and column in a node
attribute (``anemoi.graphs.nodes.attributes.RectilinearLayoutIndex``). Nodes are usually stored
row by row, so node ``n`` sits in row ``n // nlon`` and column ``n % nlon``; any other order is
mapped onto the grid with the layout attribute.

Across GPUs the processor state is split by grid point: each GPU holds every channel for one
contiguous block of nodes. Steps that look at other points (interpolation, stencils) need
the whole globe, so for those steps the state is exchanged so that each GPU holds the whole globe
for a subset of channels, and exchanged back afterwards.
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from anemoi.models.layers.geocyclic import geocyclic_pad
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.layout import latlon_cell_centres
from anemoi.utils.config import DotDict

# Smallest squared displacement treated as non-zero. Keeps the square root and its gradient finite
# when a displacement is exactly zero.
_MIN_SQUARED_ANGLE = 1e-24


def _at_least_float32(x: torch.Tensor) -> torch.Tensor:
    """x in float32, or in its own precision if that is higher."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


class SemiLagrangianLayer(nn.Module):
    """Moves groups of channels along learned paths on the sphere.

    Every group of channels has its own displacement at every grid point. The new value at a
    point is read, by interpolation, from the departure point: where the value would have come
    from by moving backwards by that displacement.

    By default a displacement is the pair of angles travelled towards the local east and the local
    north, and the departure point is worked out in coordinates rotated so that the arrival point
    lies on their equator (Ritchie, 1987). These local directions turn round across a pole. With
    ``cartesian_displacement`` a displacement is instead a 3D vector: only its part along the
    sphere's surface is used, its length is the angle travelled, and the departure point lies on
    the great circle through the arrival point in that direction. A 3D vector means the same on
    both sides of a pole.

    The departure points are worked out and the interpolation is done in at least float32, also under
    mixed precision: in bfloat16 a position on a 720-point circle can be off by most of a grid cell.

    ``grid_index`` gives the flat grid position ``row * nlon + col`` of each node, when the nodes
    are not stored row by row (see ``latlon_grid_layout``).

    Subclasses decide what is moved and how the displacements are predicted, and call ``sample``.
    """

    def __init__(
        self,
        *,
        nlat: int,
        nlon: int,
        num_heads: int,
        interpolation: str = "bicubic",
        cartesian_displacement: bool = False,
        grid_index: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if nlon % 2 != 0:
            raise ValueError(f"nlon must be even so that the grid continues across the poles, got {nlon}.")
        if interpolation not in ("bilinear", "bicubic"):
            raise ValueError(f"interpolation must be 'bilinear' or 'bicubic', got {interpolation}.")

        self.nlat = nlat
        self.nlon = nlon
        self.num_heads = num_heads
        self.interpolation = interpolation
        # Bicubic interpolation reads two points on each side, bilinear one.
        self.pad = 2 if interpolation == "bicubic" else 1

        self.cartesian_displacement = cartesian_displacement
        self.displacement_dim = 3 if cartesian_displacement else 2

        lat, lon = latlon_cell_centres(nlat, nlon)
        if grid_index is not None:
            lat, lon = lat[grid_index], lon[grid_index]
        self.register_buffer("arrival_lat", lat.float(), persistent=False)
        self.register_buffer("arrival_lon", lon.float(), persistent=False)
        if cartesian_displacement:
            position = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)
            self.register_buffer("arrival_position", position.float(), persistent=False)

        # Lazy init. Hook for diagnostics.
        self.diag_hook = None

    def departure_points(self, displacement: Tensor, layout: LatLonLayout) -> Tensor:
        """Positions, in grid_sample's [-1, 1] coordinates of the padded globe, to read each value from.

        displacement has shape (batch * local nodes, heads, displacement_dim) and is in at least float32: the
        eastward and northward angles in radians, or a 3D vector with ``cartesian_displacement``.
        Returns (batch * local nodes, heads, 2) with the column position first, as grid_sample
        expects.
        """
        start, end = layout.local_nodes
        displacement = displacement.view(layout.batch_size, end - start, self.num_heads, self.displacement_dim)
        if self.cartesian_displacement:
            departure_lat, departure_lon = self._departure_from_vector(displacement, start, end)
        else:
            departure_lat, departure_lon = self._departure_from_local_angles(displacement, start, end)

        row = (torch.pi / 2 - departure_lat) * (self.nlat / torch.pi) - 0.5 + self.pad
        col = departure_lon * (self.nlon / (2 * torch.pi)) + self.pad
        grid_x = 2 * col / (self.nlon + 2 * self.pad - 1) - 1
        grid_y = 2 * row / (self.nlat + 2 * self.pad - 1) - 1
        return torch.stack([grid_x, grid_y], dim=-1).view(-1, self.num_heads, 2)

    def _departure_from_local_angles(self, displacement: Tensor, start: int, end: int) -> tuple[Tensor, Tensor]:
        lat = self.arrival_lat[start:end].view(1, end - start, 1)
        lon = self.arrival_lon[start:end].view(1, end - start, 1)

        # Position of the departure point in the rotated coordinates, then the same point in
        # ordinary latitude and longitude.
        lon_rot = -displacement[..., 0]
        lat_rot = -displacement[..., 1]
        sin_lat = lat_rot.sin() * lat.cos() + lat_rot.cos() * lon_rot.cos() * lat.sin()
        east = lat_rot.cos() * lon_rot.sin()
        outward = lat_rot.cos() * lon_rot.cos() * lat.cos() - lat_rot.sin() * lat.sin()
        departure_lat = torch.atan2(sin_lat, torch.hypot(east, outward))
        departure_lon = torch.remainder(lon + torch.atan2(east, outward), 2 * torch.pi)
        return departure_lat, departure_lon

    def _departure_from_vector(self, displacement: Tensor, start: int, end: int) -> tuple[Tensor, Tensor]:
        position = self.arrival_position[start:end].view(1, end - start, 1, 3)

        # Keep only the part of the displacement along the sphere's surface, then go back along
        # the great circle by that angle.
        tangent = displacement - (displacement * position).sum(-1, keepdim=True) * position
        angle = tangent.square().sum(-1, keepdim=True).clamp_min(_MIN_SQUARED_ANGLE).sqrt()
        departure = angle.cos() * position - (angle.sin() / angle) * tangent

        x, y, z = departure.unbind(-1)
        departure_lat = torch.atan2(z, torch.hypot(x, y))
        departure_lon = torch.remainder(torch.atan2(y, x), 2 * torch.pi)
        return departure_lat, departure_lon

    def sample(self, values: Tensor, displacement: Tensor, layout: LatLonLayout) -> Tensor:
        """Read every value from its departure point.

        Arguments
        ---------
        values : torch.Tensor
            The values to be sampled, with shape (batch * local nodes, heads, channels per head).
        displacement : torch.Tensor
            The displacement vectors, with shape (batch * local nodes, heads, displacement_dim).
        layout : LatLonLayout
            The layout object describing the latitude-longitude grid.

        Returns
        -------
        torch.Tensor
            The sampled values, with the same shape as the input `values`.
        """
        with torch.autocast(device_type=values.device.type, enabled=False):
            grid = layout.to_globe(self.departure_points(_at_least_float32(displacement), layout))
            grid = einops.rearrange(grid, "b g k h w -> (b g) h w k")

            if self.diag_hook is not None:
                self.diag_hook(self, grid[:, :, :, 0], grid[:, :, :, 1])

            globe = layout.to_globe(values)
            local_heads = globe.shape[1]
            globe = geocyclic_pad(_at_least_float32(globe.flatten(0, 1)), self.pad)

            moved = F.grid_sample(globe, grid, mode=self.interpolation, padding_mode="zeros", align_corners=True)
            moved = moved.to(values.dtype).view(layout.batch_size, local_heads, -1, self.nlat, self.nlon)

        return layout.from_globe(moved, self.num_heads)


class SemiLagrangianWarp(SemiLagrangianLayer):
    """Learned warp from FLOWERS (Muser et al., arXiv:2603.04430), on the sphere.

    A pointwise MLP (linear layer, ReLU, linear layer) predicts one displacement per head at
    every point, and a linear layer gives the values that are moved. Each head moves
    ``out_channels // num_heads`` channels. The displacements come from each point alone, without
    looking at neighbours; values from far away reach a point only through where it reads from.
    Displacements are those described in ``SemiLagrangianLayer`` (local east and north angles in
    radians, or 3D vectors with ``cartesian_displacement``), with no time scaling.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        nlat: int,
        nlon: int,
        layer_kernels: DotDict,
        interpolation: str = "bilinear",
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
        if out_channels % num_heads != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads}).")
        self.out_channels = out_channels

        self.flow = nn.Sequential(
            layer_kernels.Linear(in_channels, out_channels),
            nn.ReLU(),
            layer_kernels.Linear(out_channels, self.displacement_dim * num_heads),
        )
        self.value = layer_kernels.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, layout: LatLonLayout) -> Tensor:
        displacement = self.flow(x).view(-1, self.num_heads, self.displacement_dim)
        values = self.value(x).view(-1, self.num_heads, self.out_channels // self.num_heads)
        moved = self.sample(values, _at_least_float32(displacement), layout)
        return moved.view(-1, self.out_channels)
