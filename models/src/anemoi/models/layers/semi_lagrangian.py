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
degrees. Nodes are stored row by row, so node ``n`` sits in row ``n // nlon`` and column
``n % nlon``. No point lies on a pole.

Across GPUs the processor state is split by grid point: each GPU holds every channel for one
contiguous block of nodes. Steps that look at other points (interpolation, stencils) need
the whole globe, so for those steps the state is exchanged so that each GPU holds the whole globe
for a subset of channels, and exchanged back afterwards.
"""

from typing import Optional

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.utils.config import DotDict

# Smallest squared displacement treated as non-zero. Keeps the square root and its gradient finite
# when a displacement is exactly zero.
_MIN_SQUARED_ANGLE = 1e-24

# Rotation rate of the Earth in radians per second. Learned velocities are measured in radians per
# 1 / EARTH_ROTATION_RATE seconds.
EARTH_ROTATION_RATE = 7.29212e-5


def _at_least_float32(x: Tensor) -> Tensor:
    """x in float32, or in its own precision if that is higher."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def latlon_cell_centres(nlat: int, nlon: int) -> tuple[Tensor, Tensor]:
    """Latitudes and longitudes in radians of the grid points, one entry per node in storage order."""
    lat = torch.pi / 2 - (torch.arange(nlat, dtype=torch.float64) + 0.5) * torch.pi / nlat
    lon = torch.arange(nlon, dtype=torch.float64) * 2 * torch.pi / nlon
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return lat.reshape(-1), lon.reshape(-1)


def latlon_grid_shape(node_coordinates: Tensor) -> tuple[int, int]:
    """Number of latitudes and longitudes of the hidden grid, read from its node coordinates.

    node_coordinates has shape (num_nodes, 2) and holds latitude and longitude in radians, as
    stored in the graph. The nodes must be the cell centres of a regular grid in the order
    described at the top of this module; ``anemoi.graphs.nodes.RegularLatLonNodes`` builds such
    a grid. Raises a ValueError for any other grid.
    """
    coords = node_coordinates.detach().to(device="cpu", dtype=torch.float64)
    num_nodes = coords.shape[0]
    nlat = int(torch.unique(coords[:, 0].round(decimals=6)).numel())
    nlon = num_nodes // nlat

    if nlat * nlon == num_nodes:
        lat, lon = latlon_cell_centres(nlat, nlon)
        lat_error = (coords[:, 0] - lat).abs().max()
        lon_error = (torch.remainder(coords[:, 1] - lon + torch.pi, 2 * torch.pi) - torch.pi).abs().max()
        if lat_error < 1e-5 and lon_error < 1e-5:
            return nlat, nlon

    raise ValueError(
        "The hidden nodes are not a regular latitude-longitude grid at cell centres, stored row by row "
        "from north to south with longitudes running east from zero. Build the hidden nodes with "
        "anemoi.graphs.nodes.RegularLatLonNodes."
    )


def geocyclic_pad(x: Tensor, pad: int) -> Tensor:
    """Pad a field of shape (..., nlat, nlon) so that it carries on across the poles and around in longitude.

    A path that goes over a pole comes down the other side half a turn further round, so the rows
    added beyond a pole are the rows next to it, in reverse order and shifted by half the longitudes.
    Columns wrap around. Needs an even ``nlon``.
    """
    half_turn = x.shape[-1] // 2
    top = x[..., :pad, :].flip(-2).roll(half_turn, dims=-1)
    bottom = x[..., -pad:, :].flip(-2).roll(half_turn, dims=-1)
    x = torch.cat([top, x, bottom], dim=-2)
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


def _comm_size_and_rank(model_comm_group: Optional[ProcessGroup]) -> tuple[int, int]:
    if model_comm_group is None:
        return 1, 0
    return dist.get_world_size(group=model_comm_group), dist.get_rank(group=model_comm_group)


class LatLonLayout:
    """Switches the processor state between being split by grid point and being split by channel group.

    Split by grid point, each GPU holds a tensor of shape (batch * local nodes, groups, k): every
    channel group for its own block of nodes. Split by channel group, each GPU holds a tensor of
    shape (batch, local groups, k, nlat, nlon): the whole globe for some of the groups. The groups
    are the unit of the split, so channels that belong together (for instance the channels moved
    by one velocity field) always land on the same GPU.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        batch_size: int,
        grid_shard_sizes: ShardSizes,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.nlat = nlat
        self.nlon = nlon
        self.batch_size = batch_size
        self.model_comm_group = model_comm_group
        self.comm_size, self.rank = _comm_size_and_rank(model_comm_group)

        # The model splits the batch and the nodes together, so the shard sizes describe a single
        # globe only for a batch of one. Without splitting, every GPU holds the whole batch.
        if self.comm_size > 1:
            if batch_size != 1:
                raise ValueError("Only batch size of 1 is supported when the model is sharded across GPUs.")
            self.grid_shard_sizes = list(grid_shard_sizes)
        else:
            self.grid_shard_sizes = [sum(grid_shard_sizes) // batch_size]

        if sum(self.grid_shard_sizes) != nlat * nlon:
            raise ValueError(
                f"The hidden grid has {sum(self.grid_shard_sizes)} nodes, but nlat * nlon = {nlat * nlon}. "
                "The processor needs a regular latitude-longitude hidden grid."
            )

    @property
    def local_nodes(self) -> tuple[int, int]:
        """Range [start, end) of the nodes held by this GPU when split by grid point."""
        return get_partition_range(self.grid_shard_sizes, self.rank)

    def group_shard_sizes(self, num_groups: int) -> list[int]:
        if num_groups < self.comm_size:
            raise ValueError(
                f"Cannot split {num_groups} channel groups over {self.comm_size} GPUs; "
                "use at least as many groups as GPUs per model."
            )
        return get_balanced_partition_sizes(num_groups, self.comm_size)

    def local_groups(self, num_groups: int) -> tuple[int, int]:
        """Range [start, end) of the channel groups held by this GPU when split by channel group."""
        return get_partition_range(self.group_shard_sizes(num_groups), self.rank)

    def to_globe(self, x: Tensor) -> Tensor:
        """(batch * local nodes, groups, k) -> (batch, local groups, k, nlat, nlon)."""
        num_groups = x.shape[1]
        x = x.view(self.batch_size, -1, num_groups, x.shape[-1])
        x = all_to_all_transpose(
            x, 2, self.group_shard_sizes(num_groups), 1, self.grid_shard_sizes, self.model_comm_group
        )
        return einops.rearrange(x, "b (h w) g k -> b g k h w", h=self.nlat, w=self.nlon)

    def from_globe(self, x: Tensor, num_groups: int) -> Tensor:
        """(batch, local groups, k, nlat, nlon) -> (batch * local nodes, groups, k)."""
        x = einops.rearrange(x, "b g k h w -> b (h w) g k")
        x = all_to_all_transpose(
            x, 1, self.grid_shard_sizes, 2, self.group_shard_sizes(num_groups), self.model_comm_group
        )
        return x.reshape(-1, num_groups, x.shape[-1])


def init_branch(*layers: nn.Module, last_scale: float = 0.1) -> None:
    """Initialise the layers of one branch, given in the order they are applied.

    Weights are drawn with Kaiming-normal scaling (fan-in, ReLU gain) and biases start at zero.
    The last layer's weights are scaled down so that the branch starts with a small output.
    """
    for i, layer in enumerate(layers):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        if i == len(layers) - 1:
            with torch.no_grad():
                layer.weight.mul_(last_scale)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class GeocyclicDepthwiseConv(nn.Module):
    """A learned square stencil applied to every channel on its own, continuing across the poles."""

    def __init__(self, num_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.num_channels = num_channels
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size, groups=num_channels, padding=0, bias=False)

    def forward(self, x: Tensor, layout: LatLonLayout) -> Tensor:
        # Each GPU applies the stencil to the whole globe for its share of the channels, using
        # its share of the stencil weights.
        start, end = layout.local_groups(self.num_channels)
        y = layout.to_globe(x.view(-1, self.num_channels, 1))
        y = geocyclic_pad(y.flatten(1, 2), self.pad)
        y = F.conv2d(y, self.conv.weight[start:end], groups=end - start)
        y = y.view(layout.batch_size, end - start, 1, layout.nlat, layout.nlon)
        return layout.from_globe(y, self.num_channels).view(-1, self.num_channels)


class SpatialMixer(nn.Module):
    """A stencil per channel followed by a linear mix of the channels at each point."""

    def __init__(self, in_channels: int, out_channels: int, layer_kernels: DotDict, kernel_size: int = 5) -> None:
        super().__init__()
        self.stencil = GeocyclicDepthwiseConv(in_channels, kernel_size)
        self.mix = layer_kernels.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, layout: LatLonLayout) -> Tensor:
        return self.mix(self.stencil(x, layout))


class LowRankBias(nn.Module):
    """A learned field over the globe, built from a few smooth patterns so that it needs few parameters.

    A small number of base maps is formed as sums of products of a latitude profile and a
    longitude profile, then mixed linearly into the output channels. The profiles start very
    small, so the field starts close to zero.
    """

    def __init__(
        self,
        num_channels: int,
        nlat: int,
        nlon: int,
        layer_kernels: DotDict,
        rank: int = 128,
        num_base_maps: int = 8,
    ) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(num_base_maps, rank) * 1e-3)
        self.lat_profiles = nn.Parameter(torch.randn(rank, nlat) * 1e-3)
        self.lon_profiles = nn.Parameter(torch.randn(rank, nlon) * 1e-3)
        self.projection = layer_kernels.Linear(num_base_maps, num_channels, bias=False)

    def forward(self, x: Tensor, layout: LatLonLayout) -> Tensor:
        """Add the field to x, of shape (batch * local nodes, channels)."""
        base_maps = torch.einsum("ck,ki,kj->ijc", self.coefficients, self.lat_profiles, self.lon_profiles)
        start, end = layout.local_nodes
        field = self.projection(base_maps.reshape(-1, base_maps.shape[-1])[start:end])
        return (x.view(layout.batch_size, end - start, -1) + field).view(x.shape)


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
        self.register_buffer("arrival_lat", lat.float(), persistent=False)
        self.register_buffer("arrival_lon", lon.float(), persistent=False)
        if cartesian_displacement:
            position = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)
            self.register_buffer("arrival_position", position.float(), persistent=False)

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

        values has shape (batch * local nodes, heads, channels per head) and displacement has shape
        (batch * local nodes, heads, displacement_dim). Returns a tensor shaped like values.
        """
        with torch.autocast(device_type=values.device.type, enabled=False):
            grid = layout.to_globe(self.departure_points(_at_least_float32(displacement), layout))
            grid = einops.rearrange(grid, "b g k h w -> (b g) h w k")

            globe = layout.to_globe(values)
            local_heads = globe.shape[1]
            globe = geocyclic_pad(_at_least_float32(globe.flatten(0, 1)), self.pad)

            moved = F.grid_sample(globe, grid, mode=self.interpolation, padding_mode="zeros", align_corners=True)
            moved = moved.to(values.dtype).view(layout.batch_size, local_heads, -1, self.nlat, self.nlon)

        return layout.from_globe(moved, self.num_heads)


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
    ) -> None:
        super().__init__(
            nlat=nlat,
            nlon=nlon,
            num_heads=num_heads,
            interpolation=interpolation,
            cartesian_displacement=cartesian_displacement,
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
    ) -> None:
        super().__init__(
            nlat=nlat,
            nlon=nlon,
            num_heads=num_heads,
            interpolation=interpolation,
            cartesian_displacement=cartesian_displacement,
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
