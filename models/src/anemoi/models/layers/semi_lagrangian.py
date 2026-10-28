# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Building blocks for learned semi-Lagrangian processors, on a regular latitude-longitude grid or a graph.

The hidden grid has ``nlat x nlon`` points at cell centres. Latitudes run from north to south,
``90 - (i + 0.5) * 180 / nlat`` degrees, and longitudes run east from zero, ``j * 360 / nlon``
degrees. No point lies on a pole. The hidden nodes carry their row and column in a node
attribute (``anemoi.graphs.nodes.attributes.RectilinearLayoutIndex``).

The processor puts its state on the grid once (``LatLonLayout.to_grid``), so the layers here see
tensors of shape (batch, local rows, nlon, ...). Across GPUs each GPU holds a band of whole
latitude rows. Interpolation needs the whole globe, so for that step the state is exchanged so
that each GPU holds the whole globe for a subset of channels, and exchanged back afterwards.

``GraphSemiLagrangianLayer`` does the same on the nodes of any graph (``GraphLayout``), with
tensors of shape (batch * local nodes, ...): it computes the same departure points and reads the
values there from the nearest nodes instead of with ``F.grid_sample``.
"""

import einops
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from anemoi.models.layers.geocyclic import geocyclic_pad
from anemoi.models.layers.layout import GraphLayout
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.layout import latlon_cell_centres
from anemoi.models.layers.layout import latlon_to_unit_vector
from anemoi.models.layers.layout import unit_vector_to_latlon
from anemoi.utils.config import DotDict

# Smallest squared displacement treated as non-zero. Keeps the square root and its gradient finite
# when a displacement is exactly zero.
_MIN_SQUARED_ANGLE = 1e-24


def _at_least_float32(x: torch.Tensor) -> torch.Tensor:
    """x in float32, or in its own precision if that is higher."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def departure_from_local_angles(lat: Tensor, lon: Tensor, displacement: Tensor) -> tuple[Tensor, Tensor]:
    """Latitude and longitude of the departure points of displacements given as local east and north angles.

    lat and lon are the arrival points, broadcastable against ``displacement[..., 0]``. The
    departure point is worked out in coordinates rotated so that the arrival point lies on their
    equator (Ritchie, 1987).
    """
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


def departure_from_vector(position: Tensor, displacement: Tensor) -> Tensor:
    """3D unit vectors of the departure points of displacements given as 3D vectors.

    position holds the arrival points as 3D unit vectors, broadcastable against displacement.
    """
    # Keep only the part of the displacement along the sphere's surface, then go back along
    # the great circle by that angle.
    tangent = displacement - (displacement * position).sum(-1, keepdim=True) * position
    angle = tangent.square().sum(-1, keepdim=True).clamp_min(_MIN_SQUARED_ANGLE).sqrt()
    return angle.cos() * position - (angle.sin() / angle) * tangent


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
        lat, lon = lat.view(nlat, nlon), lon.view(nlat, nlon)
        self.register_buffer("arrival_lat", lat.float(), persistent=False)
        self.register_buffer("arrival_lon", lon.float(), persistent=False)
        if cartesian_displacement:
            self.register_buffer("arrival_position", latlon_to_unit_vector(lat, lon).float(), persistent=False)

        # Lazy init. Hook for diagnostics.
        self.diag_hook = None

    def departure_points(self, displacement: Tensor, layout: LatLonLayout) -> Tensor:
        """Positions, in grid_sample's [-1, 1] coordinates of the padded globe, to read each value from.

        displacement has shape (batch, local rows, nlon, heads, displacement_dim) and is in at least
        float32: the eastward and northward angles in radians, or a 3D vector with
        ``cartesian_displacement``. Returns (batch, local rows, nlon, heads, 2) with the column
        position first, as grid_sample expects.
        """
        start, end = layout.local_rows
        if self.cartesian_displacement:
            departure_lat, departure_lon = self._departure_from_vector(displacement, start, end)
        else:
            departure_lat, departure_lon = self._departure_from_local_angles(displacement, start, end)

        row = (torch.pi / 2 - departure_lat) * (self.nlat / torch.pi) - 0.5 + self.pad
        col = departure_lon * (self.nlon / (2 * torch.pi)) + self.pad
        grid_x = 2 * col / (self.nlon + 2 * self.pad - 1) - 1
        grid_y = 2 * row / (self.nlat + 2 * self.pad - 1) - 1
        return torch.stack([grid_x, grid_y], dim=-1)

    def _departure_from_local_angles(self, displacement: Tensor, start: int, end: int) -> tuple[Tensor, Tensor]:
        lat = self.arrival_lat[start:end, :, None]
        lon = self.arrival_lon[start:end, :, None]
        return departure_from_local_angles(lat, lon, displacement)

    def _departure_from_vector(self, displacement: Tensor, start: int, end: int) -> tuple[Tensor, Tensor]:
        position = self.arrival_position[start:end, :, None, :]
        return unit_vector_to_latlon(departure_from_vector(position, displacement))

    def sample(self, values: Tensor, displacement: Tensor, layout: LatLonLayout) -> Tensor:
        """Read every value from its departure point.

        Arguments
        ---------
        values : torch.Tensor
            The values to be sampled, with shape (batch, local rows, nlon, heads, channels per head).
        displacement : torch.Tensor
            The displacement vectors, with shape (batch, local rows, nlon, heads, displacement_dim).
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


def knn_on_sphere(points: Tensor, queries: Tensor, k: int, chunk_elements: int = 2**26) -> Tensor:
    """Indices, of shape (queries, k), of the k points nearest to each query, nearest first.

    points (n, 3) and queries (q, 3) are 3D unit vectors, so the nearest points are those with the
    largest dot product. This is a brute-force search over every point, run in chunks of queries
    of about ``chunk_elements`` dot products each: slow, but it needs nothing beyond torch and works
    on any set of nodes.
    """
    rows = max(1, chunk_elements // points.shape[0])
    with torch.no_grad():
        return torch.cat([(chunk @ points.T).topk(k, dim=-1).indices for chunk in queries.split(rows)])


def typical_node_spacing(position: Tensor, max_queries: int = 2048) -> float:
    """Median distance, as a chord of the unit sphere, from a node to its nearest other node.

    position (n, 3) holds the nodes as 3D unit vectors. Up to ``max_queries`` evenly spaced nodes
    are used.
    """
    queries = position[:: max(1, position.shape[0] // max_queries)]
    nearest = knn_on_sphere(position, queries, 2)[:, 1]
    return (queries - position[nearest]).norm(dim=-1).median().item()


class GraphSemiLagrangianLayer(nn.Module):
    """Moves groups of channels along learned paths on the sphere, on the nodes of a graph.

    The graph counterpart of ``SemiLagrangianLayer``: the same displacements (local east and north
    angles, or 3D vectors with ``cartesian_displacement``) and the same departure points. The
    value at a departure point is read from the ``num_neighbours`` nodes nearest to it, found with
    ``knn_on_sphere``, with modified Shepard (Franke-Little) weights

        w_j = (R - d_j)^2 / (d_j^2 + (smoothing * spacing)^2),    R = distance to the next nearest node,

    normalised to sum to one. The weights go to zero at R, so the result is continuous as the
    neighbours change, and they peak at a node, so a departure point on a node reads (almost) the
    node's own value. ``smoothing`` sets how sharp that peak is, relative to the node spacing.
    The gradient reaches the displacements through the distances.

    Distances are chords of the unit sphere. The search and the weights run in at least float32.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_neighbours: int = 4,
        smoothing: float = 0.01,
        cartesian_displacement: bool = False,
    ) -> None:
        super().__init__()
        if num_neighbours < 1:
            raise ValueError(f"num_neighbours must be at least 1, got {num_neighbours}.")
        self.num_heads = num_heads
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.cartesian_displacement = cartesian_displacement
        self.displacement_dim = 3 if cartesian_displacement else 2

    def departure_points(self, displacement: Tensor, layout: GraphLayout) -> Tensor:
        """3D unit vectors of the points to read each value from.

        displacement has shape (batch * local nodes, heads, displacement_dim) and is in at least
        float32. Returns (batch * local nodes, heads, 3).
        """
        start, end = layout.local_nodes
        displacement = displacement.view(layout.batch_size, end - start, self.num_heads, self.displacement_dim)
        if self.cartesian_displacement:
            position = layout.position[start:end, None, :].to(displacement.dtype)
            departure = departure_from_vector(position, displacement)
        else:
            lat = layout.lat[start:end, None].to(displacement.dtype)
            lon = layout.lon[start:end, None].to(displacement.dtype)
            departure = latlon_to_unit_vector(*departure_from_local_angles(lat, lon, displacement))
        return departure.view(-1, self.num_heads, 3)

    def sample(self, values: Tensor, displacement: Tensor, layout: GraphLayout) -> Tensor:
        """Read every value from its departure point.

        values has shape (batch * local nodes, heads, channels per head) and displacement
        (batch * local nodes, heads, displacement_dim). Returns the moved values, shaped like values.
        """
        if self.num_neighbours + 1 > layout.lat.shape[0]:
            raise ValueError(
                f"num_neighbours ({self.num_neighbours}) must be smaller than the number of hidden "
                f"nodes ({layout.lat.shape[0]}); one extra node sets the reach of the weights."
            )
        with torch.autocast(device_type=values.device.type, enabled=False):
            departure = layout.to_field(self.departure_points(_at_least_float32(displacement), layout))
            field = _at_least_float32(layout.to_field(values))
            batch_size, local_heads, channels, num_nodes = field.shape

            # One query per (batch, head, node), all against the same nodes.
            queries = einops.rearrange(departure, "b g k n -> (b g n) k")
            points = layout.position.to(queries.dtype)
            neighbours = knn_on_sphere(points, queries, self.num_neighbours + 1)
            distance = (queries[:, None, :] - points[neighbours]).square().sum(-1)
            distance = distance.clamp_min(_MIN_SQUARED_ANGLE).sqrt()
            reach = distance[:, -1:]
            distance = distance[:, :-1]
            weight = (reach - distance).square() / (distance.square() + (self.smoothing * layout.spacing) ** 2)
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)

            # Value of each neighbour, from the field of its own batch member and head.
            field = einops.rearrange(field, "b g k n -> (b g) n k")
            owner = torch.arange(batch_size * local_heads, device=field.device).repeat_interleave(num_nodes)
            neighbour_values = field[owner[:, None], neighbours[:, :-1]]
            moved = (weight[..., None] * neighbour_values).sum(1)
            moved = einops.rearrange(moved, "(b g n) k -> b g k n", b=batch_size, g=local_heads, n=num_nodes)

        return layout.from_field(moved.to(values.dtype), self.num_heads)


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
        """Warp x, of shape (batch, local rows, nlon, in_channels)."""
        displacement = self.flow(x).unflatten(-1, (self.num_heads, self.displacement_dim))
        values = self.value(x).unflatten(-1, (self.num_heads, -1))
        moved = self.sample(values, _at_least_float32(displacement), layout)
        return moved.flatten(-2)
