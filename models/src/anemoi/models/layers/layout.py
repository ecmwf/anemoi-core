# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data.storage import NodeStorage
from torch_geometric.typing import Adj

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.halo import HaloInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes


def latlon_cell_centres(nlat: int, nlon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Latitudes and longitudes in radians of the grid points, one entry per node in storage order."""
    lat = torch.pi / 2 - (torch.arange(nlat, dtype=torch.float64) + 0.5) * torch.pi / nlat
    lon = torch.arange(nlon, dtype=torch.float64) * 2 * torch.pi / nlon
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return lat.reshape(-1), lon.reshape(-1)


def latlon_to_unit_vector(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """3D unit vectors, in the last dimension, of points given by latitude and longitude in radians."""
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)


def unit_vector_to_latlon(position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Latitude and longitude in radians, longitude in [0, 2 pi), of 3D unit vectors in the last dimension."""
    x, y, z = position.unbind(-1)
    return torch.atan2(z, torch.hypot(x, y)), torch.remainder(torch.atan2(y, x), 2 * torch.pi)


def latlon_grid_layout(
    hidden_nodes: NodeStorage, layout_node_attribute_name: str
) -> tuple[int, int, Optional[torch.Tensor]]:
    """Size of the hidden grid and the grid point of each node, read from the hidden nodes of the graph.

    hidden_nodes holds the node coordinates ``x`` (latitude and longitude in radians), the node
    attribute ``layout_node_attribute_name`` with the row and column of each node, of shape
    (num_nodes, 2), and the grid size ``_layout_shape``.
    ``anemoi.graphs.nodes.RectilinearNodes``, without ``include_poles``, with a
    ``anemoi.graphs.nodes.attributes.RectilinearLayoutIndex`` attribute builds such nodes. Every
    grid point must be a node, at the cell centre described at the top of this module.

    Returns the number of latitudes, the number of longitudes, and the flat grid position
    ``row * nlon + col`` of each node, or None when the nodes are stored row by row.
    Raises a ValueError for any other grid.
    """
    if layout_node_attribute_name not in hidden_nodes:
        raise ValueError(
            f'The hidden nodes have no layout attribute "{layout_node_attribute_name}". Build the hidden nodes with'
            f' anemoi.graphs.nodes.RectilinearNodes and add a "{layout_node_attribute_name}" attribute built with '
            f"anemoi.graphs.nodes.attributes.RectilinearLayoutIndex."
        )

    index = hidden_nodes[layout_node_attribute_name].detach().to(device="cpu", dtype=torch.long)
    coords = hidden_nodes["x"].detach().to(device="cpu", dtype=torch.float64)
    nlat, nlon = (index.max(0).values + 1).tolist()
    if index.shape != (nlat * nlon, 2):
        raise ValueError(
            f"The hidden grid has {nlat} x {nlon} points, but there are {coords.shape[0]} hidden nodes. "
            "Every grid point must be a node: build the hidden nodes with anemoi.graphs.nodes.RectilinearNodes."
        )

    grid_index = index[:, 0] * nlon + index[:, 1]
    outside = (index < 0).any(dim=1) | (index[:, 0] >= nlat) | (index[:, 1] >= nlon)
    if outside.any() or torch.bincount(grid_index.clamp(0, nlat * nlon - 1), minlength=nlat * nlon).ne(1).any():
        raise ValueError(
            f'The layout attribute "{layout_node_attribute_name}" does not give each grid point exactly one node. '
            "Build it with anemoi.graphs.nodes.attributes.RectilinearLayoutIndex on anemoi.graphs.nodes.RectilinearNodes."
        )

    lat, lon = latlon_cell_centres(nlat, nlon)
    lat_error = (coords[:, 0] - lat[grid_index]).abs().max()
    lon_error = (torch.remainder(coords[:, 1] - lon[grid_index] + torch.pi, 2 * torch.pi) - torch.pi).abs().max()
    if lat_error >= 1e-5 or lon_error >= 1e-5:
        raise ValueError(
            "The hidden nodes are not at the cell centres of a regular latitude-longitude grid, with rows "
            "from north to south and longitudes running east from zero. Build the hidden nodes with "
            "anemoi.graphs.nodes.RectilinearNodes, without include_poles."
        )

    if torch.equal(grid_index, torch.arange(nlat * nlon)):
        return nlat, nlon, None

    return nlat, nlon, grid_index


def _comm_size_and_rank(model_comm_group: Optional[ProcessGroup]) -> tuple[int, int]:
    if model_comm_group is None:
        return 1, 0
    return torch.distributed.get_world_size(group=model_comm_group), torch.distributed.get_rank(group=model_comm_group)


class _SplitLayout:
    """Shared by the layouts: the GPUs of one model, the node split the model hands over, and the
    split by channel group used by steps that need the whole field.

    The model splits the batch and the nodes together, so the node shard sizes describe a single
    globe only for a batch of one. Without splitting, every GPU holds the whole batch.
    """

    def __init__(
        self,
        num_nodes: int,
        batch_size: int,
        grid_shard_sizes: ShardSizes,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.batch_size = batch_size
        self.model_comm_group = model_comm_group
        self.comm_size, self.rank = _comm_size_and_rank(model_comm_group)

        if self.comm_size > 1:
            if batch_size != 1:
                raise ValueError("Only batch size of 1 is supported when the model is sharded across GPUs.")
            self.node_shard_sizes = list(grid_shard_sizes)
        else:
            self.node_shard_sizes = [sum(grid_shard_sizes) // batch_size]

        if sum(self.node_shard_sizes) != num_nodes:
            raise ValueError(
                f"The processor is given {sum(self.node_shard_sizes)} hidden nodes, but its hidden graph has "
                f"{num_nodes}."
            )

    @property
    def local_nodes(self) -> tuple[int, int]:
        """Range [start, end) of the nodes, in graph order, held by this GPU when split by node."""
        return get_partition_range(self.node_shard_sizes, self.rank)

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


class LatLonLayout(_SplitLayout):
    """Puts the processor state on the latitude-longitude grid, and switches it between two splits across GPUs.

    The model passes the processor a flat tensor of shape (batch * local nodes, channels), split by
    node, with the nodes in graph order. ``to_grid`` turns it into a tensor of shape
    (batch, local rows, nlon, channels): each GPU holds a band of whole latitude rows, with every
    channel. ``from_grid`` turns it back. The processor does this once, on the way in and on the way
    out, and every layer works on the grid.

    Steps that look at other points (stencils, interpolation) need the whole globe. ``to_globe``
    switches a tensor of shape (batch, local rows, nlon, groups, k) to one of shape
    (batch, local groups, k, nlat, nlon): the whole globe for some of the groups. The groups are the
    unit of that split, so channels that belong together (for instance the channels moved by one
    velocity field) always land on the same GPU. ``from_globe`` switches back.

    ``grid_index`` gives the flat grid position ``row * nlon + col`` of each node, when the nodes
    are not stored row by row.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        batch_size: int,
        grid_shard_sizes: ShardSizes,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_index: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(nlat * nlon, batch_size, grid_shard_sizes, model_comm_group)
        self.nlat = nlat
        self.nlon = nlon
        self.grid_index = grid_index
        if nlat < self.comm_size:
            raise ValueError(f"Cannot split {nlat} latitude rows over {self.comm_size} GPUs.")
        self.row_shard_sizes = get_balanced_partition_sizes(nlat, self.comm_size)

    @property
    def local_rows(self) -> tuple[int, int]:
        """Range [start, end) of the latitude rows held by this GPU on the grid."""
        return get_partition_range(self.row_shard_sizes, self.rank)

    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """(batch * local nodes, channels) -> (batch, local rows, nlon, channels).

        The nodes are first gathered into whole globes for a share of the channels, put in grid
        order, then split again into bands of rows. Across GPUs this needs at least as many
        channels as GPUs.
        """
        num_channels = x.shape[-1]
        channel_shard_sizes = self.group_shard_sizes(num_channels)
        x = x.view(self.batch_size, -1, num_channels)
        x = all_to_all_transpose(x, 2, channel_shard_sizes, 1, self.node_shard_sizes, self.model_comm_group)
        if self.grid_index is not None:
            x = x.new_zeros(x.shape).index_copy(1, self.grid_index, x)
        band_sizes = [rows * self.nlon for rows in self.row_shard_sizes]
        x = all_to_all_transpose(x, 1, band_sizes, 2, channel_shard_sizes, self.model_comm_group)
        return x.view(self.batch_size, -1, self.nlon, num_channels)

    def from_grid(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, local rows, nlon, channels) -> (batch * local nodes, channels)."""
        num_channels = x.shape[-1]
        channel_shard_sizes = self.group_shard_sizes(num_channels)
        band_sizes = [rows * self.nlon for rows in self.row_shard_sizes]
        x = x.reshape(self.batch_size, -1, num_channels)
        x = all_to_all_transpose(x, 2, channel_shard_sizes, 1, band_sizes, self.model_comm_group)
        if self.grid_index is not None:
            x = x.index_select(1, self.grid_index)
        x = all_to_all_transpose(x, 1, self.node_shard_sizes, 2, channel_shard_sizes, self.model_comm_group)
        return x.reshape(-1, num_channels)

    def to_globe(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, local rows, nlon, groups, k) -> (batch, local groups, k, nlat, nlon)."""
        num_groups = x.shape[3]
        x = all_to_all_transpose(
            x, 3, self.group_shard_sizes(num_groups), 1, self.row_shard_sizes, self.model_comm_group
        )
        return einops.rearrange(x, "b h w g k -> b g k h w")

    def from_globe(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        """(batch, local groups, k, nlat, nlon) -> (batch, local rows, nlon, groups, k)."""
        x = einops.rearrange(x, "b g k h w -> b h w g k")
        return all_to_all_transpose(
            x, 1, self.row_shard_sizes, 3, self.group_shard_sizes(num_groups), self.model_comm_group
        )


class GraphLayout(_SplitLayout):
    """The processor state on the nodes of the hidden graph, and the graph that the steps read.

    The state keeps the shape the model passes in, (batch * local nodes, channels), with the nodes
    in graph order: each GPU holds every channel for one contiguous block of nodes. Graph blocks
    read the hidden-to-hidden edges, which the processor prepares once per forward (sorted by
    destination node, split across GPUs, and with the halo exchange plan of the graph
    transformer).

    The semi-Lagrangian step needs the whole field. ``to_field`` switches a tensor of shape
    (batch * local nodes, groups, k) to one of shape (batch, local groups, k, nodes): every node for
    some of the groups. ``from_field`` switches back.

    ``lat`` and ``lon`` hold the latitude and longitude in radians of every node, in graph order.
    ``position`` gives the same nodes as 3D unit vectors, derived once per forward. ``spacing`` is
    a typical distance between neighbouring nodes, as a chord of the unit sphere.
    """

    def __init__(
        self,
        *,
        lat: torch.Tensor,
        lon: torch.Tensor,
        spacing: float,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        halo_info: Optional[HaloInfo] = None,
    ) -> None:
        super().__init__(lat.shape[0], batch_size, shard_info.nodes, model_comm_group)
        self.lat = lat
        self.lon = lon
        self.spacing = spacing
        self.shard_info = shard_info
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.halo_info = halo_info
        self._position: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        """Number of rows of the state over all GPUs; the attention size of the graph blocks."""
        return self.batch_size * self.lat.shape[0]

    @property
    def position(self) -> torch.Tensor:
        """3D unit vectors of the nodes, in graph order, derived from ``lat`` and ``lon``.

        Latitude and longitude are the stored form: they are exact at the poles, where recovering
        the longitude from a unit vector is ill-conditioned.
        """
        if self._position is None:
            self._position = latlon_to_unit_vector(self.lat, self.lon)
        return self._position

    def to_field(self, x: torch.Tensor) -> torch.Tensor:
        """(batch * local nodes, groups, k) -> (batch, local groups, k, nodes)."""
        num_groups = x.shape[1]
        x = x.view(self.batch_size, -1, num_groups, x.shape[-1])
        x = all_to_all_transpose(
            x, 2, self.group_shard_sizes(num_groups), 1, self.node_shard_sizes, self.model_comm_group
        )
        return einops.rearrange(x, "b n g k -> b g k n")

    def from_field(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        """(batch, local groups, k, nodes) -> (batch * local nodes, groups, k)."""
        x = einops.rearrange(x, "b g k n -> b n g k")
        x = all_to_all_transpose(
            x, 1, self.node_shard_sizes, 2, self.group_shard_sizes(num_groups), self.model_comm_group
        )
        return x.reshape(-1, num_groups, x.shape[-1])
