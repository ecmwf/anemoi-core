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

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes


def latlon_cell_centres(nlat: int, nlon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Latitudes and longitudes in radians of the grid points, one entry per node in storage order."""
    lat = torch.pi / 2 - (torch.arange(nlat, dtype=torch.float64) + 0.5) * torch.pi / nlat
    lon = torch.arange(nlon, dtype=torch.float64) * 2 * torch.pi / nlon
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return lat.reshape(-1), lon.reshape(-1)


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
    nlat, nlon = index.max(0).values + 1
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


class LatLonLayout:
    """Switches the processor state between being split by grid point and being split by channel group.

    Split by grid point, each GPU holds a tensor of shape (batch * local nodes, groups, k): every
    channel group for its own block of nodes. Split by channel group, each GPU holds a tensor of
    shape (batch, local groups, k, nlat, nlon): the whole globe for some of the groups. The groups
    are the unit of the split, so channels that belong together (for instance the channels moved
    by one velocity field) always land on the same GPU.

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
        self.nlat = nlat
        self.nlon = nlon
        self.grid_index = grid_index
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

    def to_globe(self, x: torch.Tensor) -> torch.Tensor:
        """(batch * local nodes, groups, k) -> (batch, local groups, k, nlat, nlon)."""
        num_groups = x.shape[1]
        x = x.view(self.batch_size, -1, num_groups, x.shape[-1])
        x = all_to_all_transpose(
            x, 2, self.group_shard_sizes(num_groups), 1, self.grid_shard_sizes, self.model_comm_group
        )
        if self.grid_index is not None:
            x = x.new_zeros(x.shape).index_copy(1, self.grid_index, x)
        return einops.rearrange(x, "b (h w) g k -> b g k h w", h=self.nlat, w=self.nlon)

    def from_globe(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        """(batch, local groups, k, nlat, nlon) -> (batch * local nodes, groups, k)."""
        x = einops.rearrange(x, "b g k h w -> b (h w) g k")
        if self.grid_index is not None:
            x = x.index_select(1, self.grid_index)
        x = all_to_all_transpose(
            x, 1, self.grid_shard_sizes, 2, self.group_shard_sizes(num_groups), self.model_comm_group
        )
        return x.reshape(-1, num_groups, x.shape[-1])
