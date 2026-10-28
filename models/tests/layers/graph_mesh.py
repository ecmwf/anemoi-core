# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A small unstructured mesh for testing the graph semi-Lagrangian layers and processors."""

import math

import torch

from anemoi.models.layers.semi_lagrangian import knn_on_sphere
from anemoi.models.layers.semi_lagrangian import latlon_to_unit_vector


def fibonacci_nodes(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Latitudes and longitudes in radians (float64) of nodes spread evenly over the sphere."""
    i = torch.arange(num_nodes, dtype=torch.float64) + 0.5
    lat = torch.asin(1 - 2 * i / num_nodes)
    lon = torch.remainder(i * math.pi * (3 - math.sqrt(5)), 2 * math.pi)
    return lat, lon


def knn_edges(lat: torch.Tensor, lon: torch.Tensor, num_neighbours: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges from the nearest nodes to each node, sorted by destination node.

    The edge attributes are the 3D offset from the destination to the source node and its length.
    """
    position = latlon_to_unit_vector(lat, lon).float()
    neighbours = knn_on_sphere(position, position, num_neighbours + 1)[:, 1:]
    dst = torch.arange(lat.shape[0]).repeat_interleave(num_neighbours)
    src = neighbours.reshape(-1)
    offset = position[src] - position[dst]
    edge_attr = torch.cat([offset, offset.norm(dim=-1, keepdim=True)], dim=-1)
    return edge_attr, torch.stack([src, dst])


def expand_edges(
    edge_attr: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges for a batch, as the graph providers give them: one copy of the graph per batch member."""
    edge_index = torch.cat([edge_index + i * num_nodes for i in range(batch_size)], dim=1)
    return edge_attr.repeat(batch_size, 1), edge_index


def hidden_nodes(lat: torch.Tensor, lon: torch.Tensor) -> dict:
    """Hidden nodes as a graph stores them: float32 coordinates, latitude first, in radians."""
    return {"x": torch.stack([lat, lon], dim=-1).float()}
