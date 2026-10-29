# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.builders import build_neighbour_table
from anemoi.graphs.create import GraphCreator
from anemoi.graphs.utils import haversine_distance


def _random_graph(num_points: int) -> HeteroData:
    """Points spread at random over the sphere, as latitude and longitude in radians."""
    generator = torch.Generator().manual_seed(0)
    latitudes = torch.asin(2 * torch.rand(num_points, generator=generator) - 1)
    longitudes = 2 * torch.pi * torch.rand(num_points, generator=generator)
    graph = HeteroData()
    graph["data"].x = torch.stack([latitudes, longitudes], dim=1)
    return graph


def _knn_graph_table(graph: HeteroData, num_neighbours: int) -> torch.Tensor:
    """The neighbours of the KNN edges that GraphCreator builds, grouped by the node that receives them."""
    subgraph = HeteroData()
    subgraph["data"].x = graph["data"].x
    config = OmegaConf.create(
        {
            "nodes": {},
            "edges": [
                {
                    "source_name": "data",
                    "target_name": "data",
                    "edge_builders": [
                        {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": num_neighbours}
                    ],
                }
            ],
        }
    )
    sources, targets = GraphCreator(config).update_graph(subgraph)[("data", "to", "data")].edge_index.cpu()
    return sources[torch.argsort(targets, stable=True)].reshape(-1, num_neighbours)


def test_neighbour_table_holds_the_nearest_nodes() -> None:
    graph = _random_graph(200)
    num_neighbours = 7

    table = build_neighbour_table(graph, "data", num_neighbours).cpu()

    assert table.shape == (200, num_neighbours) and table.dtype == torch.int32
    coords = graph["data"].x.double()
    for node in range(graph["data"].num_nodes):
        distances = haversine_distance(coords, coords[node].expand_as(coords))
        nearest = set(torch.topk(distances, num_neighbours, largest=False).indices.tolist())
        assert set(table[node].tolist()) == nearest
        assert node in nearest


@pytest.mark.parametrize("block_size", [1, 7, 64, 200])
def test_neighbour_table_is_the_same_in_blocks(block_size: int) -> None:
    graph = _random_graph(200)

    table = build_neighbour_table(graph, "data", 5, block_size=block_size)

    torch.testing.assert_close(table, build_neighbour_table(graph, "data", 5), rtol=0, atol=0)


def test_neighbour_table_has_the_neighbours_of_the_knn_edges_in_their_order() -> None:
    graph = _random_graph(300)

    table = build_neighbour_table(graph, "data", 9, block_size=64).cpu()

    torch.testing.assert_close(table.long(), _knn_graph_table(graph, 9).long(), rtol=0, atol=0)
