# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import SphericalDelaunayEdges

EDGES_NAME = ("test_nodes", "to", "test_nodes")


@pytest.fixture
def graph_with_scattered_nodes() -> HeteroData:
    """Graph with 100 nodes scattered over the globe, half of them masked in."""
    rng = np.random.default_rng(42)
    num_nodes = 100
    coords = np.stack([np.arcsin(rng.uniform(-1, 1, num_nodes)), rng.uniform(-np.pi, np.pi, num_nodes)], axis=1)
    graph = HeteroData()
    graph["test_nodes"].x = torch.tensor(coords, dtype=torch.float32)
    graph["test_nodes"].mask = torch.tensor([True] * 50 + [False] * 50).unsqueeze(-1)
    return graph


def test_init():
    """Test SphericalDelaunayEdges initialization."""
    SphericalDelaunayEdges("test_nodes", "test_nodes")


def test_fail_init_different_node_sets():
    """Test SphericalDelaunayEdges rejects source and target node sets that differ."""
    with pytest.raises(ValueError, match="single set"):
        SphericalDelaunayEdges("test_nodes1", "test_nodes2")


def test_delaunay(graph_with_scattered_nodes: HeteroData):
    """Test SphericalDelaunayEdges."""
    graph = SphericalDelaunayEdges("test_nodes", "test_nodes").update_graph(graph_with_scattered_nodes)

    assert EDGES_NAME in graph.edge_types
    edge_index = graph[EDGES_NAME].edge_index
    # The triangulation of a point cloud on the sphere leaves no node isolated.
    assert set(edge_index[0].tolist()) == set(range(graph["test_nodes"].num_nodes))
    # Both directions are registered, and no node is linked to itself.
    edges = {tuple(edge) for edge in edge_index.T.tolist()}
    assert all((dst, src) in edges for src, dst in edges)
    assert all(src != dst for src, dst in edges)


def test_delaunay_with_mask(graph_with_scattered_nodes: HeteroData):
    """Test SphericalDelaunayEdges only connects the nodes inside the mask."""
    graph = SphericalDelaunayEdges("test_nodes", "test_nodes", source_mask_attr_name="mask").update_graph(
        graph_with_scattered_nodes
    )

    mask = graph["test_nodes"].mask.squeeze()
    edge_index = graph[EDGES_NAME].edge_index.long()
    assert edge_index.shape[1] > 0
    assert mask[edge_index].all(), "Edges should not touch the nodes masked out."


def test_fail_too_few_nodes():
    """Test SphericalDelaunayEdges fails with a clear message on a node set it cannot triangulate."""
    with pytest.raises(ValueError, match="at least 4 nodes"):
        SphericalDelaunayEdges._get_delaunay_pairs(np.eye(3))


def test_fail_degenerate_nodes():
    """Test SphericalDelaunayEdges fails with a clear message on coplanar nodes."""
    equator = np.stack(
        [np.cos(np.linspace(0, 2 * np.pi, 10)), np.sin(np.linspace(0, 2 * np.pi, 10)), np.zeros(10)], axis=1
    )
    with pytest.raises(ValueError, match="degenerate"):
        SphericalDelaunayEdges._get_delaunay_pairs(equator)
