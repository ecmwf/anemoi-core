# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import CutOffEdges
from anemoi.graphs.edges import ReversedCutOffEdges


def test_init():
    """Test CutOffEdges initialization."""
    CutOffEdges("test_nodes1", "test_nodes2", 0.5)


@pytest.mark.parametrize("cutoff_factor", [-0.5, "hello", None])
def test_fail_init(cutoff_factor: str):
    """Test CutOffEdges initialization with invalid cutoff."""
    with pytest.raises(AssertionError):
        CutOffEdges("test_nodes1", "test_nodes2", cutoff_factor)


def test_cutoff(graph_with_nodes: HeteroData):
    """Test CutOffEdges."""
    builder = CutOffEdges("test_nodes", "test_nodes", 0.5)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types


def test_reversed_cutoff(graph_with_two_node_sets):
    """Test that ReversedKNNEdges actually registers an edge type in the graph."""

    graph = graph_with_two_node_sets
    reverse_graph = graph_with_two_node_sets.clone()

    builder = CutOffEdges("test_nodes1", "test_nodes2", 0.1)
    reverse_builder = ReversedCutOffEdges("test_nodes1", "test_nodes2", 0.1)

    graph = builder.update_graph(graph_with_two_node_sets)
    reverse_graph = reverse_builder.update_graph(reverse_graph)

    edge_index = graph[("test_nodes1", "to", "test_nodes2")].edge_index
    reverse_edge_index = reverse_graph[("test_nodes1", "to", "test_nodes2")].edge_index

    # The graph is arranged so that all four nodes in “test_nodes2” lie very close to one node in “test_nodes1.”
    # Consequently, running KNN from “test_nodes1”→“test_nodes2” will connect all four to that single “test_nodes1” node,
    # whereas reversing the direction links them all to a single “test_nodes2” node instead.
    print(edge_index)
    print(reverse_edge_index)
    assert len(edge_index[0].unique()) == 4
    assert len(reverse_edge_index[0].unique()) == 1
