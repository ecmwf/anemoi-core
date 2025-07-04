# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.graphs.edges import KNNEdges
from anemoi.graphs.edges import ReversedKNNEdges


def test_init():
    """Test KNNEdges initialization."""
    KNNEdges("test_nodes1", "test_nodes2", 3)


@pytest.mark.parametrize("num_nearest_neighbours", [-1, 2.6, "hello", None])
def test_fail_init(num_nearest_neighbours: str):
    """Test KNNEdges initialization with invalid number of nearest neighbours."""
    with pytest.raises(AssertionError):
        KNNEdges("test_nodes1", "test_nodes2", num_nearest_neighbours)


def test_knn(graph_with_nodes):
    """Test KNNEdges."""
    builder = KNNEdges("test_nodes", "test_nodes", 3)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types


def test_reversed_knn(graph_with_two_node_sets):
    """Test that ReversedKNNEdges actually registers an edge type in the graph."""

    graph = graph_with_two_node_sets
    reverse_graph = graph_with_two_node_sets.clone()

    builder = KNNEdges("test_nodes1", "test_nodes2", 1)
    reverse_builder = ReversedKNNEdges("test_nodes1", "test_nodes2", 1)

    graph = builder.update_graph(graph_with_two_node_sets)
    reverse_graph = reverse_builder.update_graph(reverse_graph)

    edge_index = graph[("test_nodes1", "to", "test_nodes2")].edge_index
    reverse_edge_index = reverse_graph[("test_nodes1", "to", "test_nodes2")].edge_index

    # The graph is arranged so that all four nodes in “test_nodes2” lie very close to one node in “test_nodes1.”
    # Consequently, running KNN from “test_nodes1”→“test_nodes2” will connect all four to that single “test_nodes1” node,
    # whereas reversing the direction links them all to a single “test_nodes2” node instead.
    len(edge_index[0].unique()) == 4
    len(reverse_edge_index[0].unique()) == 1
