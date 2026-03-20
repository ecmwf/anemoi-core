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


@pytest.mark.parametrize("edge_builder", [KNNEdges, ReversedKNNEdges])
def test_init(edge_builder):
    """Test KNNEdges initialization."""
    edge_builder("test_nodes1", "test_nodes2", 3)


@pytest.mark.parametrize("edge_builder", [KNNEdges, ReversedKNNEdges])
@pytest.mark.parametrize("num_nearest_neighbours", [-1, 2.6, "hello", None])
def test_fail_init(edge_builder, num_nearest_neighbours: str):
    """Test KNNEdges initialization with invalid number of nearest neighbours."""
    with pytest.raises(AssertionError):
        edge_builder("test_nodes1", "test_nodes2", num_nearest_neighbours)


@pytest.mark.parametrize("edge_builder", [KNNEdges, ReversedKNNEdges])
def test_knn(edge_builder, graph_with_nodes):
    """Test KNNEdges."""
    builder = edge_builder("test_nodes", "test_nodes", 3)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types
