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


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_init(edge_builder):
    """Test CutOffEdges initialization."""
    edge_builder("test_nodes1", "test_nodes2", 0.5)


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
@pytest.mark.parametrize("cutoff_factor", [-0.5, "hello", None])
def test_fail_init(edge_builder, cutoff_factor: str):
    """Test CutOffEdges initialization with invalid cutoff."""
    with pytest.raises(AssertionError):
        edge_builder("test_nodes1", "test_nodes2", cutoff_factor)


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_cutoff(edge_builder, graph_with_nodes: HeteroData):
    """Test CutOffEdges."""
    builder = edge_builder("test_nodes", "test_nodes", 0.5)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types
