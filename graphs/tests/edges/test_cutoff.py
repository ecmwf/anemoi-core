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
@pytest.mark.parametrize("cutoff_factor", [-0.5, "hello"])
def test_fail_init_invalid_cutoff_factor(edge_builder, cutoff_factor: str):
    """Test CutOffEdges initialization with invalid cutoff_factor."""
    with pytest.raises(AssertionError):
        edge_builder("test_nodes1", "test_nodes2", cutoff_factor=cutoff_factor)


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_fail_init_no_params(edge_builder):
    """Test CutOffEdges initialization with neither cutoff_factor nor cutoff_distance_km."""
    with pytest.raises(ValueError, match="Either cutoff_factor or cutoff_distance_km must be provided"):
        edge_builder("test_nodes1", "test_nodes2")


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_fail_init_both_params(edge_builder):
    """Test CutOffEdges initialization with both cutoff_factor and cutoff_distance_km."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        edge_builder("test_nodes1", "test_nodes2", cutoff_factor=0.5, cutoff_distance_km=500.0)


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_cutoff(edge_builder, graph_with_nodes: HeteroData):
    """Test CutOffEdges with cutoff_factor."""
    builder = edge_builder("test_nodes", "test_nodes", cutoff_factor=0.5)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_cutoff_with_distance_km(edge_builder, graph_with_nodes: HeteroData):
    """Test CutOffEdges with cutoff_distance_km."""
    builder = edge_builder("test_nodes", "test_nodes", cutoff_distance_km=500.0)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
@pytest.mark.parametrize("cutoff_distance_km", [-500.0, "hello"])
def test_fail_init_invalid_cutoff_distance_km(edge_builder, cutoff_distance_km):
    """Test CutOffEdges initialization with invalid cutoff_distance_km."""
    with pytest.raises(AssertionError):
        edge_builder("test_nodes1", "test_nodes2", cutoff_distance_km=cutoff_distance_km)
