# (C) Copyright 2024-2026 Anemoi contributors.
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
    with pytest.raises(AssertionError, match="Either cutoff_factor or cutoff_distance_km must be provided"):
        edge_builder("test_nodes1", "test_nodes2")


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_fail_init_both_params(edge_builder):
    """Test CutOffEdges initialization with both cutoff_factor and cutoff_distance_km."""
    with pytest.raises(AssertionError, match="mutually exclusive"):
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


@pytest.mark.parametrize("edge_builder", [CutOffEdges, ReversedCutOffEdges])
def test_fail_init_invalid_backend(edge_builder):
    """Test CutOffEdges initialization with an unknown backend."""
    with pytest.raises(AssertionError, match="Backend"):
        edge_builder("test_nodes1", "test_nodes2", cutoff_factor=0.5, backend="pyg")


def test_cutoff_sklearn_backend_keeps_nearest(graph_with_nodes: HeteroData, monkeypatch):
    """backend='sklearn' bypasses pyg-lib and keeps the nearest max_num_neighbours within the cut-off."""
    import torch

    from anemoi.graphs.edges.builders.euclidean import base

    monkeypatch.setattr(base, "is_pyg_lib_available", lambda: True)
    monkeypatch.setattr(CutOffEdges, "_compute_edge_index_pyg", lambda *args, **kwargs: pytest.fail("pyg-lib used"))

    k = 3
    builder = CutOffEdges(
        "test_nodes", "test_nodes", cutoff_distance_km=20000.0, max_num_neighbours=k, backend="sklearn"
    )
    edge_index = builder.update_graph(graph_with_nodes)[("test_nodes", "to", "test_nodes")].edge_index

    coords, _ = builder.get_cartesian_node_coordinates(graph_with_nodes["test_nodes"], graph_with_nodes["test_nodes"])
    dist = torch.cdist(coords.double(), coords.double())  # (source, target); the radius covers the whole sphere
    for target in range(coords.shape[0]):
        kept = edge_index[0, edge_index[1] == target]
        assert len(kept) == k
        assert dist[kept, target].max() <= torch.sort(dist[:, target]).values[k - 1] + 1e-12
