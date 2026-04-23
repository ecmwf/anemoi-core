# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

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


@pytest.mark.parametrize("edge_builder", [KNNEdges, ReversedKNNEdges])
def test_knn_exclude_self_edges(edge_builder, graph_with_nodes):
    """Test KNNEdges self-edge exclusion."""
    builder = edge_builder("test_nodes", "test_nodes", 3, exclude_self_edges=True)
    graph = builder.update_graph(graph_with_nodes)
    query_axis = 0 if edge_builder is ReversedKNNEdges else 1

    edge_index = graph["test_nodes", "to", "test_nodes"].edge_index
    assert not torch.any(edge_index[0] == edge_index[1])

    query_counts = torch.bincount(
        edge_index[query_axis].to(torch.int64),
        minlength=graph["test_nodes"].num_nodes,
    )
    assert torch.all(query_counts <= 3)
    assert torch.all(query_counts > 0)


@pytest.mark.parametrize("edge_builder", [KNNEdges, ReversedKNNEdges])
def test_knn_exclude_self_edges_with_different_masks(edge_builder, graph_with_nodes):
    """Test KNNEdges self-edge exclusion when source and target masks differ."""
    builder = edge_builder(
        "test_nodes",
        "test_nodes",
        3,
        source_mask_attr_name="mask2",
        target_mask_attr_name="mask",
        exclude_self_edges=True,
    )
    graph = builder.update_graph(graph_with_nodes)
    query_axis = 0 if edge_builder is ReversedKNNEdges else 1
    query_mask = graph["test_nodes"]["mask2" if edge_builder is ReversedKNNEdges else "mask"].squeeze()

    edge_index = graph["test_nodes", "to", "test_nodes"].edge_index
    assert not torch.any(edge_index[0] == edge_index[1])

    query_counts = torch.bincount(
        edge_index[query_axis].to(torch.int64),
        minlength=graph["test_nodes"].num_nodes,
    )
    assert torch.all(query_counts[query_mask] <= 3)
    assert torch.all(query_counts[query_mask] > 0)
    assert torch.all(query_counts[~query_mask] == 0)
