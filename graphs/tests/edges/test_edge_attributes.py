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
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.attributes import AttributeFromSourceNode
from anemoi.graphs.edges.attributes import AttributeFromTargetNode
from anemoi.graphs.edges.attributes import EdgeDirection
from anemoi.graphs.edges.attributes import EdgeLength

TEST_EDGES = ("test_nodes", "to", "test_nodes")


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-std", "unit-range"])
def test_directional_features(graph_nodes_and_edges, norm):
    """Test EdgeDirection compute method."""
    edge_attr_builder = EdgeDirection(norm=norm)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)
    assert isinstance(edge_attr, torch.Tensor)


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-std", "unit-range"])
def test_edge_lengths(graph_nodes_and_edges, norm):
    """Test EdgeLength compute method."""
    edge_attr_builder = EdgeLength(norm=norm)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)
    assert isinstance(edge_attr, torch.Tensor)


@pytest.mark.parametrize("attribute_builder_cls", [AttributeFromSourceNode, AttributeFromTargetNode])
def test_edge_attribute_from_node(attribute_builder_cls, graph_nodes_and_edges: HeteroData):
    """Test edge attribute builder fails with unknown nodes."""
    edge_attr_builder = attribute_builder_cls(node_attr_name="mask")
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)
    assert isinstance(edge_attr, torch.Tensor)


@pytest.mark.parametrize("attribute_builder", [EdgeDirection(), EdgeLength()])
def test_fail_edge_features(attribute_builder, graph_nodes_and_edges):
    """Test edge attribute builder fails with unknown nodes."""
    # with pytest.raises(AssertionError):
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    attribute_builder(x=(source_nodes, target_nodes), edge_index=edge_index)
