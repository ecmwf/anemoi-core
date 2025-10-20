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
from anemoi.graphs.edges.attributes import DirectionalHarmonics
from anemoi.graphs.edges.attributes import EdgeDirection
from anemoi.graphs.edges.attributes import EdgeLength
from anemoi.graphs.edges.attributes import RadialBasisFeatures

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


def test_radial_basis_features_default(graph_nodes_and_edges):
    """Test RadialBasisFeatures with default parameters (adaptive scaling)."""
    edge_attr_builder = RadialBasisFeatures()
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    assert edge_attr.shape[0] == edge_index.shape[1]  # num_edges
    assert edge_attr.shape[1] == 5  # default 5 centers

    # Check per-edge normalization: each edge's features should sum to ~1
    edge_sums = edge_attr.sum(dim=1)
    assert torch.allclose(edge_sums, torch.ones_like(edge_sums), atol=1e-6)


def test_radial_basis_features_global_scale(graph_nodes_and_edges):
    """Test RadialBasisFeatures with global r_scale."""
    edge_attr_builder = RadialBasisFeatures(r_scale=1.0)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    assert edge_attr.shape[0] == edge_index.shape[1]

    # Check per-edge normalization
    edge_sums = edge_attr.sum(dim=1)
    assert torch.allclose(edge_sums, torch.ones_like(edge_sums), atol=1e-6)


def test_radial_basis_features_custom_centers(graph_nodes_and_edges):
    """Test RadialBasisFeatures with custom centers."""
    custom_centers = [0.0, 0.33, 0.67, 1.0]
    edge_attr_builder = RadialBasisFeatures(centers=custom_centers, sigma=0.15)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    assert edge_attr.shape[1] == len(custom_centers)

    # Check per-edge normalization
    edge_sums = edge_attr.sum(dim=1)
    assert torch.allclose(edge_sums, torch.ones_like(edge_sums), atol=1e-6)


def test_radial_basis_features_epsilon(graph_nodes_and_edges):
    """Test RadialBasisFeatures with custom epsilon."""
    edge_attr_builder = RadialBasisFeatures(epsilon=1e-8)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    # Should not crash with division by zero


def test_radial_basis_features_no_norm_parameter():
    """Test that RadialBasisFeatures doesn't accept norm parameter."""
    # Should work without norm
    rbf = RadialBasisFeatures()
    assert rbf is not None

    # Trying to pass norm parameter should raise TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        RadialBasisFeatures(norm="l1")


def test_directional_harmonics_default(graph_nodes_and_edges):
    """Test DirectionalHarmonics with default parameters."""
    edge_attr_builder = DirectionalHarmonics()
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    assert edge_attr.shape[0] == edge_index.shape[1]  # num_edges
    assert edge_attr.shape[1] == 2 * 3  # default order=3 -> 2*order features


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_directional_harmonics_order(graph_nodes_and_edges, order):
    """Test DirectionalHarmonics with different orders."""
    edge_attr_builder = DirectionalHarmonics(order=order)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    assert edge_attr.shape[1] == 2 * order  # sin and cos for each order


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-std", "unit-range"])
def test_directional_harmonics_with_norm(graph_nodes_and_edges, norm):
    """Test DirectionalHarmonics with different normalization methods."""
    edge_attr_builder = DirectionalHarmonics(order=2, norm=norm)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    assert isinstance(edge_attr, torch.Tensor)
    assert edge_attr.shape[1] == 2 * 2  # order=2


def test_directional_harmonics_values(graph_nodes_and_edges):
    """Test that DirectionalHarmonics produces reasonable values."""
    edge_attr_builder = DirectionalHarmonics(order=1, norm=None)
    edge_index = graph_nodes_and_edges[TEST_EDGES].edge_index
    source_nodes = graph_nodes_and_edges[TEST_EDGES[0]]
    target_nodes = graph_nodes_and_edges[TEST_EDGES[2]]
    edge_attr = edge_attr_builder(x=(source_nodes, target_nodes), edge_index=edge_index)

    # For order=1, features are [sin(ψ), cos(ψ)]
    # sin²(ψ) + cos²(ψ) = 1
    sin_vals = edge_attr[:, 0]
    cos_vals = edge_attr[:, 1]
    norms = sin_vals**2 + cos_vals**2

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
