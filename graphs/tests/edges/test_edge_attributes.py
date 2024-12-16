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

from anemoi.graphs.edges.attributes import EdgeDirection
from anemoi.graphs.edges.attributes import EdgeLength


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-std"])
@pytest.mark.parametrize("luse_rotated_features", [True, False])
def test_directional_features(graph_nodes_and_edges, norm, luse_rotated_features: bool):
    """Test EdgeDirection compute method."""
    edge_attr_builder = EdgeDirection(norm=norm, luse_rotated_features=luse_rotated_features)
    edge_attr = edge_attr_builder.compute(graph_nodes_and_edges, ("test_nodes", "to", "test_nodes"))
    assert isinstance(edge_attr, torch.Tensor)


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-std"])
def test_edge_lengths(graph_nodes_and_edges, norm):
    """Test EdgeLength compute method."""
    edge_attr_builder = EdgeLength(norm=norm)
    edge_attr = edge_attr_builder.compute(graph_nodes_and_edges, ("test_nodes", "to", "test_nodes"))
    assert isinstance(edge_attr, torch.Tensor)


@pytest.mark.parametrize("attribute_builder", [EdgeDirection(), EdgeLength()])
def test_fail_edge_features(attribute_builder, graph_nodes_and_edges):
    """Test edge attribute builder fails with unknown nodes."""
    with pytest.raises(AssertionError):
        attribute_builder.compute(graph_nodes_and_edges, ("test_nodes", "to", "unknown_nodes"))
