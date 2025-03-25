# (C) Copyright 2024- Anemoi contributors.
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

from anemoi.graphs.nodes.attributes import BooleanAndMask
from anemoi.graphs.nodes.attributes import BooleanNot
from anemoi.graphs.nodes.attributes import BooleanOrMask


def test_boolean_not(graph_with_nodes: HeteroData):
    """Test attribute builder for BooleanNot."""
    node_attr_builder = BooleanNot("mask")
    mask = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_boolean_fail_multiple_masks(graph_with_nodes: HeteroData):
    """Test attribute builder for BooleanNot."""
    node_attr_builder = BooleanNot(["mask", "mask2"])
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


def test_boolean_and_mask(graph_with_nodes: HeteroData):
    """Test attribute builder for BooleanAndMask."""
    node_attr_builder = BooleanAndMask(["mask2"])
    mask = node_attr_builder.compute(graph_with_nodes, "test_nodes")
    assert torch.allclose(mask, graph_with_nodes["test_nodes"]["mask2"])

    node_attr_builder = BooleanAndMask(["mask", "mask2"])
    mask = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_boolean_or_mask(graph_with_nodes: HeteroData):
    """Test attribute builder for BooleanOrMask."""
    node_attr_builder = BooleanOrMask(["mask2"])
    mask = node_attr_builder.compute(graph_with_nodes, "test_nodes")
    print(mask.shape)
    print(graph_with_nodes["test_nodes"]["mask2"].shape)
    assert torch.allclose(mask, graph_with_nodes["test_nodes"]["mask2"])

    node_attr_builder = BooleanOrMask(["mask", "mask2"])
    mask = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_wrong_mask(graph_with_nodes: HeteroData):
    node_attr_builder = BooleanOrMask(["mask", "askdbash]"])
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")
