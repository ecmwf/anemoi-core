# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute


class ExtendedBaseNodeAttribute(BaseNodeAttribute):
    """Test implementation of BaseNodeAttribute."""

    def get_raw_values(self, nodes, **_kwargs) -> torch.Tensor:
        return torch.from_numpy(np.array(list(range(nodes.num_nodes))))


@pytest.mark.parametrize("nodes_name", ["invalid_nodes", 4])
def test_base_node_attribute_invalid_nodes_name(graph_with_nodes: HeteroData, nodes_name: str):
    """Test BaseNodeAttribute raises error with invalid nodes name."""
    with pytest.raises(AssertionError):
        ExtendedBaseNodeAttribute().compute(graph_with_nodes, nodes_name)


@pytest.mark.parametrize("norm", ["l3", "invalide"])
def test_base_node_attribute_invalid_norm(graph_with_nodes: HeteroData, norm: str):
    """Test BaseNodeAttribute raises error with invalid nodes name."""
    with pytest.raises(AssertionError):
        ExtendedBaseNodeAttribute(norm=norm).compute(graph_with_nodes, "test_nodes")


@pytest.mark.parametrize("norm", [None, "l1", "l2", "unit-max", "unit-std"])
def test_base_node_attribute_norm(graph_with_nodes: HeteroData, norm: str):
    """Test attribute builder for UniformWeights."""
    node_attr_builder = ExtendedBaseNodeAttribute(norm=norm)
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == node_attr_builder.dtype
