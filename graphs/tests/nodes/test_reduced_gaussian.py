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

from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.nodes.builders.from_reduced_gaussian import ReducedGaussianGridNodes


@pytest.mark.parametrize("grid", ["o16", "o48", "o96"])
def test_init(grid: str):
    """Test HEALPixNodes initialization."""
    node_builder = ReducedGaussianGridNodes(grid, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, ReducedGaussianGridNodes)


@pytest.mark.parametrize("grid", ["x123", "O"])
def test_fail_init1(grid: str):
    """Test ReducedGaussianGridNodes initialization with invalid grids."""
    with pytest.raises(AssertionError):
        ReducedGaussianGridNodes(grid, "test_nodes")


@pytest.mark.parametrize("grid", [4.3, -7])
def test_fail_init2(grid: str):
    """Test ReducedGaussianGridNodes initialization with invalid grids."""
    with pytest.raises(TypeError):
        ReducedGaussianGridNodes(grid, "test_nodes")


@pytest.mark.parametrize("grid", ["o16", "o48", "o96"])
def test_register_nodes(grid: str):
    """Test ReducedGaussianGridNodes register correctly the nodes."""
    node_builder = ReducedGaussianGridNodes(grid, "test_nodes")
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape[1] == 2
    assert graph["test_nodes"].node_type == "ReducedGaussianGridNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, SphericalAreaWeights])
@pytest.mark.parametrize("grid", ["o16", "o48", "o96"])
def test_register_attributes(graph_with_nodes: HeteroData, attr_class, grid: str):
    """Test ReducedGaussianGridNodes register correctly the weights."""
    node_builder = ReducedGaussianGridNodes(grid, "test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
