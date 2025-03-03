# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.graphs.nodes.attributes import BaseNodeAttribute
from anemoi.graphs.nodes.attributes import PlanarAreaWeights
from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.attributes import CutOutMask
from anemoi.graphs.nodes.attributes import BooleanNot
from anemoi.graphs.nodes.attributes import BooleanAndMask
from anemoi.graphs.nodes.attributes import BooleanOrMask


class TestBaseNodeAttribute(BaseNodeAttribute):
    """Test implementation of BaseNodeAttribute."""

    def get_raw_values(self, nodes, **_kwargs) -> torch.Tensor:
        return torch.from_numpy(np.array(list(range(nodes.num_nodes))))


@pytest.mark.parametrize("nodes_name", ["invalid_nodes", 4])
def test_base_node_attribute_invalid_nodes_name(graph_with_nodes: HeteroData, nodes_name: str):
    """Test BaseNodeAttribute raises error with invalid nodes name."""
    with pytest.raises(AssertionError):
        TestBaseNodeAttribute().compute(graph_with_nodes, nodes_name)


@pytest.mark.parametrize("norm", ["l3", "invalide"])
def test_base_node_attribute_invalid_norm(graph_with_nodes: HeteroData, norm: str):
    """Test BaseNodeAttribute raises error with invalid nodes name."""
    with pytest.raises(AssertionError):
        TestBaseNodeAttribute(norm=norm).compute(graph_with_nodes, "test_nodes")


@pytest.mark.parametrize("norm", [None, "l1", "l2", "unit-max", "unit-std"])
def test_base_node_attribute_norm(graph_with_nodes: HeteroData, norm: str):
    """Test attribute builder for UniformWeights."""
    node_attr_builder = TestBaseNodeAttribute(norm=norm)
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == getattr(torch, node_attr_builder.dtype)


def test_uniform_weights(graph_with_nodes: HeteroData):
    """Test attribute builder for UniformWeights."""
    node_attr_builder = UniformWeights()
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    # All values must be the same. Then, the mean has to be also the same
    assert torch.max(torch.abs(weights - torch.mean(weights))) == 0
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == getattr(torch, node_attr_builder.dtype)


def test_planar_area_weights(graph_with_nodes: HeteroData):
    """Test attribute builder for PlanarAreaWeights."""
    node_attr_builder = PlanarAreaWeights()
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == getattr(torch, node_attr_builder.dtype)


@pytest.mark.parametrize("fill_value", [0.0, -1.0, float("nan")])
@pytest.mark.parametrize("radius", [0.1, 1, np.pi])
def test_spherical_area_weights(graph_with_nodes: HeteroData, fill_value: float, radius: float):
    """Test attribute builder for SphericalAreaWeights with different fill values."""
    node_attr_builder = SphericalAreaWeights(fill_value=fill_value, radius=radius)
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == getattr(torch, node_attr_builder.dtype)


@pytest.mark.parametrize("radius", [-1.0, "hello", None])
def test_spherical_area_weights_wrong_radius(radius: float):
    """Test attribute builder for SphericalAreaWeights with invalid radius."""
    with pytest.raises(AssertionError):
        SphericalAreaWeights(radius=radius)


@pytest.mark.parametrize("fill_value", ["invalid", "as"])
def test_spherical_area_weights_wrong_fill_value(fill_value: str):
    """Test attribute builder for SphericalAreaWeights with invalid fill_value."""
    with pytest.raises(AssertionError):
        SphericalAreaWeights(fill_value=fill_value)


def test_cutout_mask(mocker, graph_with_nodes: HeteroData, mock_zarr_dataset_cutout):
    """Test attribute builder for CutOutMask."""
    # Add dataset attribute required by CutOutMask
    graph_with_nodes["test_nodes"]["_dataset"] = {"cutout": None}
    
    with mocker.patch("anemoi.graphs.nodes.attributes.open_dataset") as mock_open_dataset:
        mock_open_dataset.return_value = mock_zarr_dataset_cutout
        mask = CutOutMask().compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_cutout_mask_missing_dataset(graph_with_nodes: HeteroData):
    """Test CutOutMask fails when dataset attribute is missing."""    
    node_attr_builder = CutOutMask()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


def test_cutout_mask_missing_cutout(graph_with_nodes: HeteroData):
    """Test CutOutMask fails when cutout key is missing."""
    graph_with_nodes["test_nodes"]["_dataset"] = {}
    
    node_attr_builder = CutOutMask()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")
