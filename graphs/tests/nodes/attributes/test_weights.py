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

from anemoi.graphs.nodes.attributes import CosineLatWeightedAttribute
from anemoi.graphs.nodes.attributes import IsolatitudeAreaWeights
from anemoi.graphs.nodes.attributes import MaskedPlanarAreaWeights
from anemoi.graphs.nodes.attributes import PlanarAreaWeights
from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute


def test_uniform_weights(graph_with_nodes: HeteroData):
    """Test attribute builder for UniformWeights."""
    node_attr_builder = UniformWeights()
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    # All values must be the same. Then, the mean has to be also the same
    assert torch.max(torch.abs(weights - torch.mean(weights))) == 0
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == node_attr_builder.dtype


def test_planar_area_weights(graph_with_nodes: HeteroData):
    """Test attribute builder for PlanarAreaWeights."""
    node_attr_builder = PlanarAreaWeights()
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == node_attr_builder.dtype


@pytest.mark.parametrize("fill_value", [0.0, -1.0, float("nan")])
def test_spherical_area_weights(graph_with_nodes: HeteroData, fill_value: float):
    """Test attribute builder for SphericalAreaWeights with different fill values."""
    node_attr_builder = SphericalAreaWeights(fill_value=fill_value)
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == node_attr_builder.dtype


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


@pytest.mark.parametrize("attr_class", [IsolatitudeAreaWeights, CosineLatWeightedAttribute])
@pytest.mark.parametrize("norm", [None, "l1", "unit-max"])
def test_latweighted(attr_class: type[BaseNodeAttribute], graph_with_rectilinear_nodes, norm: str):
    """Test attribute builder for Lat with different fill values."""
    node_attr_builder = attr_class(norm=norm)
    weights = node_attr_builder.compute(graph_with_rectilinear_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert torch.all(weights >= 0)
    assert weights.shape[0] == graph_with_rectilinear_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == node_attr_builder.dtype


def test_masked_planar_area_weights(graph_with_nodes: HeteroData):
    """Test attribute builder for PlanarAreaWeights."""
    node_attr_builder = MaskedPlanarAreaWeights(mask_node_attr_name="interior_mask")
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]
    assert weights.dtype == node_attr_builder.dtype

    mask = graph_with_nodes["test_nodes"]["interior_mask"]
    assert torch.all(weights[~mask] == 0)


def test_masked_planar_area_weights_fail(graph_with_nodes: HeteroData):
    """Test attribute builder for AreaWeights with invalid radius."""
    with pytest.raises(AssertionError):
        node_attr_builder = MaskedPlanarAreaWeights(mask_node_attr_name="nonexisting")
        node_attr_builder.compute(graph_with_nodes, "test_nodes")
