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

from anemoi.graphs.nodes.attributes import CosineLatWeightedAttribute
from anemoi.graphs.nodes.attributes import IsolatitudeAreaWeights
from anemoi.graphs.nodes.attributes import MaskedPlanarAreaWeights
from anemoi.graphs.nodes.attributes import OrogGradientWeights
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


# ---------------------------------------------------------------------------
# OrogGradientWeights helpers & fixtures
# ---------------------------------------------------------------------------

class _OrogGradientWeightsWithMockGrad(OrogGradientWeights):
    """Test subclass that injects precomputed gradient values directly."""

    def __init__(self, grad_values: np.ndarray, **kwargs) -> None:
        super().__init__(orography_path="/mock/path", **kwargs)
        self._grad_values = grad_values

    def get_raw_values(self, nodes, **kwargs) -> torch.Tensor:
        g = self._grad_values
        positive = g[g > 0]
        p_clip = float(np.percentile(positive, self.percentile_clip)) if positive.size else 1.0
        g_norm = np.clip(g / max(p_clip, 1e-10), 0.0, 1.0)
        w = (1.0 + self.alpha * g_norm).astype(np.float32)
        return torch.from_numpy(w)


@pytest.fixture
def graph_with_flat_terrain() -> HeteroData:
    """100-node regular grid on a flat (z=0) domain."""
    n = 10
    lats = np.linspace(-0.1, 0.1, n)
    lons = np.linspace(0.0, 0.2, n)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    coords = np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)
    graph = HeteroData()
    graph["test_nodes"].x = torch.tensor(coords, dtype=torch.float32)
    graph["test_nodes"]["node_type"] = "AnemoiDatasetNodes"
    return graph


@pytest.fixture
def graph_with_ridge_terrain() -> HeteroData:
    """100-node grid where the right half sits at z=1000 (steep ridge in the middle)."""
    n = 10
    lats = np.linspace(-0.1, 0.1, n)
    lons = np.linspace(0.0, 0.2, n)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    coords = np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)
    graph = HeteroData()
    graph["test_nodes"].x = torch.tensor(coords, dtype=torch.float32)
    graph["test_nodes"]["node_type"] = "AnemoiDatasetNodes"
    return graph


def test_orog_gradient_weights_flat(graph_with_flat_terrain: HeteroData):
    """Zero gradient everywhere → all weights equal to 1.0."""
    N = graph_with_flat_terrain["test_nodes"].x.shape[0]
    g_flat = np.zeros(N, dtype=np.float32)
    attr = _OrogGradientWeightsWithMockGrad(grad_values=g_flat, alpha=2.0)
    w = attr.compute(graph_with_flat_terrain, "test_nodes")

    assert isinstance(w, torch.Tensor)
    assert w.shape[0] == N
    assert torch.all(torch.isclose(w, torch.ones_like(w))), "Zero gradient must yield weight=1.0"


def test_orog_gradient_weights_ridge(graph_with_ridge_terrain: HeteroData):
    """Step ridge → boundary nodes have higher weights than flat interior nodes."""
    x = graph_with_ridge_terrain["test_nodes"].x.numpy()
    N = x.shape[0]
    lon_median = float(np.median(x[:, 1]))
    # High gradient at the boundary, zero elsewhere
    g = np.where(np.abs(x[:, 1] - lon_median) <= 0.02, 1.0, 0.0).astype(np.float32)

    attr = _OrogGradientWeightsWithMockGrad(grad_values=g, alpha=2.0, percentile_clip=95.0)
    w = attr.compute(graph_with_ridge_terrain, "test_nodes").numpy().ravel()

    flat_left = w[x[:, 1] < lon_median - 0.02]
    flat_right = w[x[:, 1] > lon_median + 0.02]
    ridge_nodes = w[np.abs(x[:, 1] - lon_median) <= 0.02]

    assert ridge_nodes.mean() > flat_left.mean(), "Ridge nodes must be upweighted vs left flat"
    assert ridge_nodes.mean() > flat_right.mean(), "Ridge nodes must be upweighted vs right flat"


@pytest.mark.parametrize("alpha", [0.0, 1.0, 3.0])
def test_orog_gradient_weights_bounds(graph_with_ridge_terrain: HeteroData, alpha: float):
    """Weights must lie in [1.0, 1.0 + alpha] for any alpha."""
    N = graph_with_ridge_terrain["test_nodes"].x.shape[0]
    g = np.linspace(0, 1, N).astype(np.float32)
    attr = _OrogGradientWeightsWithMockGrad(grad_values=g, alpha=alpha)
    w = attr.compute(graph_with_ridge_terrain, "test_nodes").numpy().ravel()

    assert float(w.min()) >= 1.0 - 1e-5, f"min weight {w.min()} < 1.0"
    assert float(w.max()) <= 1.0 + alpha + 1e-5, f"max weight {w.max()} > 1.0 + alpha"


def test_orog_gradient_weights_invalid_alpha():
    with pytest.raises(AssertionError):
        OrogGradientWeights(orography_path="/mock/path", alpha=-1.0)


def test_orog_gradient_weights_invalid_percentile():
    with pytest.raises(AssertionError):
        OrogGradientWeights(orography_path="/mock/path", percentile_clip=110.0)
