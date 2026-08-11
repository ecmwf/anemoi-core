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


def test_planar_area_weights_exact_on_lattices():
    """Cell areas are exact on rectangular lattices, perimeter included, at any aspect ratio."""
    import numpy as np

    for dx, dy in [(0.1, 0.1), (0.5, 0.1), (1.0, 0.1)]:
        x, y = np.meshgrid(np.arange(40) * dx, np.arange(30) * dy)
        latlons = np.column_stack([x.ravel(), y.ravel()])
        areas = PlanarAreaWeights().compute_area_weights(latlons)
        np.testing.assert_allclose(areas, dx * dy, rtol=1e-9)


def test_planar_area_weights_degenerate_inputs():
    """Collinear nodes fall back to uniform weights; duplicated nodes stay finite."""
    import numpy as np

    collinear = np.column_stack([np.arange(50) * 0.1, np.zeros(50)])
    np.testing.assert_array_equal(PlanarAreaWeights().compute_area_weights(collinear), 1.0)

    x, y = np.meshgrid(np.arange(20) * 0.1, np.arange(20) * 0.1)
    duplicated = np.vstack([np.column_stack([x.ravel(), y.ravel()]), [[0.5, 0.5]]])
    areas = PlanarAreaWeights().compute_area_weights(duplicated)
    assert np.isfinite(areas).all() and (areas > 0).all()


def test_masked_planar_area_weights_subset():
    """Masked weights come from the masked nodes alone, and a fractional mask scales them.

    A dense patch inside a coarse background: tessellating everything and masking
    afterwards would bound the patch's edge cells by the background, inflating them up to
    ~15x here. Tessellated on their own they are exactly dx * dy everywhere.
    """
    import numpy as np

    coarse = np.meshgrid(np.arange(0, 10, 0.5), np.arange(0, 10, 0.5))
    fine = np.meshgrid(np.arange(4, 6, 0.1), np.arange(4, 6, 0.1))
    background = np.column_stack([coarse[0].ravel(), coarse[1].ravel()])
    patch = np.column_stack([fine[0].ravel(), fine[1].ravel()])
    inside = ((background - 5.0) ** 2).sum(axis=1) > 2.0**2  # drop background nodes under the patch
    latlons = np.vstack([background[inside], patch])
    mask = np.zeros(len(latlons), dtype=bool)
    mask[len(background[inside]) :] = True

    graph = HeteroData()
    graph["test_nodes"].x = torch.tensor(latlons)
    graph["test_nodes"]["patch"] = torch.tensor(mask).unsqueeze(-1)
    graph["test_nodes"]["patch_half"] = 0.5 * torch.tensor(mask, dtype=torch.float64).unsqueeze(-1)

    weights = MaskedPlanarAreaWeights(mask_node_attr_name="patch").compute(graph, "test_nodes")
    assert torch.all(weights[~mask] == 0)
    np.testing.assert_allclose(weights[mask].numpy(), weights[mask].max().item(), rtol=1e-6)

    halved = MaskedPlanarAreaWeights(mask_node_attr_name="patch_half").compute(graph, "test_nodes")
    np.testing.assert_allclose(halved.numpy(), 0.5 * weights.numpy(), rtol=1e-6)


def test_voronoi_region_areas_matches_convexhull():
    """`_voronoi_region_areas` matches per-cell ConvexHull volumes, non-convex regions included.

    A stored region polygon that is not convex makes plain shoelace under-count, so one is
    injected to prove the ConvexHull fallback recovers the exact area.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(0)
    latlons = np.column_stack([rng.uniform(0.6, 0.9, 2000), rng.uniform(0.0, 0.4, 2000)])
    attr = PlanarAreaWeights()
    v = Voronoi(latlons, qhull_options="Qbb Qc Qz Pp")
    trusted_idx = np.flatnonzero(attr._trusted_cells(v, latlons))

    reference = np.array([ConvexHull(v.vertices[v.regions[v.point_region[idx]]]).volume for idx in trusted_idx])
    np.testing.assert_allclose(attr._voronoi_region_areas(v, trusted_idx), reference, rtol=1e-9, atol=0.0)

    # Inject the centroid into a region's stored vertex order: re-entrant polygon, same hull.
    target = next(int(i) for i in trusted_idx if len(v.regions[v.point_region[i]]) >= 4)
    region = list(v.regions[v.point_region[target]])
    hull_area = ConvexHull(v.vertices[region]).volume
    v.vertices = np.vstack([v.vertices, v.vertices[region].mean(axis=0)])
    region.insert(1, len(v.vertices) - 1)
    v.regions[v.point_region[target]] = region

    areas = attr._voronoi_region_areas(v, np.array([target]))
    np.testing.assert_allclose(areas[0], hull_area, rtol=1e-9, atol=0.0)
