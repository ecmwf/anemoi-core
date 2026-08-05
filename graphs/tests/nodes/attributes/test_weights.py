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


def test_planar_area_weights_shoelace_matches_convexhull():
    """`_voronoi_region_areas` reproduces the original per-node ConvexHull volumes.

    Locks the vectorized shoelace area to the behaviour it replaces, on a clustered
    rectangular patch resembling a regional cutout (the production stress case).
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(0)
    latlons = np.column_stack([rng.uniform(0.6, 0.9, 5000), rng.uniform(0.0, 0.4, 5000)])

    attr = PlanarAreaWeights()
    resolution = attr._compute_mean_nearest_distance(latlons)
    boundary_points = attr._get_boundary_ring(latlons, resolution)
    extended_points = np.vstack([latlons, boundary_points])
    v = Voronoi(extended_points, qhull_options="QJ Pp")

    # Reference: the original per-node implementation.
    reference = np.array([ConvexHull(v.vertices[v.regions[v.point_region[idx]]]).volume for idx in range(len(latlons))])

    areas = attr._voronoi_region_areas(v, len(latlons))

    assert areas.shape == reference.shape
    np.testing.assert_allclose(areas, reference, rtol=1e-9, atol=0.0)


def test_planar_area_weights_degenerate_fallback():
    """A region with an interior vertex falls back to the exact ConvexHull area.

    Heavy qhull joggling can emit interior Voronoi vertices, making a stored region
    polygon non-convex so plain shoelace under-counts its area (observed on real
    stretched-grid data: up to ~21% on a few cells). The detector must flag such a
    region and the ConvexHull fallback must recover the exact hull area. Small random
    fixtures rarely emit a genuine (non-collinear) interior vertex, so we inject one
    into a real Voronoi region to exercise that path deterministically.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, (200, 2))

    # Boundary ring so the first len(points) regions are all bounded (>= 3 vertices),
    # matching how compute_area_weights calls the method.
    resolution = PlanarAreaWeights()._compute_mean_nearest_distance(points)
    boundary_points = PlanarAreaWeights()._get_boundary_ring(points, resolution)
    extended_points = np.vstack([points, boundary_points])
    v = Voronoi(extended_points, qhull_options="QJ Pp")
    n = len(points)

    # Pick a bounded region (no -1) with >= 4 vertices to deform.
    target = next(
        i
        for i in range(n)
        if v.regions[v.point_region[i]]
        and -1 not in v.regions[v.point_region[i]]
        and len(v.regions[v.point_region[i]]) >= 4
    )
    region = list(v.regions[v.point_region[target]])
    hull_area = ConvexHull(v.vertices[region]).volume  # correct (convex) cell area

    # Inject the region centroid (always interior to the convex hull) into the stored
    # vertex order, creating a re-entrant polygon that shoelace under-counts but whose
    # convex hull is unchanged.
    centroid = v.vertices[region].mean(axis=0)
    interior_idx = len(v.vertices)
    v.vertices = np.vstack([v.vertices, centroid])
    region.insert(1, interior_idx)
    v.regions[v.point_region[target]] = region

    # Plain shoelace on the deformed polygon must be materially wrong, so a correct
    # result can only come from the ConvexHull fallback (proves the fallback is load-bearing).
    poly = v.vertices[region]
    x, y = poly[:, 0], poly[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    assert abs(shoelace - hull_area) / hull_area > 1e-3, "injected vertex did not create a re-entrant polygon"

    # The production method must detect the non-convex region and fall back to ConvexHull.
    areas = PlanarAreaWeights._voronoi_region_areas(v, n)
    np.testing.assert_allclose(areas[target], hull_area, rtol=1e-9, atol=0.0)
