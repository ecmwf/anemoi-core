# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes import StretchedTriNodes
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.nodes.builders.from_refined_icosahedron import IcosahedralNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import LimitedAreaIcosahedralNodes

# ---------------------------------------------------------------------------
# Helpers for this test file only
# ---------------------------------------------------------------------------

_SINGLE_RESOLUTION_MODE_KWARGS = dict(
    global_resolution=0,
    lam_resolution=1,
    name="test_nodes",
    reference_node_name="data",
    mask_attr_name=None,
    margin_radius_km=500.0,
)

_ONE_LAM_REGION = [
    {
        "lam_resolution": 1,
        "reference_node_name": "data",
        "mask_attr_name": None,
        "margin_radius_km": 500.0,
    }
]

_TWO_LAM_REGIONS = [
    {
        "lam_resolution": 2,
        "reference_node_name": "data",
        "mask_attr_name": None,
        "margin_radius_km": 500.0,
    },
    {
        "lam_resolution": 1,
        "reference_node_name": "data",
        "mask_attr_name": None,
        "margin_radius_km": 1000.0,
    },
]


def _small_graph_with_data_nodes() -> HeteroData:
    """Return a HeteroData with a small set of 'data' reference nodes."""
    # A few lat/lon points in radians covering a small region
    lats_deg = np.array([-45.0, -40.0, -35.0, -30.0, -25.0])
    lons_deg = np.array([145.0, 150.0, 155.0, 160.0, 165.0])
    coords = np.deg2rad(np.array([[lat, lon] for lat in lats_deg for lon in lons_deg]))
    graph = HeteroData()
    graph["data"].x = torch.tensor(coords, dtype=torch.float32)
    graph["data"]["_grid_reference_distance"] = 0.1
    return graph


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_single_resolution():
    """StretchedTriNodes initialises in single resolution mode."""
    node_builder = StretchedTriNodes(**_SINGLE_RESOLUTION_MODE_KWARGS)

    # Inheritance and type checks
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, StretchedTriNodes)
    assert isinstance(node_builder, LimitedAreaIcosahedralNodes)

    # Coverage of relevant attributes
    assert node_builder._lam_regions_raw is None
    assert node_builder._region_mask_builders is None
    assert hasattr(node_builder, "area_mask_builder")
    assert isinstance(node_builder.area_mask_builder, KNNAreaMaskBuilder)

    # Check area_mask_builder fields match input
    assert node_builder.area_mask_builder.reference_node_name == _SINGLE_RESOLUTION_MODE_KWARGS["reference_node_name"]
    assert node_builder.area_mask_builder.margin_radius_km == _SINGLE_RESOLUTION_MODE_KWARGS["margin_radius_km"]
    assert node_builder.area_mask_builder.mask_attr_name == _SINGLE_RESOLUTION_MODE_KWARGS["mask_attr_name"]

    # Check resolution attributes
    assert node_builder.global_resolution == _SINGLE_RESOLUTION_MODE_KWARGS["global_resolution"]
    assert hasattr(node_builder, "resolutions")
    assert isinstance(node_builder.resolutions, list)
    assert max(node_builder.resolutions) == _SINGLE_RESOLUTION_MODE_KWARGS["lam_resolution"]


def test_init_multi_resolution_one_region():
    """StretchedTriNodes initialises in multi resolution mode with one entry."""
    node_builder = StretchedTriNodes(global_resolution=0, name="test_nodes", lam_regions=_ONE_LAM_REGION)
    # Inheritance and type checks
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, StretchedTriNodes)
    assert isinstance(node_builder, IcosahedralNodes)

    # Coverage of relevant attributes
    assert node_builder.global_resolution == 0
    assert node_builder._lam_regions_raw is not None
    assert isinstance(node_builder._lam_regions_raw, list)
    assert len(node_builder._lam_regions_raw) == 1
    region = node_builder._lam_regions_raw[0]
    assert isinstance(region, dict)
    assert region["lam_resolution"] == 1
    assert region["reference_node_name"] == "data"
    assert region["mask_attr_name"] is None
    assert region["margin_radius_km"] == 500.0

    assert node_builder._region_mask_builders is not None
    assert isinstance(node_builder._region_mask_builders, list)
    assert len(node_builder._region_mask_builders) == 1
    builder = node_builder._region_mask_builders[0]
    assert isinstance(builder, KNNAreaMaskBuilder)
    assert builder.reference_node_name == "data"
    assert builder.margin_radius_km == 500.0
    assert builder.mask_attr_name is None

    # area_mask_builder should be None in multi-resolution mode
    assert hasattr(node_builder, "area_mask_builder")
    assert node_builder.area_mask_builder is None
    assert "area_mask_builder" not in node_builder.hidden_attributes

    # resolutions should be a list up to max lam_resolution
    assert hasattr(node_builder, "resolutions")
    assert isinstance(node_builder.resolutions, list)
    assert max(node_builder.resolutions) == 1


def test_init_multi_resolution_two_regions():
    """StretchedTriNodes initialises in multi resolution mode with two entries."""
    node_builder = StretchedTriNodes(global_resolution=0, name="test_nodes", lam_regions=_TWO_LAM_REGIONS)
    # Inheritance and type checks
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, StretchedTriNodes)
    assert isinstance(node_builder, IcosahedralNodes)

    # Coverage of relevant attributes
    assert node_builder.global_resolution == 0
    assert node_builder._lam_regions_raw is not None
    assert isinstance(node_builder._lam_regions_raw, list)
    assert len(node_builder._lam_regions_raw) == 2
    for region, expected in zip(node_builder._lam_regions_raw, _TWO_LAM_REGIONS):
        assert isinstance(region, dict)
        assert region["lam_resolution"] == expected["lam_resolution"]
        assert region["reference_node_name"] == expected["reference_node_name"]
        assert region["mask_attr_name"] == expected["mask_attr_name"]
        assert region["margin_radius_km"] == expected["margin_radius_km"]

    assert node_builder._region_mask_builders is not None
    assert isinstance(node_builder._region_mask_builders, list)
    assert len(node_builder._region_mask_builders) == 2
    for builder, expected in zip(node_builder._region_mask_builders, _TWO_LAM_REGIONS):
        assert isinstance(builder, KNNAreaMaskBuilder)
        assert builder.reference_node_name == expected["reference_node_name"]
        assert builder.margin_radius_km == expected["margin_radius_km"]
        assert builder.mask_attr_name == expected["mask_attr_name"]

    # area_mask_builder should be None in multi-resolution mode
    assert hasattr(node_builder, "area_mask_builder")
    assert node_builder.area_mask_builder is None
    assert "area_mask_builder" not in node_builder.hidden_attributes

    # resolutions should be a list up to max lam_resolution
    assert hasattr(node_builder, "resolutions")
    assert isinstance(node_builder.resolutions, list)
    assert max(node_builder.resolutions) == 2


# ---------------------------------------------------------------------------
# update_graph tests
# ---------------------------------------------------------------------------


def test_update_graph_single_resolution():
    """update_graph in single resolution mode stores expected hidden attributes."""
    node_builder = StretchedTriNodes(**_SINGLE_RESOLUTION_MODE_KWARGS)
    graph = _small_graph_with_data_nodes()

    graph = node_builder.update_graph(graph, {})
    assert "_resolutions" in graph["test_nodes"]
    assert "_nx_graph" in graph["test_nodes"]
    assert "_node_ordering" in graph["test_nodes"]
    assert "_area_mask_builder" in graph["test_nodes"]


def test_update_graph_multi_resolution():
    """update_graph in multi resolution mode stores expected hidden attributes."""
    node_builder = StretchedTriNodes(global_resolution=0, name="test_nodes", lam_regions=_TWO_LAM_REGIONS)
    graph = _small_graph_with_data_nodes()

    graph = node_builder.update_graph(graph, {})
    assert "_resolutions" in graph["test_nodes"]
    assert "_nx_graph" in graph["test_nodes"]
    assert "_node_ordering" in graph["test_nodes"]
    assert graph["test_nodes"].get("_area_mask_builder") is None
