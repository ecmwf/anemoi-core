# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.graphs.nodes import RegularLatLonNodes
from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder


@pytest.mark.parametrize(("resolution", "nlat"), [(1.0, 180), (0.5, 360), (2.5, 72), (0.25, 720)])
def test_init(resolution: float, nlat: int):
    node_builder = RegularLatLonNodes(resolution, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert node_builder.nlat == nlat
    assert node_builder.nlon == 2 * nlat


@pytest.mark.parametrize("resolution", [0.7, 7.0, 0.0, -1.0])
def test_fail_init(resolution: float):
    with pytest.raises(ValueError, match="whole number"):
        RegularLatLonNodes(resolution, "test_nodes")


def test_coordinates_are_cell_centres_row_by_row():
    """Rows run from north to south, longitudes east from zero, and no node lies on a pole."""
    coords = torch.rad2deg(RegularLatLonNodes(30.0, "test_nodes").get_coordinates())
    lat = coords[:, 0].view(6, 12)
    lon = coords[:, 1].view(6, 12)

    expected_lat = torch.tensor([75.0, 45.0, 15.0, -15.0, -45.0, -75.0], dtype=torch.float64)
    torch.testing.assert_close(lat, expected_lat[:, None].expand(6, 12))
    torch.testing.assert_close(lon, torch.arange(0, 360, 30, dtype=torch.float64).expand(6, 12))


def test_register_nodes():
    graph = RegularLatLonNodes(2.0, "test_nodes").register_nodes(HeteroData())
    assert graph["test_nodes"].x.shape == (90 * 180, 2)
    assert graph["test_nodes"].x.dtype == torch.float32
    assert graph["test_nodes"].node_type == "RegularLatLonNodes"


def test_register_area_weights():
    node_builder = RegularLatLonNodes(2.0, "test_nodes")
    graph = node_builder.register_nodes(HeteroData())
    config = {
        "area_weight": {"_target_": SphericalAreaWeights.__module__ + ".SphericalAreaWeights", "norm": "unit-max"}
    }
    graph = node_builder.register_attributes(graph, config)
    weights = graph["test_nodes"]["area_weight"].view(90, 180)
    # Cells shrink towards the poles.
    assert weights[45, 0] > weights[10, 0] > weights[0, 0]
