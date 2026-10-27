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
from pydantic import TypeAdapter
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.rotated_grid import geographic_to_rotated
from anemoi.graphs.generate.rotated_grid import rotation_matrix
from anemoi.graphs.nodes import LatLonNodes
from anemoi.graphs.nodes import LimitedAreaRectilinearNodes
from anemoi.graphs.nodes import RectilinearNodes
from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.schemas.node_schemas import NodeBuilderSchemas


@pytest.mark.parametrize(("resolution", "nlat"), [(1.0, 180), (0.5, 360), (2.5, 72), (0.25, 720)])
def test_init(resolution: float, nlat: int):
    node_builder = RectilinearNodes(resolution, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert node_builder.nlat == nlat
    assert node_builder.nlon == 2 * nlat


@pytest.mark.parametrize("resolution", [0.7, 7.0, [1.0, 0.7]])
def test_fail_init(resolution: float):
    with pytest.raises(ValueError, match="whole number"):
        RectilinearNodes(resolution, "test_nodes")


@pytest.mark.parametrize("resolution", [0.0, -1.0, [1.0, 0.0]])
def test_fail_init_non_positive(resolution: float):
    with pytest.raises(ValueError, match="positive"):
        RectilinearNodes(resolution, "test_nodes")


def test_coordinates_are_cell_centres_row_by_row():
    """Rows run from north to south, longitudes east from zero, and no node lies on a pole."""
    coords = torch.rad2deg(RectilinearNodes(30.0, "test_nodes").get_coordinates())
    lat = coords[:, 0].view(6, 12)
    lon = coords[:, 1].view(6, 12)

    expected_lat = torch.tensor([75.0, 45.0, 15.0, -15.0, -45.0, -75.0], dtype=torch.float64)
    torch.testing.assert_close(lat, expected_lat[:, None].expand(6, 12))
    torch.testing.assert_close(lon, torch.arange(0, 360, 30, dtype=torch.float64).expand(6, 12))


def test_coordinates_with_poles():
    node_builder = RectilinearNodes(30.0, "test_nodes", include_poles=True)
    assert (node_builder.nlat, node_builder.nlon) == (7, 12)
    lat = torch.rad2deg(node_builder.get_coordinates()[:, 0]).view(7, 12)
    expected_lat = torch.arange(90.0, -91.0, -30.0, dtype=torch.float64)
    torch.testing.assert_close(lat, expected_lat[:, None].expand(7, 12))


def test_coordinates_anisotropic():
    node_builder = RectilinearNodes([30.0, 45.0], "test_nodes")
    coords = torch.rad2deg(node_builder.get_coordinates())
    assert (node_builder.nlat, node_builder.nlon) == (6, 8)
    torch.testing.assert_close(coords[:8, 1], torch.arange(0, 360, 45, dtype=torch.float64))


def test_register_nodes():
    graph = RectilinearNodes(2.0, "test_nodes").register_nodes(HeteroData())
    assert graph["test_nodes"].x.shape == (90 * 180, 2)
    assert graph["test_nodes"].x.dtype == torch.float32
    assert graph["test_nodes"].node_type == "RectilinearNodes"


def test_layout_is_row_major():
    graph = RectilinearNodes(30.0, "test_nodes", include_poles=True).update_graph(HeteroData())
    assert graph["test_nodes"]["_layout_shape"] == [7, 12]
    node = torch.arange(7 * 12)
    expected = torch.stack([node // 12, node % 12], dim=-1)
    torch.testing.assert_close(graph["test_nodes"]["_layout_index"].cpu(), expected)


def test_register_area_weights():
    node_builder = RectilinearNodes(2.0, "test_nodes")
    graph = node_builder.register_nodes(HeteroData())
    config = {
        "area_weight": {"_target_": SphericalAreaWeights.__module__ + ".SphericalAreaWeights", "norm": "unit-max"}
    }
    graph = node_builder.register_attributes(graph, config)
    weights = graph["test_nodes"]["area_weight"].view(90, 180)
    # Cells shrink towards the poles.
    assert weights[45, 0] > weights[10, 0] > weights[0, 0]


def _graph_with_reference(lat_range: tuple[float, float], lon_range: tuple[float, float], step: float = 1.0):
    lat, lon = torch.meshgrid(
        torch.arange(lat_range[0], lat_range[1] + step / 2, step, dtype=torch.float64),
        torch.arange(lon_range[0], lon_range[1] + step / 2, step, dtype=torch.float64),
        indexing="ij",
    )
    return LatLonNodes(lat.reshape(-1), lon.reshape(-1), "reference").register_nodes(HeteroData())


def _rotated_nodes(graph: HeteroData, node_builder: LimitedAreaRectilinearNodes) -> torch.Tensor:
    rotation = rotation_matrix(*node_builder.get_centre(), node_builder.rotation)
    return torch.rad2deg(geographic_to_rotated(graph["lam"].x.cpu().to(torch.float64), rotation))


def _check_rotated_grid(graph: HeteroData, node_builder: LimitedAreaRectilinearNodes, resolution: float) -> None:
    """The nodes form a regular grid at whole multiples of the resolution in the rotated frame."""
    rotated = _rotated_nodes(graph, node_builder).view(*graph["lam"]["_layout_shape"], 2)
    steps = rotated / resolution
    torch.testing.assert_close(steps, steps.round(), atol=1e-4, rtol=0)
    assert (rotated[:-1, :, 0] > rotated[1:, :, 0]).all(), "Rows must run from rotated north to south."
    assert (rotated[:, :-1, 1] < rotated[:, 1:, 1]).all(), "Columns must run east."


@pytest.mark.parametrize(
    ("lat_range", "lon_range"),
    [((40.0, 50.0), (0.0, 10.0)), ((75.0, 90.0), (-180.0, 179.0)), ((-10.0, 10.0), (170.0, 190.0))],
    ids=["mid-latitudes", "north-pole", "antimeridian"],
)
@pytest.mark.parametrize("rotation", [0.0, 30.0, -90.0])
def test_limited_area_covers_reference(lat_range: tuple[float, float], lon_range: tuple[float, float], rotation: float):
    graph = _graph_with_reference(lat_range, lon_range)
    node_builder = LimitedAreaRectilinearNodes(0.5, "reference", "lam", margin_radius_km=200.0, rotation=rotation)
    graph = node_builder.update_graph(graph)

    _check_rotated_grid(graph, node_builder, 0.5)
    rotation = rotation_matrix(*node_builder.get_centre(), node_builder.rotation)
    rotated_reference = torch.rad2deg(geographic_to_rotated(graph["reference"].x.cpu().to(torch.float64), rotation))
    rotated_nodes = _rotated_nodes(graph, node_builder)
    assert (rotated_reference.min(dim=0).values > rotated_nodes.min(dim=0).values).all()
    assert (rotated_reference.max(dim=0).values < rotated_nodes.max(dim=0).values).all()


def test_limited_area_centred_on_reference():
    graph = _graph_with_reference((80.0, 90.0), (-180.0, 170.0), step=10.0)
    node_builder = LimitedAreaRectilinearNodes(1.0, "reference", "lam")
    graph = node_builder.update_graph(graph)
    _check_rotated_grid(graph, node_builder, 1.0)
    # The centre of the reference nodes, here the North Pole, is a node.
    assert torch.rad2deg(graph["lam"].x[:, 0]).max() > 90.0 - 1e-4


def test_limited_area_rotation_turns_grid():
    """An area long from west to east gives a wide grid, and a tall one once turned by 90 degrees."""
    graph = _graph_with_reference((-5.0, 5.0), (0.0, 40.0))
    graph = LimitedAreaRectilinearNodes(1.0, "reference", "lam", rotation=0.0).update_graph(graph)
    num_rows, num_cols = graph["lam"]["_layout_shape"]
    graph = LimitedAreaRectilinearNodes(1.0, "reference", "turned", rotation=90.0).update_graph(graph)
    assert num_rows < num_cols
    turned_rows, turned_cols = graph["turned"]["_layout_shape"]
    # The margin is wider in rotated longitude away from the rotated equator, so only the rows match.
    assert turned_rows == num_cols
    assert turned_cols < turned_rows


def test_limited_area_crop_to_margin():
    graph = _graph_with_reference((40.0, 50.0), (0.0, 10.0))
    full = LimitedAreaRectilinearNodes(0.5, "reference", "full", margin_radius_km=300.0)
    cropped = LimitedAreaRectilinearNodes(0.5, "reference", "cropped", margin_radius_km=300.0, crop_to_margin=True)
    graph = cropped.update_graph(full.update_graph(graph))

    shape = graph["full"]["_layout_shape"]
    assert graph["cropped"]["_layout_shape"] == shape
    assert graph["cropped"].num_nodes < graph["full"].num_nodes

    # Put the coordinates of the cropped nodes back on the full grid.
    index = graph["cropped"]["_layout_index"].cpu()
    grid = torch.full((*shape, 2), float("nan"))
    grid[index[:, 0], index[:, 1]] = graph["cropped"].x.cpu()
    kept = ~grid[..., 0].isnan()
    assert kept.sum() == graph["cropped"].num_nodes
    assert not kept.all()
    torch.testing.assert_close(grid[kept], graph["full"].x.cpu().view(*shape, 2)[kept])


def test_limited_area_fails_for_global_reference():
    graph = RectilinearNodes(10.0, "reference").register_nodes(HeteroData())
    with pytest.raises(ValueError, match="RectilinearNodes"):
        LimitedAreaRectilinearNodes(1.0, "reference", "lam").update_graph(graph)


@pytest.mark.parametrize(
    "config",
    [
        {"_target_": "anemoi.graphs.nodes.RectilinearNodes", "resolution": 1.0},
        {"_target_": "anemoi.graphs.nodes.RectilinearNodes", "resolution": [0.25, 0.25], "include_poles": True},
        {
            "_target_": "anemoi.graphs.nodes.LimitedAreaRectilinearNodes",
            "resolution": 0.1,
            "reference_node_name": "data",
            "margin_radius_km": 200.0,
            "crop_to_margin": True,
            "rotation": 30.0,
        },
    ],
)
def test_schema(config: dict):
    TypeAdapter(NodeBuilderSchemas).validate_python(config)
