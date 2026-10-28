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

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.adr import Advection
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.geocyclic import geocyclic_pad
from anemoi.models.layers.geometry import RegularLatLonGeometry
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.layout import latlon_cell_centres
from anemoi.models.layers.layout import latlon_grid_layout
from anemoi.models.layers.processor import ADRProcessor
from anemoi.models.layers.processor import FlowersProcessor
from anemoi.models.layers.semi_lagrangian import GridSampler
from anemoi.models.layers.utils import load_layer_kernels


def _unit_vectors(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)


def _smooth_field(points: torch.Tensor) -> torch.Tensor:
    """A smooth function on the sphere, written in terms of 3D position so it has no pole artefacts."""
    x, y, z = points.unbind(-1)
    return x + 0.5 * y * z + 0.3 * z**2 - 0.2 * x * y


def _geometry(nlat: int, nlon: int, order: torch.Tensor | None = None, **kwargs) -> RegularLatLonGeometry:
    return RegularLatLonGeometry(
        hidden_nodes=_hidden_nodes(nlat, nlon, order), layout_node_attribute_name="layout_index", **kwargs
    )


def _layout(nlat: int, nlon: int, batch_size: int = 1, order: torch.Tensor | None = None) -> LatLonLayout:
    shard_info = GraphShardInfo(nodes=[batch_size * nlat * nlon])
    return _geometry(nlat, nlon, order).bind(None, batch_size, shard_info)


def _on_grid(x: torch.Tensor, nlat: int, nlon: int) -> torch.Tensor:
    """Values given one per grid point, row by row, as the layers take them: (1, nlat, nlon, ...)."""
    return x.view(1, nlat, nlon, *x.shape[1:])


def test_latlon_cell_centres_order():
    lat, lon = latlon_cell_centres(4, 8)
    assert lat.shape == lon.shape == (32,)
    # Row by row from the north, longitudes running east from zero, no point on a pole.
    torch.testing.assert_close(lat[:8], torch.full((8,), torch.pi / 2 - torch.pi / 8, dtype=torch.float64))
    torch.testing.assert_close(lon[:8], torch.arange(8, dtype=torch.float64) * torch.pi / 4)
    assert lat[0] > lat[8] > lat[16] > lat[24]
    assert lat.abs().max() < torch.pi / 2


@pytest.mark.parametrize("pad", [1, 2])
def test_geocyclic_pad_continues_across_poles(pad):
    nlat, nlon = 6, 12
    lat, lon = latlon_cell_centres(nlat, nlon)
    field = _smooth_field(_unit_vectors(lat, lon)).view(nlat, nlon)

    # Rows beyond a pole have latitudes past +-90 degrees. Plugging those latitudes into the usual
    # formula for a 3D position gives the point on the other side of the pole, which is exactly
    # where the padded values must come from.
    rows = torch.arange(-pad, nlat + pad, dtype=torch.float64)
    cols = torch.arange(-pad, nlon + pad, dtype=torch.float64)
    lat_ext = torch.pi / 2 - (rows + 0.5) * torch.pi / nlat
    lon_ext = cols * 2 * torch.pi / nlon
    lat_ext, lon_ext = torch.meshgrid(lat_ext, lon_ext, indexing="ij")
    expected = _smooth_field(_unit_vectors(lat_ext, lon_ext))

    torch.testing.assert_close(geocyclic_pad(field, pad), expected)


def _hidden_nodes(nlat: int, nlon: int, order: torch.Tensor | None = None) -> dict:
    """Hidden nodes as a graph stores them, with nodes stored in the given grid order (row by row by default).

    Coordinates are float32 radians, latitude first, with the row and column of each node in
    "layout_index" and the grid size in "_layout_shape".
    """
    lat, lon = latlon_cell_centres(nlat, nlon)
    order = torch.arange(nlat * nlon) if order is None else order
    return {
        "x": torch.stack([lat, lon], dim=-1).float()[order],
        "layout_index": torch.stack([order // nlon, order % nlon], dim=-1),
        "_layout_shape": [nlat, nlon],
    }


def _shuffled(nlat: int, nlon: int) -> torch.Tensor:
    return torch.randperm(nlat * nlon, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize(("nlat", "nlon"), [(180, 360), (90, 180), (6, 12)])
def test_grid_layout_row_by_row(nlat, nlon):
    assert latlon_grid_layout(_hidden_nodes(nlat, nlon), "layout_index") == (nlat, nlon, None)


def test_grid_layout_accepts_longitudes_from_minus_180():
    nodes = _hidden_nodes(6, 12)
    nodes["x"][:, 1] = torch.remainder(nodes["x"][:, 1] + torch.pi, 2 * torch.pi) - torch.pi
    assert latlon_grid_layout(nodes, "layout_index") == (6, 12, None)


def test_grid_layout_of_shuffled_nodes():
    order = _shuffled(6, 12)
    nlat, nlon, grid_index = latlon_grid_layout(_hidden_nodes(6, 12, order), "layout_index")
    assert (nlat, nlon) == (6, 12)
    torch.testing.assert_close(grid_index, order)


def _without_attribute(nodes: dict) -> dict:
    del nodes["layout_index"]
    return nodes


def _reversed_rows(nodes: dict) -> dict:
    nodes["x"] = nodes["x"].view(6, 12, 2).flip(0).reshape(-1, 2)
    return nodes


def _with_poles(nodes: dict) -> dict:
    lat = torch.linspace(torch.pi / 2, -torch.pi / 2, 7)
    lon = torch.arange(12) * torch.pi / 6
    nodes["x"] = torch.stack(torch.meshgrid(lat, lon, indexing="ij"), dim=-1).reshape(-1, 2)
    nodes["layout_index"] = torch.stack(torch.meshgrid(torch.arange(7), torch.arange(12), indexing="ij"), -1).view(
        -1, 2
    )
    nodes["_layout_shape"] = [7, 12]
    return nodes


def _incomplete(nodes: dict) -> dict:
    return {"x": nodes["x"][:-1], "layout_index": nodes["layout_index"][:-1], "_layout_shape": [6, 12]}


def _repeated_grid_point(nodes: dict) -> dict:
    nodes["layout_index"][1] = nodes["layout_index"][0]
    return nodes


@pytest.mark.parametrize(
    "make_wrong_nodes",
    [
        pytest.param(_without_attribute, id="no-layout-attribute"),
        pytest.param(_reversed_rows, id="south-to-north"),
        pytest.param(_with_poles, id="points-on-the-poles"),
        pytest.param(_incomplete, id="incomplete"),
        pytest.param(_repeated_grid_point, id="repeated-grid-point"),
    ],
)
def test_grid_layout_rejects_other_grids(make_wrong_nodes):
    with pytest.raises(ValueError, match="RectilinearNodes"):
        latlon_grid_layout(make_wrong_nodes(_hidden_nodes(6, 12)), "layout_index")


def test_layout_round_trip_single_process():
    nlat, nlon, batch_size, channels = 4, 8, 2, 6
    layout = _layout(nlat, nlon, batch_size)
    x = torch.randn(batch_size * nlat * nlon, channels)
    grid = layout.to_state(x)
    assert grid.shape == (batch_size, nlat, nlon, channels)
    # Node n of batch b sits at row n // nlon and column n % nlon.
    torch.testing.assert_close(grid[1, 3, 5], x[nlat * nlon + 3 * nlon + 5])
    torch.testing.assert_close(layout.from_state(grid), x)

    groups = grid.view(batch_size, nlat, nlon, 3, 2)
    globe = layout.to_field(groups)
    assert globe.shape == (batch_size, 3, 2, nlat, nlon)
    torch.testing.assert_close(globe[1, 2, :, 3, 5], groups[1, 3, 5, 2])
    torch.testing.assert_close(layout.from_field(globe, 3), groups)


def test_layout_round_trip_with_grid_index():
    nlat, nlon, batch_size, channels = 4, 8, 2, 6
    order = _shuffled(nlat, nlon)
    layout = _layout(nlat, nlon, batch_size, order)
    x = torch.randn(batch_size * nlat * nlon, channels)
    grid = layout.to_state(x)
    # Node n of batch b sits at the grid point order[n].
    node = 7
    row, col = divmod(order[node].item(), nlon)
    torch.testing.assert_close(grid[1, row, col], x[nlat * nlon + node])
    torch.testing.assert_close(layout.from_state(grid), x)


@pytest.mark.parametrize("kind", ["adr", "flowers"])
def test_processors_do_not_depend_on_node_order(kind):
    """A processor on shuffled nodes gives the same values, node by node, as on nodes stored row by row."""
    nlat, nlon, channels = 8, 16, 16
    order = _shuffled(nlat, nlon)

    def build(order):
        torch.manual_seed(0)
        common = {
            "num_layers": 1,
            "num_channels": channels,
            "num_chunks": 1,
            "hidden_nodes": _hidden_nodes(nlat, nlon, order),
            "geometry": {
                "_target_": "anemoi.models.layers.geometry.RegularLatLonGeometry",
                "layout_node_attribute_name": "layout_index",
                "bias_rank": 4,
                "bias_base_maps": 2,
            },
            "layer_kernels": load_layer_kernels(instance=False),
        }
        if kind == "flowers":
            return FlowersProcessor(**common, num_heads=4)
        bias_kwargs = {"bias_rank": 4, "bias_base_maps": 2}
        return ADRProcessor(
            **common,
            timestep="6h",
            advection={
                "_target_": "anemoi.models.layers.adr.Advection",
                "advection_channels": 8,
                "num_velocity_fields": 4,
                "velocity_hidden_dim": 8,
                **bias_kwargs,
            },
            diffusion={"_target_": "anemoi.models.layers.adr.Diffusion", **bias_kwargs},
            reaction={
                "_target_": "anemoi.models.layers.adr.Reaction",
                "hidden_dim": 16,
                "num_layers": 2,
                **bias_kwargs,
            },
        )

    row_by_row, shuffled = build(None), build(order)
    # Give the advection a visible effect: start from random rather than near-zero weights.
    with torch.no_grad():
        for parameter in row_by_row.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
    shuffled.load_state_dict(row_by_row.state_dict())

    x = torch.randn(nlat * nlon, channels)
    shard_info = GraphShardInfo(nodes=[nlat * nlon])
    expected = row_by_row(x, batch_size=1, shard_info=shard_info)
    out = shuffled(x[order], batch_size=1, shard_info=shard_info)
    torch.testing.assert_close(out, expected[order], atol=1e-5, rtol=1e-5)


def test_layout_rejects_wrong_grid_size():
    with pytest.raises(ValueError, match="hidden graph has 32"):
        _geometry(4, 8).bind(None, 1, GraphShardInfo(nodes=[31]))


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_zero_displacement_is_identity(interpolation):
    nlat, nlon, heads = 10, 20, 3
    layer = GridSampler(num_heads=heads, interpolation=interpolation)
    values = torch.randn(1, nlat, nlon, heads, 4)
    moved = layer.sample(values, torch.zeros(1, nlat, nlon, heads, 2), _layout(nlat, nlon))
    torch.testing.assert_close(moved, values, atol=1e-4, rtol=0)


def _local_directions(lat: torch.Tensor, lon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    east = torch.stack([-lon.sin(), lon.cos(), torch.zeros_like(lon)], dim=-1)
    north = torch.stack([-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()], dim=-1)
    return east, north


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_sampling_reads_from_departure_points(interpolation):
    """Every value must be read from the departure point of its displacement.

    In coordinates rotated so that the arrival point sits on their equator, with the local east
    along that equator, the departure point lies at longitude -east and latitude -north. Written
    with the arrival point's own position, east and north directions, that is
    cos(n) cos(e) * position - cos(n) sin(e) * east_dir - sin(n) * north_dir.
    The displacements go up to about 20 degrees in random directions, so many paths from the
    polar rows cross a pole.
    """
    torch.manual_seed(0)
    nlat, nlon, heads = 90, 180, 4
    layer = GridSampler(num_heads=heads, interpolation=interpolation)

    lat, lon = latlon_cell_centres(nlat, nlon)
    position = _unit_vectors(lat, lon)[:, None, :]
    east_dir, north_dir = (d[:, None, :] for d in _local_directions(lat, lon))
    values = _smooth_field(position).expand(-1, heads).unsqueeze(-1).float().contiguous()

    displacement = (torch.rand(nlat * nlon, heads, 2, dtype=torch.float64) * 2 - 1) * 0.35
    e, n = displacement[..., :1], displacement[..., 1:]
    departure = n.cos() * e.cos() * position - n.cos() * e.sin() * east_dir - n.sin() * north_dir

    moved = layer.sample(
        _on_grid(values, nlat, nlon), _on_grid(displacement.float(), nlat, nlon), _layout(nlat, nlon)
    ).reshape(values.shape)
    # torch's bicubic kernel (cubic convolution with a = -0.75) is only first-order accurate, so on
    # a smooth field it is less accurate than bilinear.
    atol = 2e-3 if interpolation == "bilinear" else 5e-3
    torch.testing.assert_close(moved.squeeze(-1), _smooth_field(departure).float(), atol=atol, rtol=0)


def test_sampling_across_the_north_pole():
    """A point next to the north pole, moving south, reads from the far side of the pole."""
    nlat, nlon = 30, 60
    layer = GridSampler(num_heads=1)
    lat, lon = latlon_cell_centres(nlat, nlon)
    values = _smooth_field(_unit_vectors(lat, lon)).float().view(-1, 1, 1)

    # Moving south by 3 grid spacings means coming from 3 spacings further north: over the pole
    # and down to 2.5 spacings from the pole, half a turn round.
    displacement = torch.zeros(1, nlat, nlon, 1, 2)
    displacement[..., 1] = -3 * torch.pi / nlat
    moved = layer.sample(_on_grid(values, nlat, nlon), displacement, _layout(nlat, nlon)).view(nlat, nlon)

    expected_lat = torch.full((nlon,), torch.pi / 2 - 2.5 * torch.pi / nlat, dtype=torch.float64)
    expected = _smooth_field(_unit_vectors(expected_lat, lon[:nlon] + torch.pi)).float()
    torch.testing.assert_close(moved[0], expected, atol=2e-3, rtol=0)


def test_semi_lagrangian_advection_without_velocity_projects_down_and_up():
    nlat, nlon = 8, 16
    advection = Advection(
        num_channels=16,
        advection_channels=8,
        num_velocity_fields=8,
        velocity_hidden_dim=12,
        time_step=0.2,
        layer_kernels=load_layer_kernels(),
        geometry=_geometry(nlat, nlon, bias_rank=4, bias_base_maps=2),
    ).transport
    torch.nn.init.zeros_(advection.velocity_out.mix.weight)
    layout = _layout(nlat, nlon)
    x = torch.randn(1, nlat, nlon, 16)
    torch.testing.assert_close(advection(x, layout), advection.up(advection.down(x, layout)), atol=1e-4, rtol=0)


def test_adr_block_matches_paper_parameter_counts():
    """Per-layer parameter counts of Table 2 in the paper (0.25 degree model, 361 x 720 processor grid).

    Velocity network, advection and diffusion match exactly. The reaction MLP is smaller by the
    part that reads the 128 encoded static channels, which this block does not receive.
    """
    block = ADRProcessorBlock(
        num_channels=1024,
        geometry=_geometry(361, 720),
        time_step=0.2,
        layer_kernels=load_layer_kernels(),
        advection={"_target_": "anemoi.models.layers.adr.Advection"},
        diffusion={"_target_": "anemoi.models.layers.adr.Diffusion"},
        reaction={"_target_": "anemoi.models.layers.adr.Reaction"},
    )

    def count(*modules):
        return sum(p.numel() for m in modules for p in m.parameters())

    adv = block.advection.transport
    velocity = count(adv.velocity_norm, adv.velocity_in, adv.velocity_bias, adv.velocity_out)
    advection = count(adv.down, adv.up) + block.advection.share.numel()
    diffusion = count(block.diffusion)
    reaction = count(block.reaction)

    assert velocity == 9_112_576 // 8
    assert advection == 12_810_240 // 8
    assert diffusion == 9_798_656 // 8
    static_part = 2 * 128 + 128 * 896
    assert reaction == 29_663_232 // 8 - static_part


def test_low_rank_bias_adds_a_field_shared_over_the_batch():
    nlat, nlon, channels = 4, 8, 6
    bias = _geometry(nlat, nlon, bias_rank=3, bias_base_maps=2).position_bias(channels, load_layer_kernels())
    x = torch.zeros(2, nlat, nlon, channels)
    out = bias(x, _layout(nlat, nlon, batch_size=2))
    assert out.shape == x.shape
    torch.testing.assert_close(out[0], out[1])
    assert out.abs().sum() > 0


def test_rejects_odd_number_of_longitudes():
    with pytest.raises(ValueError, match="nlon must be even"):
        _geometry(4, 9)


def test_warp_without_displacement_returns_the_values():
    nlat, nlon = 8, 16
    warp = _geometry(nlat, nlon).warp(in_channels=12, out_channels=8, num_heads=2, layer_kernels=load_layer_kernels())
    torch.nn.init.zeros_(warp.flow[-1].weight)
    torch.nn.init.zeros_(warp.flow[-1].bias)
    layout = _layout(nlat, nlon)
    x = torch.randn(1, nlat, nlon, 12)
    torch.testing.assert_close(warp(x, layout), warp.value(x), atol=1e-4, rtol=0)


def test_warp_rejects_heads_that_do_not_divide_channels():
    with pytest.raises(ValueError, match="divisible"):
        _geometry(4, 8).warp(in_channels=8, out_channels=10, num_heads=4, layer_kernels=load_layer_kernels())


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_cartesian_displacement_follows_great_circles(interpolation):
    """A 3D displacement moves along the great circle in its direction by its length.

    Any part of the vector pointing away from the sphere's surface is ignored.
    """
    torch.manual_seed(0)
    nlat, nlon, heads = 90, 180, 4
    layer = GridSampler(num_heads=heads, interpolation=interpolation, cartesian_displacement=True)
    lat, lon = latlon_cell_centres(nlat, nlon)
    position = _unit_vectors(lat, lon)[:, None, :]
    values = _smooth_field(position).expand(-1, heads).unsqueeze(-1).float().contiguous()

    direction = torch.randn(nlat * nlon, heads, 3, dtype=torch.float64)
    direction = direction - (direction * position).sum(-1, keepdim=True) * position
    direction = direction / direction.norm(dim=-1, keepdim=True)
    angle = torch.rand(nlat * nlon, heads, 1, dtype=torch.float64) * 0.5
    departure = angle.cos() * position - angle.sin() * direction
    radial = torch.randn(nlat * nlon, heads, 1, dtype=torch.float64) * position

    displacement = _on_grid((angle * direction + radial).float(), nlat, nlon)
    moved = layer.sample(_on_grid(values, nlat, nlon), displacement, _layout(nlat, nlon)).reshape(values.shape)
    atol = 2e-3 if interpolation == "bilinear" else 5e-3
    torch.testing.assert_close(moved.squeeze(-1), _smooth_field(departure).float(), atol=atol, rtol=0)


@pytest.mark.parametrize("component", ["east", "north"])
def test_cartesian_and_local_displacements_agree_along_one_direction(component):
    """Moving purely east or purely north gives the same departure point in both forms."""
    nlat, nlon = 30, 60
    local = GridSampler(num_heads=1)
    cartesian = GridSampler(num_heads=1, cartesian_displacement=True)
    lat, lon = latlon_cell_centres(nlat, nlon)
    east_dir, north_dir = _local_directions(lat, lon)

    angle = 0.3
    local_displacement = torch.zeros(nlat * nlon, 1, 2)
    local_displacement[..., 0 if component == "east" else 1] = angle
    vector = angle * (east_dir if component == "east" else north_dir)

    layout = _layout(nlat, nlon)

    def grid_cells(layer, displacement):
        # Row and column of each departure point; columns n and n + nlon are the same place.
        grid = layer.departure_points(_on_grid(displacement, nlat, nlon), layout).reshape(-1, 2).double()
        col = (grid[:, 0] + 1) / 2 * (nlon + 2 * layer.pad - 1) - layer.pad
        row = (grid[:, 1] + 1) / 2 * (nlat + 2 * layer.pad - 1) - layer.pad
        return row, col

    row_c, col_c = grid_cells(cartesian, vector.float().view(-1, 1, 3))
    row_l, col_l = grid_cells(local, local_displacement)
    torch.testing.assert_close(row_c, row_l, atol=1e-4, rtol=0)
    column_gap = torch.remainder(col_c - col_l + nlon / 2, nlon) - nlon / 2
    torch.testing.assert_close(column_gap, torch.zeros_like(column_gap), atol=1e-4, rtol=0)


def test_cartesian_displacement_across_the_north_pole():
    nlat, nlon = 30, 60
    layer = GridSampler(num_heads=1, cartesian_displacement=True)
    lat, lon = latlon_cell_centres(nlat, nlon)
    values = _smooth_field(_unit_vectors(lat, lon)).float().view(-1, 1, 1)
    _, north_dir = _local_directions(lat, lon)

    # Moving south by 3 grid spacings: the value comes from over the pole, half a turn round.
    displacement = (-3 * torch.pi / nlat) * north_dir
    displacement = displacement.float().view(1, nlat, nlon, 1, 3)
    moved = layer.sample(_on_grid(values, nlat, nlon), displacement, _layout(nlat, nlon)).view(nlat, nlon)

    expected_lat = torch.full((nlon,), torch.pi / 2 - 2.5 * torch.pi / nlat, dtype=torch.float64)
    expected = _smooth_field(_unit_vectors(expected_lat, lon[:nlon] + torch.pi)).float()
    torch.testing.assert_close(moved[0], expected, atol=2e-3, rtol=0)


@pytest.mark.parametrize("cartesian_displacement", [False, True])
def test_layers_predict_one_displacement_per_head(cartesian_displacement):
    kernels = load_layer_kernels()
    geometry = _geometry(8, 16, bias_rank=4, bias_base_maps=2)
    advection = Advection(
        num_channels=16,
        advection_channels=8,
        num_velocity_fields=4,
        velocity_hidden_dim=12,
        time_step=0.2,
        layer_kernels=kernels,
        geometry=geometry,
        cartesian_displacement=cartesian_displacement,
    ).transport
    warp = geometry.warp(
        in_channels=16,
        out_channels=8,
        num_heads=4,
        layer_kernels=kernels,
        cartesian_displacement=cartesian_displacement,
    )
    dim = 3 if cartesian_displacement else 2
    assert advection.velocity_out.mix.out_features == 4 * dim
    assert warp.flow[-1].out_features == 4 * dim
