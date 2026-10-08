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

from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.semi_lagrangian import LatLonLayout
from anemoi.models.layers.semi_lagrangian import LowRankBias
from anemoi.models.layers.semi_lagrangian import SemiLagrangianAdvection
from anemoi.models.layers.semi_lagrangian import SemiLagrangianLayer
from anemoi.models.layers.semi_lagrangian import SemiLagrangianWarp
from anemoi.models.layers.semi_lagrangian import geocyclic_pad
from anemoi.models.layers.semi_lagrangian import latlon_cell_centres
from anemoi.models.layers.semi_lagrangian import latlon_grid_shape
from anemoi.models.layers.utils import load_layer_kernels


def _unit_vectors(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)


def _smooth_field(points: torch.Tensor) -> torch.Tensor:
    """A smooth function on the sphere, written in terms of 3D position so it has no pole artefacts."""
    x, y, z = points.unbind(-1)
    return x + 0.5 * y * z + 0.3 * z**2 - 0.2 * x * y


def _layout(nlat: int, nlon: int, batch_size: int = 1) -> LatLonLayout:
    return LatLonLayout(nlat, nlon, batch_size, [batch_size * nlat * nlon])


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


def _node_coordinates(nlat: int, nlon: int) -> torch.Tensor:
    """Coordinates as a graph stores them: float32 radians, latitude first."""
    lat, lon = latlon_cell_centres(nlat, nlon)
    return torch.stack([lat, lon], dim=-1).float()


@pytest.mark.parametrize(("nlat", "nlon"), [(180, 360), (90, 180), (6, 12)])
def test_grid_shape_from_node_coordinates(nlat, nlon):
    assert latlon_grid_shape(_node_coordinates(nlat, nlon)) == (nlat, nlon)


def test_grid_shape_accepts_longitudes_from_minus_180():
    coords = _node_coordinates(6, 12)
    coords[:, 1] = torch.remainder(coords[:, 1] + torch.pi, 2 * torch.pi) - torch.pi
    assert latlon_grid_shape(coords) == (6, 12)


def _reversed_rows(coords: torch.Tensor) -> torch.Tensor:
    return coords.view(6, 12, 2).flip(0).reshape(-1, 2)


def _with_poles(coords: torch.Tensor) -> torch.Tensor:
    lat = torch.linspace(torch.pi / 2, -torch.pi / 2, 7)
    lon = torch.arange(12) * torch.pi / 6
    return torch.stack(torch.meshgrid(lat, lon, indexing="ij"), dim=-1).reshape(-1, 2)


@pytest.mark.parametrize(
    "make_wrong_grid",
    [
        pytest.param(_reversed_rows, id="south-to-north"),
        pytest.param(
            lambda c: c[torch.randperm(c.shape[0], generator=torch.Generator().manual_seed(0))], id="shuffled"
        ),
        pytest.param(_with_poles, id="points-on-the-poles"),
        pytest.param(lambda c: c[:-1], id="incomplete"),
    ],
)
def test_grid_shape_rejects_other_grids(make_wrong_grid):
    with pytest.raises(ValueError, match="RegularLatLonNodes"):
        latlon_grid_shape(make_wrong_grid(_node_coordinates(6, 12)))


def test_layout_round_trip_single_process():
    nlat, nlon, batch_size, groups = 4, 8, 2, 3
    layout = _layout(nlat, nlon, batch_size)
    x = torch.randn(batch_size * nlat * nlon, groups, 5)
    globe = layout.to_globe(x)
    assert globe.shape == (batch_size, groups, 5, nlat, nlon)
    # Node n of batch b sits at row n // nlon and column n % nlon.
    torch.testing.assert_close(globe[1, 2, :, 3, 5], x[nlat * nlon + 3 * nlon + 5, 2])
    torch.testing.assert_close(layout.from_globe(globe, groups), x)


def test_layout_rejects_wrong_grid_size():
    with pytest.raises(ValueError, match="regular latitude-longitude"):
        LatLonLayout(4, 8, 1, [31])


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_zero_displacement_is_identity(interpolation):
    nlat, nlon, heads = 10, 20, 3
    layer = SemiLagrangianLayer(nlat=nlat, nlon=nlon, num_heads=heads, interpolation=interpolation)
    values = torch.randn(nlat * nlon, heads, 4)
    moved = layer.sample(values, torch.zeros(nlat * nlon, heads, 2), _layout(nlat, nlon))
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
    layer = SemiLagrangianLayer(nlat=nlat, nlon=nlon, num_heads=heads, interpolation=interpolation)

    lat, lon = latlon_cell_centres(nlat, nlon)
    position = _unit_vectors(lat, lon)[:, None, :]
    east_dir, north_dir = (d[:, None, :] for d in _local_directions(lat, lon))
    values = _smooth_field(position).expand(-1, heads).unsqueeze(-1).float().contiguous()

    displacement = (torch.rand(nlat * nlon, heads, 2, dtype=torch.float64) * 2 - 1) * 0.35
    e, n = displacement[..., :1], displacement[..., 1:]
    departure = n.cos() * e.cos() * position - n.cos() * e.sin() * east_dir - n.sin() * north_dir

    moved = layer.sample(values, displacement.float(), _layout(nlat, nlon))
    # torch's bicubic kernel (cubic convolution with a = -0.75) is only first-order accurate, so on
    # a smooth field it is less accurate than bilinear.
    atol = 2e-3 if interpolation == "bilinear" else 5e-3
    torch.testing.assert_close(moved.squeeze(-1), _smooth_field(departure).float(), atol=atol, rtol=0)


def test_sampling_across_the_north_pole():
    """A point next to the north pole, moving south, reads from the far side of the pole."""
    nlat, nlon = 30, 60
    layer = SemiLagrangianLayer(nlat=nlat, nlon=nlon, num_heads=1)
    lat, lon = latlon_cell_centres(nlat, nlon)
    values = _smooth_field(_unit_vectors(lat, lon)).float().view(-1, 1, 1)

    # Moving south by 3 grid spacings means coming from 3 spacings further north: over the pole
    # and down to 2.5 spacings from the pole, half a turn round.
    displacement = torch.zeros(nlat * nlon, 1, 2)
    displacement[..., 1] = -3 * torch.pi / nlat
    moved = layer.sample(values, displacement, _layout(nlat, nlon)).view(nlat, nlon)

    expected_lat = torch.full((nlon,), torch.pi / 2 - 2.5 * torch.pi / nlat, dtype=torch.float64)
    expected = _smooth_field(_unit_vectors(expected_lat, lon[:nlon] + torch.pi)).float()
    torch.testing.assert_close(moved[0], expected, atol=2e-3, rtol=0)


def test_semi_lagrangian_advection_without_velocity_projects_down_and_up():
    nlat, nlon = 8, 16
    advection = SemiLagrangianAdvection(
        num_channels=16,
        advection_channels=8,
        num_heads=8,
        velocity_hidden_dim=12,
        nlat=nlat,
        nlon=nlon,
        time_step=0.2,
        layer_kernels=load_layer_kernels(),
        bias_rank=4,
        bias_base_maps=2,
    )
    torch.nn.init.zeros_(advection.velocity_out.mix.weight)
    layout = _layout(nlat, nlon)
    x = torch.randn(nlat * nlon, 16)
    torch.testing.assert_close(advection(x, layout), advection.up(advection.down(x, layout)), atol=1e-4, rtol=0)


def test_adr_block_matches_paper_parameter_counts():
    """Per-layer parameter counts of Table 2 in the paper (0.25 degree model, 361 x 720 processor grid).

    Velocity network, advection and diffusion match exactly. The reaction MLP is smaller by the
    part that reads the 128 encoded static channels, which this block does not receive.
    """
    block = ADRProcessorBlock(
        num_channels=1024,
        advection_channels=768,
        num_heads=768,
        velocity_hidden_dim=384,
        reaction_hidden_dim=896,
        reaction_num_layers=4,
        nlat=361,
        nlon=720,
        time_step=0.2,
        layer_kernels=load_layer_kernels(),
    )

    def count(*modules):
        return sum(p.numel() for m in modules for p in m.parameters())

    adv = block.advection
    velocity = count(adv.velocity_norm, adv.velocity_in, adv.velocity_bias, adv.velocity_out)
    advection = count(adv.down, adv.up) + block.advection_share.numel()
    diffusion = count(block.layer_norm_diffusion, block.diffusion, block.diffusion_bias)
    reaction = count(block.layer_norm_reaction, block.reaction, block.reaction_bias)

    assert velocity == 9_112_576 // 8
    assert advection == 12_810_240 // 8
    assert diffusion == 9_798_656 // 8
    static_part = 2 * 128 + 128 * 896
    assert reaction == 29_663_232 // 8 - static_part


def test_low_rank_bias_adds_a_field_shared_over_the_batch():
    nlat, nlon, channels = 4, 8, 6
    bias = LowRankBias(channels, nlat, nlon, load_layer_kernels(), rank=3, num_base_maps=2)
    x = torch.zeros(2 * nlat * nlon, channels)
    out = bias(x, _layout(nlat, nlon, batch_size=2)).view(2, nlat * nlon, channels)
    torch.testing.assert_close(out[0], out[1])
    assert out.abs().sum() > 0


def test_rejects_odd_number_of_longitudes():
    with pytest.raises(ValueError, match="nlon must be even"):
        SemiLagrangianLayer(nlat=4, nlon=9, num_heads=1)


def test_warp_without_displacement_returns_the_values():
    nlat, nlon = 8, 16
    warp = SemiLagrangianWarp(
        in_channels=12, out_channels=8, num_heads=2, nlat=nlat, nlon=nlon, layer_kernels=load_layer_kernels()
    )
    torch.nn.init.zeros_(warp.flow[-1].weight)
    torch.nn.init.zeros_(warp.flow[-1].bias)
    layout = _layout(nlat, nlon)
    x = torch.randn(nlat * nlon, 12)
    torch.testing.assert_close(warp(x, layout), warp.value(x), atol=1e-4, rtol=0)


def test_warp_rejects_heads_that_do_not_divide_channels():
    with pytest.raises(ValueError, match="divisible"):
        SemiLagrangianWarp(
            in_channels=8, out_channels=10, num_heads=4, nlat=4, nlon=8, layer_kernels=load_layer_kernels()
        )


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_cartesian_displacement_follows_great_circles(interpolation):
    """A 3D displacement moves along the great circle in its direction by its length.

    Any part of the vector pointing away from the sphere's surface is ignored.
    """
    torch.manual_seed(0)
    nlat, nlon, heads = 90, 180, 4
    layer = SemiLagrangianLayer(
        nlat=nlat, nlon=nlon, num_heads=heads, interpolation=interpolation, cartesian_displacement=True
    )
    lat, lon = latlon_cell_centres(nlat, nlon)
    position = _unit_vectors(lat, lon)[:, None, :]
    values = _smooth_field(position).expand(-1, heads).unsqueeze(-1).float().contiguous()

    direction = torch.randn(nlat * nlon, heads, 3, dtype=torch.float64)
    direction = direction - (direction * position).sum(-1, keepdim=True) * position
    direction = direction / direction.norm(dim=-1, keepdim=True)
    angle = torch.rand(nlat * nlon, heads, 1, dtype=torch.float64) * 0.5
    departure = angle.cos() * position - angle.sin() * direction
    radial = torch.randn(nlat * nlon, heads, 1, dtype=torch.float64) * position

    moved = layer.sample(values, (angle * direction + radial).float(), _layout(nlat, nlon))
    atol = 2e-3 if interpolation == "bilinear" else 5e-3
    torch.testing.assert_close(moved.squeeze(-1), _smooth_field(departure).float(), atol=atol, rtol=0)


@pytest.mark.parametrize("component", ["east", "north"])
def test_cartesian_and_local_displacements_agree_along_one_direction(component):
    """Moving purely east or purely north gives the same departure point in both forms."""
    nlat, nlon = 30, 60
    local = SemiLagrangianLayer(nlat=nlat, nlon=nlon, num_heads=1)
    cartesian = SemiLagrangianLayer(nlat=nlat, nlon=nlon, num_heads=1, cartesian_displacement=True)
    lat, lon = latlon_cell_centres(nlat, nlon)
    east_dir, north_dir = _local_directions(lat, lon)

    angle = 0.3
    local_displacement = torch.zeros(nlat * nlon, 1, 2)
    local_displacement[..., 0 if component == "east" else 1] = angle
    vector = angle * (east_dir if component == "east" else north_dir)

    layout = _layout(nlat, nlon)

    def grid_cells(layer, displacement):
        # Row and column of each departure point; columns n and n + nlon are the same place.
        grid = layer.departure_points(displacement, layout).squeeze(1).double()
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
    layer = SemiLagrangianLayer(nlat=nlat, nlon=nlon, num_heads=1, cartesian_displacement=True)
    lat, lon = latlon_cell_centres(nlat, nlon)
    values = _smooth_field(_unit_vectors(lat, lon)).float().view(-1, 1, 1)
    _, north_dir = _local_directions(lat, lon)

    # Moving south by 3 grid spacings: the value comes from over the pole, half a turn round.
    displacement = (-3 * torch.pi / nlat) * north_dir
    moved = layer.sample(values, displacement.float().view(-1, 1, 3), _layout(nlat, nlon)).view(nlat, nlon)

    expected_lat = torch.full((nlon,), torch.pi / 2 - 2.5 * torch.pi / nlat, dtype=torch.float64)
    expected = _smooth_field(_unit_vectors(expected_lat, lon[:nlon] + torch.pi)).float()
    torch.testing.assert_close(moved[0], expected, atol=2e-3, rtol=0)


@pytest.mark.parametrize("cartesian_displacement", [False, True])
def test_layers_predict_one_displacement_per_head(cartesian_displacement):
    kernels = load_layer_kernels()
    advection = SemiLagrangianAdvection(
        num_channels=16,
        advection_channels=8,
        num_heads=4,
        velocity_hidden_dim=12,
        nlat=8,
        nlon=16,
        time_step=0.2,
        layer_kernels=kernels,
        bias_rank=4,
        bias_base_maps=2,
        cartesian_displacement=cartesian_displacement,
    )
    warp = SemiLagrangianWarp(
        in_channels=16,
        out_channels=8,
        num_heads=4,
        nlat=8,
        nlon=16,
        layer_kernels=kernels,
        cartesian_displacement=cartesian_displacement,
    )
    dim = 3 if cartesian_displacement else 2
    assert advection.velocity_out.mix.out_features == 4 * dim
    assert warp.flow[-1].out_features == 4 * dim
