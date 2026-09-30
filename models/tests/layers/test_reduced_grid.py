# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttentionWrapper
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.reduced_grid import ReducedGridCrossNeighbourhoodMask
from anemoi.models.layers.reduced_grid import ReducedGridNeighbourhoodMask
from anemoi.models.layers.reduced_grid import matching_position

KERNEL_SIZE = (5, 7)


def _family(grid: ReducedGrid) -> str:
    return "healpix" if grid.is_shifted else "octahedral"


def self_attention(grid: ReducedGrid, kernel_size, backend: str) -> NeighbourhoodAttentionWrapper:
    """Neighbourhood self attention on ``grid``, with the points in grid order."""
    neighbourhood = GridNeighbourhood(_family(grid), tuple(kernel_size), backend, grid, grid, is_self_attention=True)
    return NeighbourhoodAttentionWrapper(neighbourhood)


def cross_attention(
    query_grid: ReducedGrid, key_grid: ReducedGrid, kernel_size, backend: str
) -> NeighbourhoodAttentionWrapper:
    """Neighbourhood cross attention from ``query_grid`` to ``key_grid``, with the points in grid order."""
    neighbourhood = GridNeighbourhood(_family(key_grid), tuple(kernel_size), backend, query_grid, key_grid)
    return NeighbourhoodAttentionWrapper(neighbourhood)


def exact_matching(pos, n, shift, other_n, other_shift):
    """Nearest position in another row, from the longitudes as exact fractions; halfway rounds up."""
    return math.floor((pos + Fraction(shift, 2)) * Fraction(other_n, n) - Fraction(other_shift, 2) + Fraction(1, 2))


def window(grid: ReducedGrid, row: int, pos: int, kernel_size: tuple[int, int]) -> list[int]:
    """Point indices seen by one query, worked out directly from the definition."""
    kernel_h, kernel_w = kernel_size
    starts = grid.row_starts.tolist()
    first_row = min(max(row - kernel_h // 2, 0), grid.num_rows - kernel_h)
    n = grid.row_lengths[row]
    keys = set()
    for other in range(first_row, first_row + kernel_h):
        other_n = grid.row_lengths[other]
        centre = exact_matching(pos, n, grid.shifts[row], other_n, grid.shifts[other])
        keys |= {starts[other] + (centre + o) % other_n for o in range(-(kernel_w // 2), kernel_w // 2 + 1)}
    return sorted(keys)


def reference_attention(query, key, value, grid, kernel_size):
    out = torch.empty_like(query)
    rows, positions = grid.rows_and_positions
    for i in range(grid.num_points):
        keys = window(grid, int(rows[i]), int(positions[i]), kernel_size)
        scores = (query[..., i : i + 1, :] @ key[..., keys, :].transpose(-1, -2)) * query.shape[-1] ** -0.5
        out[..., i, :] = (scores.softmax(dim=-1) @ value[..., keys, :])[..., 0, :]
    return out


def grid_coords(grid: ReducedGrid) -> torch.Tensor:
    lats = torch.linspace(89.0, -89.0, grid.num_rows, dtype=torch.float64)
    coords = [(math.radians(lats[r]), 2 * math.pi * j / n) for r, n in enumerate(grid.row_lengths) for j in range(n)]
    return torch.tensor(coords)


@pytest.mark.parametrize(("n", "num_points"), [(16, 1600), (48, 10944), (96, 40320), (1280, 6599680)])
def test_octahedral_sizes(n, num_points):
    grid = ReducedGrid.octahedral(n)
    assert grid.num_rows == 2 * n
    assert grid.num_points == num_points
    assert grid.row_lengths[0] == grid.row_lengths[-1] == 20


def test_from_coords_recovers_the_rows():
    grid = ReducedGrid.octahedral(8)
    assert ReducedGrid.from_coords(grid_coords(grid)) == grid


def test_from_coords_rejects_other_layouts():
    coords = grid_coords(ReducedGrid.octahedral(8))
    with pytest.raises(ValueError):
        ReducedGrid.from_coords(coords.flip(0))
    shifted = coords.clone()
    shifted[:, 1] += 0.01
    with pytest.raises(ValueError):
        ReducedGrid.from_coords(shifted)


def test_mask_matches_the_definition():
    grid = ReducedGrid.octahedral(8)
    attention = self_attention(grid, KERNEL_SIZE, backend="sdpa")
    mask = attention._mask_on(torch.device("cpu"))
    rows, positions = grid.rows_and_positions

    assert (mask.sum(dim=1) == KERNEL_SIZE[0] * KERNEL_SIZE[1]).all()
    for i in range(grid.num_points):
        expected = torch.zeros(grid.num_points, dtype=torch.bool)
        expected[window(grid, int(rows[i]), int(positions[i]), KERNEL_SIZE)] = True
        assert torch.equal(mask[i], expected), f"point {i}"


def test_window_wraps_round_the_globe():
    grid = ReducedGrid.octahedral(8)
    mask = self_attention(grid, KERNEL_SIZE, backend="sdpa")._mask_on(torch.device("cpu"))
    start, n = int(grid.row_starts[4]), grid.row_lengths[4]
    # The first point of a row sees the last points of the same row.
    assert mask[start, start + n - 1] and mask[start, start + n - KERNEL_SIZE[1] // 2]


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
@pytest.mark.parametrize("kernel_size", [KERNEL_SIZE, (3, 45)])
def test_backends_match_reference(backend, kernel_size):
    grid = ReducedGrid.octahedral(6)
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, grid.num_points, 8, generator=generator) for _ in range(3))
    attention = self_attention(grid, kernel_size, backend=backend)
    torch.testing.assert_close(
        attention(q, k, v, q.shape[0]), reference_attention(q, k, v, grid, kernel_size), rtol=1e-4, atol=1e-5
    )


def test_window_wider_than_short_rows_takes_their_whole_ring():
    grid, kernel_size = ReducedGrid.octahedral(8), (5, 27)
    mask = self_attention(grid, kernel_size, backend="sdpa")._mask_on(torch.device("cpu"))
    # A point of the first row sees rows 0-4 of 20, 24, 28, 32 and 36 points: the two shortest whole.
    assert mask[0].sum() == 20 + 24 + 27 + 27 + 27
    rows, positions = grid.rows_and_positions
    for i in range(grid.num_points):
        expected = torch.zeros(grid.num_points, dtype=torch.bool)
        expected[window(grid, int(rows[i]), int(positions[i]), kernel_size)] = True
        assert torch.equal(mask[i], expected), f"point {i}"


def cross_window(query_grid, key_grid, row, pos, kernel_size):
    """Key indices seen by one query of another grid, worked out directly from the definition."""
    kernel_h, kernel_w = kernel_size
    lat = query_grid.row_latitudes[row]
    nearest = min(range(key_grid.num_rows), key=lambda r: abs(key_grid.row_latitudes[r] - lat))
    first_row = min(max(nearest - kernel_h // 2, 0), key_grid.num_rows - kernel_h)
    n = query_grid.row_lengths[row]
    starts = key_grid.row_starts.tolist()
    keys = set()
    for other in range(first_row, first_row + kernel_h):
        other_n = key_grid.row_lengths[other]
        centre = exact_matching(pos, n, query_grid.shifts[row], other_n, key_grid.shifts[other])
        keys |= {starts[other] + (centre + o) % other_n for o in range(-(kernel_w // 2), kernel_w // 2 + 1)}
    return sorted(keys)


def reference_cross_attention(query, key, value, query_grid, key_grid, kernel_size):
    out = torch.empty_like(query)
    rows, positions = query_grid.rows_and_positions
    for i in range(query_grid.num_points):
        keys = cross_window(query_grid, key_grid, int(rows[i]), int(positions[i]), kernel_size)
        scores = (query[..., i : i + 1, :] @ key[..., keys, :].transpose(-1, -2)) * query.shape[-1] ** -0.5
        out[..., i, :] = (scores.softmax(dim=-1) @ value[..., keys, :])[..., 0, :]
    return out


def test_octahedral_latitudes_are_symmetric_and_decreasing():
    lats = torch.tensor(ReducedGrid.octahedral(24).row_latitudes)
    assert (lats[1:] < lats[:-1]).all()
    torch.testing.assert_close(lats, -lats.flip(0))


@pytest.mark.parametrize(
    ("n_query", "n_key", "kernel_size"),
    [(4, 8, (5, 7)), (8, 4, (3, 5)), (6, 6, (5, 7)), (6, 12, (7, 27)), (8, 4, (3, 61))],
)
def test_cross_mask_matches_the_definition(n_query, n_key, kernel_size):

    query_grid, key_grid = ReducedGrid.octahedral(n_query), ReducedGrid.octahedral(n_key)
    mask = cross_attention(query_grid, key_grid, kernel_size, backend="sdpa")._mask_on(torch.device("cpu"))
    rows, positions = query_grid.rows_and_positions
    for i in range(query_grid.num_points):
        expected = torch.zeros(key_grid.num_points, dtype=torch.bool)
        expected[cross_window(query_grid, key_grid, int(rows[i]), int(positions[i]), kernel_size)] = True
        assert torch.equal(mask[i], expected), f"query {i}"


def test_cross_mask_on_the_same_grid_is_self_attention():

    grid = ReducedGrid.octahedral(8)
    cross = cross_attention(grid, grid, KERNEL_SIZE, backend="sdpa")._mask_on(torch.device("cpu"))
    own = self_attention(grid, KERNEL_SIZE, backend="sdpa")._mask_on(torch.device("cpu"))
    assert torch.equal(cross, own)


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
@pytest.mark.parametrize(("n_query", "n_key"), [(4, 8), (8, 4)])
def test_cross_backends_match_reference(backend, n_query, n_key):

    query_grid, key_grid = ReducedGrid.octahedral(n_query), ReducedGrid.octahedral(n_key)
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, 2, query_grid.num_points, 8, generator=generator)
    k, v = (torch.randn(1, 2, key_grid.num_points, 8, generator=generator) for _ in range(2))
    attention = cross_attention(query_grid, key_grid, (3, 5), backend=backend)
    expected = reference_cross_attention(q, k, v, query_grid, key_grid, (3, 5))
    torch.testing.assert_close(attention(q, k, v, q.shape[0]), expected, rtol=1e-4, atol=1e-5)


def healpix_ring_coords(nside):
    healpy = pytest.importorskip("healpy")
    theta, phi = healpy.pix2ang(nside, np.arange(12 * nside**2), nest=False)
    return torch.from_numpy(np.stack([np.pi / 2 - theta, phi], axis=-1))


@pytest.mark.parametrize("nside", [1, 2, 4, 8])
def test_healpix_grid_matches_healpy(nside):
    grid = ReducedGrid.healpix(nside)
    assert grid.num_points == 12 * nside**2
    assert grid.num_rows == 4 * nside - 1
    from_healpy = ReducedGrid.from_coords(healpix_ring_coords(nside))
    assert from_healpy == grid
    np.testing.assert_allclose(from_healpy.row_latitudes, grid.row_latitudes, atol=1e-9)


def test_unordered_coords_are_sorted_into_rings():
    healpy = pytest.importorskip("healpy")
    nside = 4
    theta, phi = healpy.pix2ang(nside, np.arange(12 * nside**2), nest=True)
    nested = torch.from_numpy(np.stack([np.pi / 2 - theta, phi], axis=-1))
    grid, order = ReducedGrid.from_unordered_coords(nested)
    assert grid == ReducedGrid.healpix(nside)
    torch.testing.assert_close(nested[order], healpix_ring_coords(nside))


@pytest.mark.parametrize("kernel_size", [(3, 5), (5, 13)])
def test_healpix_mask_matches_the_definition(kernel_size):
    grid = ReducedGrid.healpix(4)
    mask = self_attention(grid, kernel_size, backend="sdpa")._mask_on(torch.device("cpu"))
    rows, positions = grid.rows_and_positions
    for i in range(grid.num_points):
        expected = torch.zeros(grid.num_points, dtype=torch.bool)
        expected[window(grid, int(rows[i]), int(positions[i]), kernel_size)] = True
        assert torch.equal(mask[i], expected), f"point {i}"


def test_healpix_matching_is_the_nearest_longitude():
    """With the shifts, the matching position is the point of the other ring nearest in longitude."""
    grid = ReducedGrid.healpix(4)
    for row in range(grid.num_rows - 1):
        n, other_n = grid.row_lengths[row], grid.row_lengths[row + 1]
        s, other_s = grid.shifts[row], grid.shifts[row + 1]
        for pos in range(n):
            lon = (pos + s / 2) / n
            centre = int(matching_position(pos, n, other_n, s, other_s)) % other_n
            distances = [abs(((q + other_s / 2) / other_n - lon + 0.5) % 1 - 0.5) for q in range(other_n)]
            assert distances[centre] <= min(distances) + 1e-12


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_healpix_backends_match_reference(backend):
    grid = ReducedGrid.healpix(2)
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, grid.num_points, 8, generator=generator) for _ in range(3))
    attention = self_attention(grid, (3, 5), backend=backend)
    torch.testing.assert_close(
        attention(q, k, v, q.shape[0]), reference_attention(q, k, v, grid, (3, 5)), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    ("query_grid", "key_grid"),
    [(ReducedGrid.healpix(2), ReducedGrid.octahedral(4)), (ReducedGrid.octahedral(4), ReducedGrid.healpix(4))],
    ids=["healpix-to-octahedral", "octahedral-to-healpix"],
)
def test_cross_between_healpix_and_octahedral_matches_the_definition(query_grid, key_grid):

    kernel_size = (3, 5)
    mask = cross_attention(query_grid, key_grid, kernel_size, backend="sdpa")._mask_on(torch.device("cpu"))
    rows, positions = query_grid.rows_and_positions
    for i in range(query_grid.num_points):
        expected = torch.zeros(key_grid.num_points, dtype=torch.bool)
        expected[cross_window(query_grid, key_grid, int(rows[i]), int(positions[i]), kernel_size)] = True
        assert torch.equal(mask[i], expected), f"query {i}"


@pytest.mark.parametrize(
    ("query_grid", "key_grid", "kernel_size"),
    [
        (ReducedGrid.octahedral(8), ReducedGrid.octahedral(16), (5, 5)),
        (ReducedGrid.octahedral(16), ReducedGrid.octahedral(8), (3, 5)),
        (ReducedGrid.octahedral(12), ReducedGrid.octahedral(24), (7, 27)),
        (ReducedGrid.octahedral(16), ReducedGrid.octahedral(8), (3, 61)),
        (ReducedGrid.octahedral(8), ReducedGrid.octahedral(40), (5, 5)),
        (ReducedGrid.healpix(4), ReducedGrid.healpix(8), (5, 7)),
        (ReducedGrid.healpix(16), ReducedGrid.healpix(4), (3, 5)),
        (ReducedGrid.healpix(8), ReducedGrid.octahedral(8), (3, 5)),
        (ReducedGrid.octahedral(8), ReducedGrid.healpix(8), (5, 13)),
    ],
    ids=[
        "finer-keys",
        "coarser-keys",
        "wider-than-polar-rows",
        "wider-than-three-polar-rows",
        "keys-between-queries",
        "healpix",
        "healpix-coarser-keys",
        "healpix-to-octahedral",
        "octahedral-to-healpix",
    ],
)
def test_connection_counts_match_the_mask(query_grid, key_grid, kernel_size):
    rule = ReducedGridCrossNeighbourhoodMask(query_grid, key_grid, kernel_size)
    mask = rule(0, 0, torch.arange(query_grid.num_points)[:, None], torch.arange(key_grid.num_points)[None, :])
    assert torch.equal(rule.times_attended(), mask.sum(0))
    assert torch.equal(rule.keys_per_query(), mask.sum(1))


@pytest.mark.parametrize("grid", [ReducedGrid.octahedral(8), ReducedGrid.healpix(4)], ids=["octahedral", "healpix"])
def test_connection_counts_on_one_grid_match_the_mask(grid):
    rule = ReducedGridNeighbourhoodMask(grid, (5, 13))
    mask = rule(0, 0, torch.arange(grid.num_points)[:, None], torch.arange(grid.num_points)[None, :])
    assert torch.equal(rule.times_attended(), mask.sum(0))
    assert torch.equal(rule.keys_per_query(), mask.sum(1))


@pytest.mark.parametrize(
    ("n_key", "kernel_size", "every_key_read"),
    [(96, (5, 5), True), (320, (5, 5), False), (320, (9, 9), True)],
)
def test_keys_left_between_the_queries_of_a_coarser_grid(n_key, kernel_size, every_key_read):
    counts = ReducedGridCrossNeighbourhoodMask(
        ReducedGrid.octahedral(48), ReducedGrid.octahedral(n_key), kernel_size
    ).times_attended()
    assert bool((counts > 0).all()) == every_key_read


@pytest.mark.parametrize("grid", [ReducedGrid.octahedral(8), ReducedGrid.healpix(4)], ids=["octahedral", "healpix"])
def test_coords_are_the_positions_the_grid_is_recognised_from(grid):
    assert ReducedGrid.from_coords(grid.coords) == grid
    rows, _ = grid.rows_and_positions
    expected_lat = torch.tensor(grid.row_latitudes, dtype=torch.float64)[rows]
    torch.testing.assert_close(torch.rad2deg(grid.coords[:, 0]), expected_lat)
