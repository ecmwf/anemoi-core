# (C) Copyright 2026 Anemoi contributors.
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
from scipy.spatial import ConvexHull

from anemoi.graphs.generate.icosahedral_diamonds import IcosahedralDiamonds
from anemoi.graphs.generate.icosahedral_diamonds import label_icosahedron
from anemoi.models.layers.icosahedral_grid import IcosahedralNeighbourhood
from anemoi.models.layers.icosahedral_grid import hexagon_offsets
from anemoi.models.layers.icosahedral_neighbourhood import IcosahedralAttentionWrapper
from anemoi.models.layers.icosahedral_neighbourhood import IcosahedralGridNeighbourhood
from anemoi.models.layers.icosahedral_neighbourhood import IcosahedralShards
from anemoi.models.layers.neighbourhood_attention import build_grid_neighbourhood


def icosahedron_coords(resolution: int) -> torch.Tensor:
    """Latitude and longitude in radians of the refined icosahedron in diamond order, shape ``(10 * 4**r + 2, 2)``."""
    phi = (1 + 5**0.5) / 2
    vertices = np.array(
        [(s1, s2 * phi, 0) for s1 in (-1, 1) for s2 in (-1, 1)]
        + [(0, s1, s2 * phi) for s1 in (-1, 1) for s2 in (-1, 1)]
        + [(s2 * phi, 0, s1) for s1 in (-1, 1) for s2 in (-1, 1)],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    corners = vertices[label_icosahedron(vertices, ConvexHull(vertices).simplices)]
    xyz = IcosahedralDiamonds(resolution).positions(corners)
    return torch.as_tensor(np.stack([np.arcsin(xyz[:, 2]), np.arctan2(xyz[:, 1], xyz[:, 0])], axis=1))


def _sets(rows) -> list[set]:
    return [set(int(p) for p in row if p >= 0) for row in rows]


@pytest.mark.parametrize(("resolution", "radius"), [(2, 1), (3, 2), (3, 3), (4, 4)])
def test_hexagons_away_from_the_vertices_are_the_points_within_radius(resolution: int, radius: int) -> None:
    """For every point that is not an exception, the hexagon read from the tables holds exactly the points within ``radius`` hops."""
    tables = IcosahedralNeighbourhood(resolution, radius)
    points = np.setdiff1d(np.arange(tables.num_points), tables.exceptions)
    windows = tables.windows(points)
    true = tables.within_radius(points)
    for k in range(len(points)):
        window = windows[k][windows[k] >= 0]
        assert len(window) == len(set(window.tolist())), "a hexagon holds a point twice"
        assert set(window.tolist()) == set(true.indices[true.indptr[k] : true.indptr[k + 1]].tolist())
        assert len(window) == tables.window_size


@pytest.mark.parametrize("radius", [1, 2, 3, 4])
def test_exceptions_are_the_points_close_to_the_vertices(radius: int) -> None:
    """The exceptions are the points within ``radius - 1`` hops of the twelve icosahedron vertices."""
    tables = IcosahedralNeighbourhood(4, radius)
    vertices = np.flatnonzero(np.diff(tables.adjacency.indptr) == 5)
    assert len(vertices) == 12
    near = IcosahedralNeighbourhood(4, radius - 1).within_radius(vertices) if radius > 1 else None
    expected = vertices if near is None else np.unique(near.indices)
    np.testing.assert_array_equal(tables.exceptions, np.sort(expected))
    assert len(tables.exceptions) == 12 * (1 + 5 * radius * (radius - 1) // 2)


def test_hexagon_offsets() -> None:
    for radius in range(1, 6):
        offsets = hexagon_offsets(radius)
        assert len(offsets) == 3 * radius * (radius + 1) + 1
        assert np.abs(offsets).max() == radius and np.abs(offsets.sum(axis=1)).max() == radius


@pytest.mark.parametrize("own", [slice(0, 642), slice(0, 100), slice(300, 301), slice(500, 642)])
def test_shard_holds_exactly_the_keys_its_queries_reach(own: slice) -> None:
    tables = IcosahedralNeighbourhood(3, 2)
    shard = tables.shard(own)
    queries = np.arange(own.start, own.stop)
    reached = np.unique(tables.within_radius(queries).indices)
    np.testing.assert_array_equal(np.union1d(queries, shard.halo), np.union1d(queries, reached))
    assert not np.isin(shard.halo, queries).any()
    assert shard.num_keys == len(queries) + len(shard.halo)

    # The tables point to the shard's own key positions; queries only to own points that are not exceptions.
    key_points = np.concatenate([queries, shard.halo])
    in_table = shard.keys[shard.keys >= 0]
    assert np.isin(key_points[in_table], tables.table[shard.keys >= 0]).all()
    np.testing.assert_array_equal(key_points[shard.keys[shard.keys >= 0]], tables.table[shard.keys >= 0])
    own_queries = queries[shard.queries[shard.queries >= 0]]
    assert not np.isin(own_queries, tables.exceptions).any()
    np.testing.assert_array_equal(own_queries, tables.table[shard.queries >= 0])
    np.testing.assert_array_equal(queries[shard.exception_queries], np.intersect1d(queries, tables.exceptions))


def test_config_needs_diamond_order() -> None:
    coords = icosahedron_coords(3)
    neighbourhood = build_grid_neighbourhood("neighbourhood", {"grid": "icosahedral", "radius": 2}, coords)
    assert isinstance(neighbourhood, IcosahedralGridNeighbourhood)
    assert neighbourhood.tables.resolution == 3 and neighbourhood.backend == "triton"

    shuffled = coords[torch.randperm(coords.shape[0], generator=torch.Generator().manual_seed(0))]
    with pytest.raises(ValueError, match="diamond order"):
        build_grid_neighbourhood("neighbourhood", {"grid": "icosahedral", "radius": 2}, shuffled)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"grid": "icosahedral"}, "needs a 'radius'"),
        ({"grid": "icosahedral", "radius": 2, "kernel_size": [3, 3]}, "Unknown neighbourhood settings"),
        ({"grid": "icosahedral", "radius": 2, "backend": "flex"}, "backend must be one of"),
    ],
)
def test_config_errors(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_grid_neighbourhood("neighbourhood", config, icosahedron_coords(2))


def test_cross_attention_is_refused() -> None:
    coords = icosahedron_coords(2)
    with pytest.raises(ValueError, match="only does self attention"):
        build_grid_neighbourhood("neighbourhood", {"grid": "icosahedral", "radius": 1}, coords, coords)


def test_attention_rejects_other_sizes() -> None:
    neighbourhood = build_grid_neighbourhood(
        "neighbourhood", {"grid": "icosahedral", "radius": 1, "backend": "sdpa"}, icosahedron_coords(2)
    )
    q = torch.randn(1, 2, neighbourhood.tables.num_points, 8)
    with pytest.raises(ValueError, match=f"set up for {q.shape[-2]} queries"):
        neighbourhood.attention_module()(q[:, :, :-1], q, q, 1)


def test_split_attention_matches_whole() -> None:
    """Each GPU's band attends from its own points to its own points and halo as attention over the whole grid does."""
    neighbourhood = build_grid_neighbourhood(
        "neighbourhood", {"grid": "icosahedral", "radius": 2, "backend": "sdpa"}, icosahedron_coords(3)
    )
    num_points = neighbourhood.tables.num_points
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, num_points, 8, generator=generator, dtype=torch.float64) for _ in range(3))
    whole = IcosahedralAttentionWrapper(neighbourhood, neighbourhood.whole_shard)(q, k, v, 1)

    sizes = [100, 3, 300, num_points - 403]
    shards = IcosahedralShards(neighbourhood)
    for rank in range(len(sizes)):
        band = shards.plan(sizes, rank, torch.device("cpu")).band
        keys = band.key_points
        out = band.attention(q[:, :, band.query_points], k[:, :, keys], v[:, :, keys], 1)
        torch.testing.assert_close(out, whole[:, :, band.query_points])


def test_halo_exchange_plan_is_consistent() -> None:
    """What each GPU sends to another is what that GPU expects to receive from it, in the order it holds its halo."""
    neighbourhood = build_grid_neighbourhood(
        "neighbourhood", {"grid": "icosahedral", "radius": 3, "backend": "sdpa"}, icosahedron_coords(3)
    )
    sizes = [161, 160, 161, 160]
    starts = np.cumsum([0] + sizes)
    shards = IcosahedralShards(neighbourhood)
    plans = [shards.plan(sizes, rank, torch.device("cpu")) for rank in range(len(sizes))]
    for rank, plan in enumerate(plans):
        received = np.concatenate(
            [starts[source] + plans[source].halo_info.send_indices[rank].numpy() for source in range(len(sizes))]
        )
        np.testing.assert_array_equal(received, plan.band.key_points[sizes[rank] :].numpy())
        assert sum(plan.halo_info.recv_counts) == len(received)
