# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch

from anemoi.graphs.edges.builders.euclidean import lookup_knn as lk
from anemoi.graphs.edges.builders.euclidean.lookup_knn import LookupKNN
from anemoi.graphs.edges.builders.euclidean.lookup_knn import LookupKNNNotSupported
from anemoi.graphs.edges.builders.euclidean.lookup_knn import mean_spacing_deg

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _sphere(n: int, seed: int) -> torch.Tensor:
    v = torch.randn(n, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (v / v.norm(dim=1, keepdim=True)).float()


def _stress_queries() -> torch.Tensor:
    """Poles, 45 degrees, the date line and the equator."""
    s = math.sqrt(0.5)
    pts = [[0, 0, 1], [0, 0, -1], [s, 0, s], [-s, 0, -s], [0, s, s], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]]
    lat = torch.linspace(-90, 90, 721, dtype=torch.float64)
    ring = lk._latlon_deg_to_xyz(lat, torch.full_like(lat, 180.0))
    return torch.cat([torch.tensor(pts, dtype=torch.float64), ring]).float()


def _assert_exact(x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, k: int) -> None:
    """The returned neighbours have the same distances as the true k nearest (ties may pick other points)."""
    d = torch.cdist(y.double(), x.double())
    ref = torch.topk(d, k, dim=1, largest=False).values
    got = torch.sort(d.gather(1, idx.long().cpu()), dim=1).values
    assert (idx >= 0).all()
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 3, 4])
def test_lookup_knn_exact(device, k):
    x = _sphere(3000, seed=0).to(device)
    y = torch.cat([_sphere(5000, seed=1), _stress_queries()]).to(device)
    table = LookupKNN(x, k=4)
    idx, dist = table.query(y, k)
    assert idx.shape == (y.shape[0], k) and dist.shape == (y.shape[0], k)
    _assert_exact(x.cpu(), y.cpu(), idx.cpu(), k)


@pytest.mark.parametrize("device", DEVICES)
def test_lookup_knn_cell_size_does_not_change_result(device):
    x = _sphere(2000, seed=2).to(device)
    y = _sphere(3000, seed=3).to(device)
    for cell_deg in (1.0, 3.0, 8.0):
        idx, _ = LookupKNN(x, k=3, cell_deg=cell_deg).query(y, 3)
        _assert_exact(x.cpu(), y.cpu(), idx.cpu(), 3)


def test_lookup_knn_matches_pyg_format():
    """Same edge index layout as torch_geometric.nn.knn: rows (query index, searched index), k per query."""
    x, y = _sphere(2000, seed=4), _sphere(500, seed=5)
    edge_index = lk.lookup_knn(x, y, 3, name="x")
    assert edge_index.shape == (2, 1500)
    assert torch.equal(edge_index[0], torch.arange(500).repeat_interleave(3))
    _assert_exact(x, y, edge_index[1].view(-1, 3), 3)


def test_mean_spacing():
    # 41,253 square degrees on the sphere: one point per square degree is a mean spacing of 1 degree
    assert abs(mean_spacing_deg(41253) - 1.0) < 1e-3


def test_stretched_point_set():
    """Dense region plus coarse global points: exact, and not refused (the set covers the globe)."""
    lat, lon = torch.meshgrid(torch.linspace(40, 60, 120), torch.linspace(0, 30, 150), indexing="ij")
    dense = lk._latlon_deg_to_xyz(lat.flatten().double(), lon.flatten().double()).float()
    x = torch.cat([dense, _sphere(6000, seed=12)])
    y = torch.cat([_sphere(3000, seed=13), dense[::7] + 1e-3]).float()
    y = y / y.norm(dim=1, keepdim=True)
    idx, _ = LookupKNN(x, k=3).query(y, 3)
    _assert_exact(x, y, idx, 3)


def test_regional_point_set_is_refused():
    """A set covering a small region only: some cell centres are far from every point. No fallback: an error."""
    lat, lon = torch.meshgrid(torch.linspace(40, 50, 40), torch.linspace(0, 10, 40), indexing="ij")
    x = lk._latlon_deg_to_xyz(lat.flatten().double(), lon.flatten().double()).float()
    with pytest.raises(LookupKNNNotSupported):
        LookupKNN(x, k=3)
    lk.clear_lookup_cache()
    with pytest.raises(LookupKNNNotSupported, match="brute_force"):
        lk.lookup_knn(x, _sphere(10, seed=6), 3, name="regional")


def test_cache_by_name_and_k():
    """One table per (node-set name, k): a table built for a larger k would scan longer candidate lists."""
    lk.clear_lookup_cache()
    x, y = _sphere(2000, seed=14), _sphere(500, seed=15)
    lk.lookup_knn(x, y, 10, name="hidden")
    table = lk._CACHE[("hidden", 10)].table
    lk.lookup_knn(x, y, 10, name="hidden")
    assert lk._CACHE[("hidden", 10)].table is table and table.k == 10
    _assert_exact(x, y, lk.lookup_knn(x, y, 24, name="hidden")[1].view(-1, 24), 24)
    lk.lookup_knn(x, y, 10, name="hidden[mask]")
    assert set(lk._CACHE) == {("hidden", 10), ("hidden", 24), ("hidden[mask]", 10)}


def test_cache_name_collision_rebuilds(caplog):
    """A different node set under a cached name is detected and rebuilt (with a warning), never answered stale."""
    lk.clear_lookup_cache()
    x1, x2, y = _sphere(2000, seed=16), _sphere(2000, seed=17), _sphere(500, seed=18)
    lk.lookup_knn(x1, y, 3, name="hidden")
    with caplog.at_level("WARNING"):
        e = lk.lookup_knn(x2, y, 3, name="hidden")
    assert "differ from its cached table" in caplog.text
    _assert_exact(x2, y, e[1].view(-1, 3), 3)


def test_cache_changing_node_set_rebuilds():
    """A node set that changes between calls under one name, as in a graph rebuilt per batch, is never stale."""
    lk.clear_lookup_cache()
    y = _sphere(500, seed=21)
    for n, seed in ((2000, 22), (2500, 23), (2500, 24)):  # a new size, then a new set of the same size
        x = _sphere(n, seed=seed)
        _assert_exact(x, y, lk.lookup_knn(x, y, 3, name="data")[1].view(-1, 3), 3)


def test_cache_limit(monkeypatch):
    lk.clear_lookup_cache()
    monkeypatch.setattr(lk, "MAX_CACHED_NODE_SETS", 2)
    x, y = _sphere(2000, seed=19), _sphere(100, seed=20)
    lk.lookup_knn(x, y, 3, name="a")
    lk.lookup_knn(x, y, 3, name="b")
    with pytest.raises(RuntimeError, match="tables cached"):
        lk.lookup_knn(x, y, 3, name="c")
    with pytest.raises(RuntimeError, match="tables cached"):  # a further k of a cached name is a further table
        lk.lookup_knn(x, y, 4, name="a")
    lk.clear_lookup_cache()


def test_slow_build_warns(caplog):
    with caplog.at_level("WARNING"):
        LookupKNN(_sphere(500, seed=9), k=3, warn_build_seconds=0.0)
    assert "looks like about" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares the Triton cell formula with its torch twin")
def test_kernel_and_torch_cells_agree():
    """The cell formula is written twice (Triton, and torch for the CPU path): hold them to the same answer."""
    import triton
    import triton.language as tl

    from anemoi.graphs.triton import lookup_knn as tk

    @triton.jit
    def _cells(q_ptr, out_ptr, Q, nlat, nlon, lat0, dlat, dlon, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < Q
        qx = tl.load(q_ptr + offs * 3 + 0, mask=m, other=1.0)
        qy = tl.load(q_ptr + offs * 3 + 1, mask=m, other=0.0)
        qz = tl.load(q_ptr + offs * 3 + 2, mask=m, other=0.0)
        tl.store(out_ptr + offs, tk.cell_index(qx, qy, qz, nlat, nlon, lat0, dlat, dlon), mask=m)

    table = LookupKNN(_sphere(3000, seed=25).cuda(), k=3)
    q = torch.cat([_sphere(20000, seed=26), _stress_queries()]).cuda().contiguous()
    out = torch.empty(q.shape[0], dtype=torch.int32, device="cuda")
    _cells[(triton.cdiv(q.shape[0], 128),)](
        q, out, q.shape[0], table.nlat, table.nlon, table.lat0, table.dlat, table.dlon, BLOCK=128
    )
    assert torch.equal(out.long(), table._cells_torch(q))


@pytest.mark.parametrize("builder_cls", ["KNNEdges", "ReversedKNNEdges", "MutualKNNEdges"])
def test_invalid_backend(builder_cls):
    from anemoi.graphs import edges

    with pytest.raises(AssertionError):
        getattr(edges, builder_cls)("a", "b", 3, backend="kdtree")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("builder_cls", ["KNNEdges", "ReversedKNNEdges", "MutualKNNEdges"])
def test_knn_edge_builders_lookup_backend(builder_cls, device):
    """backend="lookup" gives the same edges as the default brute-force backend (up to ties)."""
    from anemoi.graphs import edges

    x, y = _sphere(2000, seed=7).to(device), _sphere(3000, seed=8).to(device)
    lookup = getattr(edges, builder_cls)("a", "b", 3, backend="lookup")
    brute = getattr(edges, builder_cls)("a", "b", 3)
    lk.clear_lookup_cache()
    e_lookup = lookup.compute_edge_index_from_coords(x, y, **lookup.prepare_method_kwargs(x, y)).cpu()
    assert lk._CACHE, "the lookup table was not built"
    e_brute = brute.compute_edge_index_from_coords(x, y, **brute.prepare_method_kwargs(x, y)).cpu()
    if builder_cls == "MutualKNNEdges":  # a tie broken differently can add or drop a mutual edge
        assert abs(e_lookup.shape[1] - e_brute.shape[1]) <= 0.001 * e_brute.shape[1]
        return
    assert e_lookup.shape == e_brute.shape
    xc, yc = x.cpu(), y.cpu()

    def dists(ei):  # (source, target) edges, grouped by the query side
        query = ei[1] if builder_cls == "KNNEdges" else ei[0]
        order = torch.argsort(query, stable=True)
        return torch.sort((xc[ei[0]] - yc[ei[1]]).norm(dim=1)[order].view(-1, 3), dim=1).values

    assert torch.allclose(dists(e_lookup), dists(e_brute), atol=1e-6, rtol=0)
