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
from anemoi.graphs.edges.builders.euclidean.lookup_knn import estimate_spacing_deg

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _sphere(n: int, seed: int) -> torch.Tensor:
    v = torch.randn(n, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (v / v.norm(dim=1, keepdim=True)).float()


def _stress_queries() -> torch.Tensor:
    """Poles, the 45 degree switch between the two grids, the date line and the equator."""
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
    edge_index = lk.lookup_knn(x, y, 3)
    assert edge_index.shape == (2, 1500)
    assert torch.equal(edge_index[0], torch.arange(500).repeat_interleave(3))
    _assert_exact(x, y, edge_index[1].view(-1, 3), 3)


def test_estimate_spacing():
    # a regular latitude/longitude grid of 2 degrees: nearest neighbours are about 2 degrees apart or closer
    lat, lon = torch.meshgrid(torch.arange(-89.0, 90.0, 2.0), torch.arange(0.0, 360.0, 2.0), indexing="ij")
    x = lk._latlon_deg_to_xyz(lat.flatten().double(), lon.flatten().double()).float()
    assert 0.5 < estimate_spacing_deg(x) <= 2.0 + 1e-6


def test_regional_point_set_is_refused():
    """A set covering a small region only: some cell centres are far from every point."""
    lat, lon = torch.meshgrid(torch.linspace(40, 50, 40), torch.linspace(0, 10, 40), indexing="ij")
    x = lk._latlon_deg_to_xyz(lat.flatten().double(), lon.flatten().double()).float()
    with pytest.raises(LookupKNNNotSupported):
        LookupKNN(x, k=3)
    lk._CACHE.clear()
    assert lk.lookup_knn(x, _sphere(10, seed=6), 3) is None


def test_slow_build_warns(caplog):
    with caplog.at_level("WARNING"):
        LookupKNN(_sphere(500, seed=9), k=3, warn_build_seconds=0.0)
    assert "may take about" in caplog.text


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
    lk._CACHE.clear()
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
