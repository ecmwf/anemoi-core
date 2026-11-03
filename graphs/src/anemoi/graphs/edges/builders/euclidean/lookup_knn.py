# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Exact k-nearest-neighbour search in a fixed point set covering the unit sphere, by table lookup.

Brute-force KNN compares every query with every searched point. When the searched set is fixed (a mesh) and only the
queries vary, the geometry can be done once: a table on a latitude/longitude grid stores, for each cell, every point
that can be among the k nearest of any location in the cell. A query then reads its cell's candidates and keeps the k
nearest of a handful of distances.

Exactness. Let c be a cell centre, r the largest distance from c to a location in the cell and d_k(c) the distance
from c to its k-th nearest point. For a query q in the cell, |q - c| <= r, so q has at least k points within
d_k(c) + r, and any of its k nearest points x satisfies |x - c| <= |x - q| + |q - c| <= d_k(c) + 2r. The table stores
every point within d_k(c) + 2r of c, so it holds the true k nearest of every query in the cell. Distances are chord
(Euclidean) distances between unit vectors, which order points like great-circle distances; ties (equal distances) may
be broken differently from a brute-force search.

Layout. One equal-angle latitude/longitude grid over the sphere. Candidate lists are stored flat with per-cell
offsets and lengths, so the many small cells near the poles cost little. Points are given as unit vectors, so no
convention for the range of latitude or longitude is assumed; the grid's own latitude comes from asin(z) and its
longitude from atan2(y, x), and the longitude index wraps. On CUDA the query is the fused Triton kernel in
``anemoi.graphs.triton.lookup_knn``, which computes each query's cell, streams its candidates and keeps the k best
in registers.

Cell size. Only the speed and the table size depend on it, not the result. The default is half the mean point
spacing sqrt(4 pi / N), so no grid-specific formula is needed; on regular global grids this is within a few percent of
the nearest-neighbour distance, and on stretched grids it sizes the cells between the fine and the coarse spacing. The
table is built for point sets that cover the globe (possibly with varying density); for regional point sets the build
raises LookupKNNNotSupported (there is no automatic fallback: use the brute-force backend for those).

The KNN edge builders use this search with ``backend: lookup``. It pays off when there are many more queries than
table cells (a few per searched point) or when the same node set is searched repeatedly; a build expected to be slow
is announced with a warning. Tables are cached per searched node set (see ``lookup_knn``).
"""

import logging
import math
import time

import torch

from anemoi.graphs.triton import lookup_knn as triton_knn
from anemoi.graphs.utils import is_pyg_lib_available

LOGGER = logging.getLogger(__name__)

# Upper bound on the error of a float32 chord distance between unit vectors, relative to float64. pyg-lib computes
# squared differences directly, so the error is a few float32 ulps of the coordinates (~1e-7); 1e-6 leaves headroom.
FP32_SLACK = 1e-6


class LookupKNNNotSupported(ValueError):
    """The point set is not one the lookup table is built for (see the module docstring)."""


def _latlon_deg_to_xyz(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    lat, lon = torch.deg2rad(lat), torch.deg2rad(lon)
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=1)


def mean_spacing_deg(n: int) -> float:
    """Mean point spacing in degrees of n points covering the sphere: the side of a square of area 4 pi / n."""
    return math.degrees(math.sqrt(4 * math.pi / n))


class LookupKNN:
    """Exact k-nearest search in a fixed point set covering the unit sphere (see the module docstring).

    Parameters
    ----------
    points : torch.Tensor
        (N, 3) unit vectors of the searched set.
    k : int
        Largest number of neighbours that will be queried.
    cell_deg : float, optional
        Cell size in degrees. Default: half the mean point spacing sqrt(4 pi / N), at least 0.02 degrees.
    warn_build_seconds : float
        Log a warning once a build looks like taking longer than this, estimated from its measured first chunk of
        cells. The build is a search from every cell centre (about 6 cells per point), so its cost grows with the
        square of the number of points.
    max_dk_spacing : float
        Refuse to build if some cell centre's k-th nearest point is farther than this many mean point spacings times
        sqrt(k) (the set does not cover the globe, e.g. a regional grid; on a uniform grid the k-th nearest point is
        about sqrt(k / pi) spacings away).
    margin_deg : float
        Each cell's radius is computed for the cell grown by this much on every side, so a query that float32
        rounding puts in the neighbouring cell is still covered.
    """

    def __init__(
        self,
        points: torch.Tensor,
        k: int,
        cell_deg: float | None = None,
        warn_build_seconds: float = 30.0,
        max_dk_spacing: float = 8.0,
        margin_deg: float = 1e-3,
    ) -> None:
        self.points = points.float().contiguous()
        self.k = k
        n = points.shape[0]
        if n < k:
            raise LookupKNNNotSupported(f"{n} points, fewer than k={k}; use the brute_force backend")
        self.spacing_deg = mean_spacing_deg(n)
        if cell_deg is None:
            # Half the mean spacing. The cell size only trades build time against query time, never the result:
            # candidates per cell are the points within d_k + 2r of the centre, about (sqrt(k) + 1.25 c/s)^2, while
            # the number of cells grows as (s/c)^2. Measured on a GH200 at k=3 on a 167k-point mesh, s/4 queries
            # about 18% faster than s/2 but builds 3.6x slower and stores 3x more; s/1.5 saves little build time and
            # queries 10-30% slower. The factor does not depend on k or on the type of grid.
            cell_deg = max(self.spacing_deg / 2, 0.02)
        self.nlat = math.ceil(180.0 / cell_deg)
        self.nlon = math.ceil(360.0 / cell_deg)
        self.lat0 = -90.0
        self.dlat = 180.0 / self.nlat
        self.dlon = 360.0 / self.nlon
        # pyg-lib's brute-force kernel slows down quickly with k; above ~16 the dense float64 search is as fast
        self._use_pyg = is_pyg_lib_available() and k <= 16
        self._warn_build_seconds = warn_build_seconds
        self._cell_deg = cell_deg
        self.margin = margin_deg
        self.max_dk_deg = max_dk_spacing * math.sqrt(k) * self.spacing_deg

        counts, cand = self._build_grid(points.double())
        self.cand = cand.to(torch.int32)
        self.start = torch.zeros_like(counts)
        self.start[1:] = torch.cumsum(counts, 0)[:-1]
        self.len = counts.to(torch.int32)
        LOGGER.info(
            "LookupKNN: %d points (spacing %.3f deg), k=%d, %dx%d cells of %.3f deg, "
            "candidates per cell mean %.1f max %d, %.1f MB",
            n,
            self.spacing_deg,
            k,
            self.nlat,
            self.nlon,
            self.dlat,
            float(counts.float().mean()),
            int(counts.max()),
            (self.cand.numel() * 4 + self.start.numel() * 8 + self.len.numel() * 4) / 1e6,
        )

    def _build_grid(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidate counts per cell and the flat candidate list."""
        device = pts.device
        lat_c = self.lat0 + (torch.arange(self.nlat, device=device, dtype=torch.float64) + 0.5) * self.dlat
        hl, hn = 0.5 * self.dlat + self.margin, 0.5 * self.dlon + self.margin
        centre0 = _latlon_deg_to_xyz(lat_c, torch.zeros_like(lat_c))
        r_row = (
            torch.stack(
                [
                    (
                        _latlon_deg_to_xyz((lat_c + a * hl).clamp(-90, 90), torch.full_like(lat_c, b * hn)) - centre0
                    ).norm(dim=1)
                    for a in (-1, 1)
                    for b in (-1, 1)
                ],
                dim=1,
            )
            .max(dim=1)
            .values
        )  # largest distance from a cell centre to its (grown) cell, per latitude row
        max_dk_chord = 2 * math.sin(math.radians(min(self.max_dk_deg, 180.0)) / 2)

        ncell = self.nlat * self.nlon
        use_pyg = self._use_pyg
        if use_pyg:
            cell_chunk = 65536
            pts32 = pts.float().contiguous()
        else:
            # about 1 GB of float64 distances: this chunk bounds a (cells x all points) matrix, so unlike the
            # query chunk it has to shrink as the node set grows
            cell_chunk = max(1, min(8192, (1 << 27) // pts.shape[0]))
        counts, flat = [], []
        started = time.perf_counter()
        for s in range(0, ncell, cell_chunk):
            idx = torch.arange(s, min(s + cell_chunk, ncell), device=device)
            ilat, ilon = idx // self.nlon, idx % self.nlon
            centre = _latlon_deg_to_xyz(lat_c[ilat], -180.0 + (ilon.double() + 0.5) * self.dlon)
            if use_pyg:
                c, f = self._chunk_pyg(centre, pts, pts32, r_row[ilat], max_dk_chord)
            else:
                c, f = self._chunk_cdist(centre, pts, r_row[ilat], max_dk_chord)
            counts.append(c)
            flat.append(f)
            if s == 0:
                self._warn_if_slow(started, cell_chunk, ncell)
        return torch.cat(counts), torch.cat(flat)

    def _warn_if_slow(self, started: float, done: int, ncell: int) -> None:
        """Announce a slow build, from the measured first chunk rather than from any assumption about the device.

        The first chunk includes the one-off costs of the search (allocation, and the kernels' first launch), so
        this over- rather than under-estimates, which is the right way round for a warning.
        """
        expected_s = (time.perf_counter() - started) * ncell / min(done, ncell)
        if expected_s > self._warn_build_seconds:
            LOGGER.warning(
                "LookupKNN: building the table for %d points (%d cells of %.3f deg) on %s looks like about %.0f s, "
                "from its first %d cells (it is a search from every cell centre, done once per node set).",
                self.points.shape[0],
                ncell,
                self._cell_deg,
                self.points.device,
                expected_s,
                min(done, ncell),
            )

    def _check_dk(self, dk: torch.Tensor, max_dk_chord: float) -> None:
        if float(dk.max()) > max_dk_chord:
            raise LookupKNNNotSupported(
                f"a cell centre's {self.k}-th nearest point is more than {self.max_dk_deg:.2f} deg away: the point "
                "set does not cover the globe (e.g. a regional grid); use the brute_force backend for it"
            )

    def _chunk_cdist(
        self, centre: torch.Tensor, pts: torch.Tensor, r: torch.Tensor, max_dk_chord: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidates of a chunk of cells by a dense float64 distance matrix (no pyg-lib)."""
        dist = torch.cdist(centre, pts)
        cand_k = min(max(4 * self.k, 16), pts.shape[0])
        while True:
            dsort, isort = torch.topk(dist, cand_k, dim=1, largest=False)
            self._check_dk(dsort[:, self.k - 1], max_dk_chord)
            bound = dsort[:, self.k - 1] + 2 * r + 1e-9
            inside = dsort <= bound[:, None]
            if cand_k == pts.shape[0] or not bool(inside[:, -1].any()):
                break
            cand_k = min(2 * cand_k, pts.shape[0])  # some cell needs more candidates: widen and redo
        return inside.sum(dim=1), isort[inside]  # row-major, so each cell's candidates stay contiguous

    def _chunk_pyg(
        self, centre: torch.Tensor, pts: torch.Tensor, pts32: torch.Tensor, r: torch.Tensor, max_dk_chord: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidates of a chunk of cells with pyg-lib: knn for d_k, then a radius search, checked in float64.

        The k points pyg-lib returns are real points, so their largest float64 distance is an upper bound on d_k.
        The radius search uses the chunk's largest bound plus the float32 slack, so it returns every point within
        any cell's bound; the float64 distances then keep, per cell, exactly the points within its own bound.
        """
        from torch_geometric.nn import knn
        from torch_geometric.nn import radius

        n = centre.shape[0]
        c32 = centre.float().contiguous()
        ei = knn(pts32, c32, k=self.k)  # rows (centre, point), grouped by centre, k per centre
        dk = (pts[ei[1]] - centre[ei[0]]).norm(dim=1).view(n, self.k).max(dim=1).values
        self._check_dk(dk, max_dk_chord)
        bound = dk + 2 * r + 1e-9
        r_max = float(bound.max()) + 2 * FP32_SLACK
        cap = getattr(self, "_radius_cap", 64)
        while True:
            ei = radius(pts32, c32, r=r_max, max_num_neighbors=cap)
            if ei.shape[1] == 0 or int(torch.bincount(ei[0], minlength=n).max()) < cap:
                break
            cap *= 2  # some centre hit the cap: its list may be truncated, widen and redo
        self._radius_cap = cap
        keep = (pts[ei[1]] - centre[ei[0]]).norm(dim=1) <= bound[ei[0]]
        rows, cols = ei[0][keep], ei[1][keep]
        order = torch.argsort(rows, stable=True)
        return torch.bincount(rows, minlength=n), cols[order]

    @property
    def max_len(self) -> int:
        return int(self.len.max())

    def query(
        self, queries: torch.Tensor, k: int, block: int | None = None, num_warps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indices (int64) and chord distances of the k nearest points for each query, (Q, k) each.

        The neighbours of a query are not sorted by distance. On CUDA this is the fused Triton kernel in
        ``anemoi.graphs.triton.lookup_knn``, which documents ``block`` and ``num_warps``; otherwise torch.
        """
        assert k <= self.k, f"table built for k <= {self.k}, asked for {k}"
        q = queries.float().contiguous()
        if q.is_cuda and triton_knn.HAS_TRITON:
            out_i, out_d = triton_knn.query(
                q, self.points, self.start, self.len, self.cand, k,
                self.nlat, self.nlon, self.lat0, self.dlat, self.dlon, block=block, num_warps=num_warps,
            )  # fmt: skip
            return out_i.long(), out_d
        return self._query_torch(q, k)

    def _cells_torch(self, q: torch.Tensor) -> torch.Tensor:
        return triton_knn.cells_reference(q, self.nlat, self.nlon, self.lat0, self.dlat, self.dlon)

    def _query_torch(self, q: torch.Tensor, k: int, query_chunk: int = 1 << 20) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU (or no-Triton) query: gather each cell's candidates, padded to the widest cell.

        ``query_chunk`` bounds a (queries x widest cell) gather, a few dozen values per query, so it is much larger
        than ``cell_chunk`` in the build, which bounds a (cells x all points) distance matrix.
        """
        pts = self.points.to(q.device)
        start, length, cand = self.start.to(q.device), self.len.to(q.device), self.cand.to(q.device)
        width = self.max_len
        out_i, out_d = [], []
        for s in range(0, q.shape[0], query_chunk):
            qc = q[s : s + query_chunk]
            cell = self._cells_torch(qc)
            j = torch.arange(width, device=q.device)
            valid = j[None, :] < length[cell][:, None]
            c = cand[(start[cell][:, None] + j[None, :]).clamp(max=cand.numel() - 1)].long()
            c = torch.where(valid, c, torch.zeros_like(c))
            d = (pts[c] - qc[:, None, :]).norm(dim=2)
            d = torch.where(valid, d, torch.full_like(d, float("inf")))
            dk, pos = torch.topk(d, k, dim=1, largest=False)
            out_i.append(torch.gather(c, 1, pos))
            out_d.append(dk)
        return torch.cat(out_i), torch.cat(out_d)


MAX_CACHED_NODE_SETS = 64  # far more tables than any graph has; more means the cache is being misused
_CACHE: "dict[tuple[str, int], _CacheEntry]" = {}  # (name of the searched node set, k) -> its table
_WARNED_REBUILD: set[str] = set()


def _meta(x: torch.Tensor) -> tuple:
    """Shape, device and dtype of a coordinate tensor: compared on the host, so without synchronisation."""
    return (tuple(x.shape), str(x.device), x.dtype)


def _fingerprint(x: torch.Tensor) -> torch.Tensor:
    """Summary of the coordinate values, to detect that a cached node set has changed. Stays on the device."""
    xd = x.double()
    return torch.stack([xd.sum(), (xd * xd).sum(), xd[0, 0], xd[-1, -1]])


class _CacheEntry:
    """A cached table and what is needed to tell that it was built from the node set it is being used for.

    Every call compares the metadata, which happens on the host and so never synchronises: a node set of a
    different size, as a set of observations rebuilt per batch is, is recognised there. The coordinate values are
    compared once per table, on the first reuse, because comparing them on every call would synchronise with the
    device inside the training loop and drain its queue for far more than the comparison itself costs. A node set
    that changes under a fixed name therefore shows up on its first reuse, and a table that survives that check is
    then trusted for the rest of the run.
    """

    def __init__(self, x: torch.Tensor, k: int) -> None:
        self.meta = _meta(x)
        self.fingerprint: torch.Tensor | None = _fingerprint(x)
        self.table = LookupKNN(x, k=k)

    def matches(self, x: torch.Tensor) -> bool:
        if self.meta != _meta(x):
            return False
        if self.fingerprint is None:
            return True
        same = bool(torch.equal(self.fingerprint, _fingerprint(x)))  # once per table: the only synchronisation
        self.fingerprint = None
        return same


def clear_lookup_cache() -> None:
    """Drop all cached tables (and free their memory)."""
    _CACHE.clear()
    _WARNED_REBUILD.clear()


def lookup_knn(x: torch.Tensor, y: torch.Tensor, k: int, name: str) -> torch.Tensor:
    """Drop-in for ``torch_geometric.nn.knn(x, y, k)`` with a fixed ``x``: edge index with rows (y index, x index).

    The table for ``x`` is built on first use and cached under ``name``, the name of the searched node set (with its
    mask, if any), together with k. Each k gets its own table: a cell's candidates are the points within d_k + 2r of
    its centre, so a table built for a larger k holds longer lists and every query scans all of them. On a 167k-point
    mesh, k=3 queries cost 1.05 ms on their own table and 3.69 ms on one built for k=24.

    Each use checks ``x`` against the cached node set (see ``_CacheEntry``): if they differ (two different node sets
    under the same name, or a node set whose coordinates change between calls, as in a graph rebuilt per batch), the
    table is rebuilt with a warning. Raises LookupKNNNotSupported for point sets the table does not support.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=y.device)
    key = (name, k)
    entry = _CACHE.get(key)
    if entry is not None and not entry.matches(x):
        if name not in _WARNED_REBUILD:
            LOGGER.warning(
                "LookupKNN: the coordinates of node set %r differ from its cached table; rebuilding. Tables are "
                "cached by node-set name: two different node sets with the same name rebuild each other's table, "
                "and a node set whose coordinates change between calls is rebuilt every time (the lookup backend "
                "is meant for fixed node sets).",
                name,
            )
            _WARNED_REBUILD.add(name)
        entry = None
    if entry is None:
        if key not in _CACHE and len(_CACHE) >= MAX_CACHED_NODE_SETS:
            raise RuntimeError(
                f"LookupKNN: more than {MAX_CACHED_NODE_SETS} tables cached; the cache is keyed by the name of the "
                "searched node set and k, and is not meant for changing ones (see clear_lookup_cache)."
            )
        _CACHE[key] = _CacheEntry(x, k=k)
    idx, _ = _CACHE[key].table.query(y, k)
    rows = torch.arange(y.shape[0], device=y.device).repeat_interleave(k)
    return torch.stack([rows, idx.reshape(-1).to(y.device)], dim=0)
