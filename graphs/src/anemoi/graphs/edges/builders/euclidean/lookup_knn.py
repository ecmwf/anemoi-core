# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Exact k-nearest-neighbour search in a global point set on the unit sphere, by table lookup.

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

Layout. Queries with |lat| <= 45 deg use a grid over [-45, 45] deg; the others are rotated by 90 deg about the x axis,
(x, y, z) -> (x, -z, y), which puts them within 45 deg of the rotated equator, and use a second grid in that frame, so
no cell is polar. Candidate lists are stored flat with per-cell offsets and lengths. On CUDA a fused Triton kernel
computes each query's cell, streams its candidates and keeps the k best in registers.

Cell size. Only the speed and the table size depend on it, not the result. The default is half the searched set's
point spacing, measured from the points themselves (median nearest-neighbour distance of a sample), so no grid-specific
formula is needed. The table is built for global, roughly uniform point sets; for others (regional or strongly
stretched grids) the build refuses and callers fall back to brute force.

The KNN edge builders use this search with ``backend: lookup``. It pays off when there are many more queries than
table cells (a few per searched point) or when the same node set is searched repeatedly; a build expected to be slow
is announced with a warning.
"""

import logging
import math

import torch

LOGGER = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with CUDA builds of torch
    HAS_TRITON = False

SIN45 = math.sqrt(0.5)


class LookupKNNNotSupported(ValueError):
    """The point set is not one the lookup table is built for (see the module docstring)."""


def _latlon_deg_to_xyz(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    lat, lon = torch.deg2rad(lat), torch.deg2rad(lon)
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=1)


def _rotate(xyz: torch.Tensor) -> torch.Tensor:
    """(x, y, z) -> (x, -z, y): rotation by +90 deg about the x axis."""
    return torch.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], dim=1)


def estimate_spacing_deg(points: torch.Tensor, sample: int = 4096, seed: int = 0) -> float:
    """Typical point spacing in degrees: median nearest-neighbour distance of up to ``sample`` points."""
    n = points.shape[0]
    if n < 2:
        raise LookupKNNNotSupported("need at least two points")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[: min(sample, n)].to(points.device)
    pts = points.double()
    nn = []
    rows = max(1, (1 << 25) // n)  # about 256 MB of float64 distances per block
    for s in range(0, idx.numel(), rows):
        d = torch.cdist(pts[idx[s : s + rows]], pts)
        d[torch.arange(d.shape[0], device=d.device), idx[s : s + rows]] = float("inf")
        nn.append(d.min(dim=1).values)
    chord = torch.cat(nn).median().clamp(max=2.0)
    return math.degrees(2 * math.asin(float(chord) / 2))


if HAS_TRITON:

    @triton.jit
    def _cell(qx, qy, qz, nlat, nlon, lat0, dlat, dlon, BLOCK: tl.constexpr):
        """Cell of each query, the same formula as the build, in float32."""
        rot = tl.abs(qz) > 0.7071067811865476
        uy = tl.where(rot, -qz, qy)
        uz = tl.where(rot, qy, qz)
        rad2deg: tl.constexpr = 57.29577951308232
        lat = libdevice.asin(tl.minimum(tl.maximum(uz, -1.0), 1.0)) * rad2deg
        lon = libdevice.atan2(uy, qx) * rad2deg
        ilat = tl.floor((lat - lat0) / dlat).to(tl.int32)
        ilat = tl.minimum(tl.maximum(ilat, 0), nlat - 1)
        ilon = tl.floor((lon + 180.0) / dlon).to(tl.int32)
        ilon = tl.where(ilon >= nlon, ilon - nlon, ilon)
        ilon = tl.where(ilon < 0, ilon + nlon, ilon)
        return (rot.to(tl.int32) * nlat + ilat) * nlon + ilon

    @triton.jit
    def _query_kernel(
        q_ptr,
        pts_ptr,
        start_ptr,
        len_ptr,
        cand_ptr,
        out_i_ptr,
        out_d_ptr,
        Q,
        nlat,
        nlon,
        lat0,
        dlat,
        dlon,
        K: tl.constexpr,
        KP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < Q
        qx = tl.load(q_ptr + offs * 3 + 0, mask=m, other=1.0)
        qy = tl.load(q_ptr + offs * 3 + 1, mask=m, other=0.0)
        qz = tl.load(q_ptr + offs * 3 + 2, mask=m, other=0.0)
        cell = _cell(qx, qy, qz, nlat, nlon, lat0, dlat, dlon, BLOCK)
        start = tl.load(start_ptr + cell, mask=m, other=0)
        n = tl.load(len_ptr + cell, mask=m, other=0)

        # the k best squared distances per query; padding columns (KP > K) hold -inf so they are never the worst
        col = tl.arange(0, KP)[None, :]
        inf = float("inf")
        best_d = tl.where(col < K, tl.full([BLOCK, KP], inf, tl.float32), tl.full([BLOCK, KP], -inf, tl.float32))
        best_i = tl.full([BLOCK, KP], -1, dtype=tl.int32)
        for j in range(tl.max(n, axis=0)):
            valid = m & (j < n)
            c = tl.load(cand_ptr + start + j, mask=valid, other=0)
            px = tl.load(pts_ptr + c * 3 + 0, mask=valid, other=0.0)
            py = tl.load(pts_ptr + c * 3 + 1, mask=valid, other=0.0)
            pz = tl.load(pts_ptr + c * 3 + 2, mask=valid, other=0.0)
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            d = tl.where(valid, dx * dx + dy * dy + dz * dz, inf)
            # replace the current worst of the k best if the candidate is closer
            worst, wpos = tl.max(best_d, axis=1, return_indices=True)
            upd = (col == wpos[:, None]) & (d < worst)[:, None]
            best_d = tl.where(upd, d[:, None], best_d)
            best_i = tl.where(upd, c[:, None], best_i)

        o = offs[:, None] * K + col
        om = m[:, None] & (col < K)
        tl.store(out_i_ptr + o, best_i, mask=om)
        tl.store(out_d_ptr + o, tl.sqrt(best_d), mask=om)


class LookupKNN:
    """Exact k-nearest search in a fixed, global point set on the unit sphere (see the module docstring).

    Parameters
    ----------
    points : torch.Tensor
        (N, 3) unit vectors of the searched set.
    k : int
        Largest number of neighbours that will be queried.
    cell_deg : float, optional
        Cell size in degrees. Default: half the measured point spacing, at least 0.02 degrees.
    warn_build_seconds : float
        Log a warning before a build expected to take longer than this. The build is a brute-force search from every
        cell centre (about 6 cells per point), so its cost grows with the square of the number of points.
    max_dk_spacing : float
        Refuse to build if some cell centre's k-th nearest point is farther than this many point spacings (the set
        does not cover the globe roughly uniformly).
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
            raise LookupKNNNotSupported(f"{n} points, fewer than k={k}")
        self.spacing_deg = estimate_spacing_deg(points)
        if cell_deg is None:
            cell_deg = max(self.spacing_deg / 2, 0.02)
        self.nlat = math.ceil(90.0 / cell_deg)
        self.nlon = math.ceil(360.0 / cell_deg)
        self.lat0 = -45.0
        self.dlat = 90.0 / self.nlat
        self.dlon = 360.0 / self.nlon
        ncell = 2 * self.nlat * self.nlon
        pairs_per_second = 2e10 if points.is_cuda else 1e9  # rough: GH200 GPU / 72-core Grace CPU
        expected_s = ncell * n / pairs_per_second
        if expected_s > warn_build_seconds:
            LOGGER.warning(
                "LookupKNN: building the table for %d points (%d cells of %.3f deg) on %s may take about %.0f s "
                "(it is a brute-force search from every cell centre, done once per node set).",
                n,
                ncell,
                cell_deg,
                points.device,
                expected_s,
            )
        self.margin = margin_deg
        self.max_dk_deg = max_dk_spacing * self.spacing_deg

        counts, lists = [], []
        for frame in (points.double(), _rotate(points.double())):
            c, x = self._build_grid(frame)
            counts.append(c)
            lists.append(x)
        counts = torch.cat(counts)
        self.cand = torch.cat(lists).to(torch.int32)
        self.start = torch.zeros_like(counts)
        self.start[1:] = torch.cumsum(counts, 0)[:-1]
        self.len = counts.to(torch.int32)
        LOGGER.info(
            "LookupKNN: %d points (spacing %.3f deg), k=%d, 2 x %dx%d cells of %.3f deg, "
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
        """Candidate counts per cell and the flat candidate list of one grid, in the frame of ``pts``."""
        device = pts.device
        lat_c = self.lat0 + (torch.arange(self.nlat, device=device, dtype=torch.float64) + 0.5) * self.dlat
        hl, hn = 0.5 * self.dlat + self.margin, 0.5 * self.dlon + self.margin
        centre0 = _latlon_deg_to_xyz(lat_c, torch.zeros_like(lat_c))
        r_row = (
            torch.stack(
                [
                    (_latlon_deg_to_xyz(lat_c + a * hl, torch.full_like(lat_c, b * hn)) - centre0).norm(dim=1)
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
        chunk = max(1, min(8192, (1 << 27) // pts.shape[0]))  # about 1 GB of float64 distances per chunk
        counts, flat = [], []
        cand_k = min(max(4 * self.k, 16), pts.shape[0])
        for s in range(0, ncell, chunk):
            idx = torch.arange(s, min(s + chunk, ncell), device=device)
            ilat, ilon = idx // self.nlon, idx % self.nlon
            centre = _latlon_deg_to_xyz(lat_c[ilat], -180.0 + (ilon.double() + 0.5) * self.dlon)
            dist = torch.cdist(centre, pts)
            while True:
                dsort, isort = torch.topk(dist, cand_k, dim=1, largest=False)
                if float(dsort[:, self.k - 1].max()) > max_dk_chord:
                    raise LookupKNNNotSupported(
                        f"a cell centre's {self.k}-th nearest point is more than {self.max_dk_deg:.2f} deg away: "
                        "the point set does not cover the globe roughly uniformly"
                    )
                bound = dsort[:, self.k - 1] + 2 * r_row[ilat] + 1e-9
                inside = dsort <= bound[:, None]
                if cand_k == pts.shape[0] or not bool(inside[:, -1].any()):
                    break
                cand_k = min(2 * cand_k, pts.shape[0])  # some cell needs more candidates: widen and redo
            counts.append(inside.sum(dim=1))
            flat.append(isort[inside])  # row-major, so each cell's candidates stay contiguous
        return torch.cat(counts), torch.cat(flat)

    @property
    def max_len(self) -> int:
        return int(self.len.max())

    def query(self, queries: torch.Tensor, k: int, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """Indices (int64) and chord distances of the k nearest points for each query, (Q, k) each.

        The neighbours of a query are not sorted by distance.
        """
        assert k <= self.k, f"table built for k <= {self.k}, asked for {k}"
        q = queries.float().contiguous()
        if q.is_cuda and HAS_TRITON:
            out_i = torch.empty((q.shape[0], k), dtype=torch.int32, device=q.device)
            out_d = torch.empty((q.shape[0], k), dtype=torch.float32, device=q.device)
            if q.shape[0] > 0:
                _query_kernel[(triton.cdiv(q.shape[0], block),)](
                    q, self.points, self.start, self.len, self.cand, out_i, out_d, q.shape[0],
                    self.nlat, self.nlon, self.lat0, self.dlat, self.dlon,
                    K=k, KP=triton.next_power_of_2(k), BLOCK=block,
                )  # fmt: skip
            return out_i.long(), out_d
        return self._query_torch(q, k)

    def _cells_torch(self, q: torch.Tensor) -> torch.Tensor:
        """The kernel's cell formula in torch."""
        rot = q[:, 2].abs() > SIN45
        uy = torch.where(rot, -q[:, 2], q[:, 1])
        uz = torch.where(rot, q[:, 1], q[:, 2])
        lat = torch.rad2deg(torch.asin(uz.clamp(-1, 1)))
        lon = torch.rad2deg(torch.atan2(uy, q[:, 0]))
        ilat = torch.floor((lat - self.lat0) / self.dlat).long().clamp(0, self.nlat - 1)
        ilon = torch.floor((lon + 180.0) / self.dlon).long() % self.nlon
        return (rot.long() * self.nlat + ilat) * self.nlon + ilon

    def _query_torch(self, q: torch.Tensor, k: int, chunk: int = 1 << 20) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU (or no-Triton) query: gather each cell's candidates, padded to the widest cell."""
        pts = self.points.to(q.device)
        start, length, cand = self.start.to(q.device), self.len.to(q.device), self.cand.to(q.device)
        width = self.max_len
        out_i, out_d = [], []
        for s in range(0, q.shape[0], chunk):
            qc = q[s : s + chunk]
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


_CACHE: dict[tuple, LookupKNN | None] = {}  # None: the point set is not supported
_CACHE_SIZE = 8
_MISSING = object()


def _fingerprint(x: torch.Tensor) -> tuple:
    xd = x.double()
    return (
        tuple(x.shape),
        str(x.device),
        float(xd.sum()),
        float((xd**2).sum(dim=0)[0]),
        float(xd[0, 0]),
        float(xd[-1, -1]),
    )


def lookup_knn(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor | None:
    """Drop-in for ``torch_geometric.nn.knn(x, y, k)`` with a fixed ``x``: edge index with rows (y index, x index).

    The table for ``x`` is built on first use and cached. Returns None if ``x`` is not a point set the table supports
    (see the module docstring; the reason is logged once), in which case the caller uses its brute-force search.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=y.device)
    key = _fingerprint(x)
    table = _CACHE.get(key, _MISSING)
    if table is _MISSING or (table is not None and table.k < k):
        try:
            table = LookupKNN(x, k=max(k, 4))
        except LookupKNNNotSupported as e:
            LOGGER.warning("LookupKNN not used for this point set (%s); using brute-force KNN.", e)
            table = None
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = table
    if table is None:
        return None
    idx, _ = table.query(y, k)
    rows = torch.arange(y.shape[0], device=y.device).repeat_interleave(k)
    return torch.stack([rows, idx.reshape(-1).to(y.device)], dim=0)
