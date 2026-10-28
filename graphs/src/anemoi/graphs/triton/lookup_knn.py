# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fused query kernel of the lookup-table k-nearest-neighbour search.

The table is built in ``anemoi.graphs.edges.builders.euclidean.lookup_knn``, which also holds the exactness
argument. Here a program takes a tile of queries, computes each one's cell of the latitude/longitude grid, streams
that cell's candidates and keeps the k best squared distances in registers, so the (queries x candidates) distances
are never written out.

``cell_index`` is the single source of the cell formula: the kernel calls it, and ``cells_reference`` is the same
formula in torch for the CPU path. ``test_lookup_knn.py`` asserts that the two agree.
"""

import logging

import torch

LOGGER = logging.getLogger(__name__)


class DummyModule:
    """Dummy module to allow import of Triton modules when Triton is not available."""

    def __getattribute__(self, name):
        return DummyModule()

    def __call__(self, *_, **__):
        return DummyModule()

    def __getitem__(self, key):
        return DummyModule()


try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError as e:  # pragma: no cover - triton ships with CUDA builds of torch
    LOGGER.debug(
        "Triton failed to import, the lookup KNN query falls back to torch: %s", e
    )
    triton = DummyModule()
    tl = DummyModule()
    libdevice = DummyModule()
    HAS_TRITON = False


@triton.jit
def cell_index(qx, qy, qz, nlat, nlon, lat0, dlat, dlon):
    """Cell of each query on the latitude/longitude grid, in float32."""
    rad2deg: tl.constexpr = (
        57.29577951308232  # a jitted function may not read a plain module global
    )
    lat = libdevice.asin(tl.minimum(tl.maximum(qz, -1.0), 1.0)) * rad2deg
    lon = libdevice.atan2(qy, qx) * rad2deg
    ilat = tl.floor((lat - lat0) / dlat).to(tl.int32)
    ilat = tl.minimum(tl.maximum(ilat, 0), nlat - 1)
    ilon = tl.floor((lon + 180.0) / dlon).to(tl.int32)
    ilon = tl.where(ilon >= nlon, ilon - nlon, ilon)
    ilon = tl.where(ilon < 0, ilon + nlon, ilon)
    return ilat * nlon + ilon


def cells_reference(
    q: torch.Tensor, nlat: int, nlon: int, lat0: float, dlat: float, dlon: float
) -> torch.Tensor:
    """``cell_index`` in torch, for queries on the CPU or without Triton. Kept equal to it by a test."""
    lat = torch.rad2deg(torch.asin(q[:, 2].clamp(-1, 1)))
    lon = torch.rad2deg(torch.atan2(q[:, 1], q[:, 0]))
    ilat = torch.floor((lat - lat0) / dlat).long().clamp(0, nlat - 1)
    ilon = torch.floor((lon + 180.0) / dlon).long() % nlon
    return ilat * nlon + ilon


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
    cell = cell_index(qx, qy, qz, nlat, nlon, lat0, dlat, dlon)
    start = tl.load(start_ptr + cell, mask=m, other=0)
    n = tl.load(len_ptr + cell, mask=m, other=0)

    # the k best squared distances per query; padding columns (KP > K) hold -inf so they are never the worst
    col = tl.arange(0, KP)[None, :]
    inf = float("inf")
    best_d = tl.where(
        col < K,
        tl.full([BLOCK, KP], inf, tl.float32),
        tl.full([BLOCK, KP], -inf, tl.float32),
    )
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


def query(
    q: torch.Tensor,
    points: torch.Tensor,
    start: torch.Tensor,
    lengths: torch.Tensor,
    cand: torch.Tensor,
    k: int,
    nlat: int,
    nlon: int,
    lat0: float,
    dlat: float,
    dlon: float,
    block: int | None = None,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (int32) and chord distances of the k nearest points for each query, (Q, k) each, unordered.

    ``block`` is the number of queries per program; by default 64 for k <= 4 and 32 above, with one warp. On a GH200
    this was the fastest of 20 settings (16-256 queries x 1-8 warps) for k = 3-64 on two meshes, and within 7% for
    k = 1; larger tiles spill the per-query top-k buffer out of registers as k grows. A fixed rule avoids autotuning,
    whose first call costs seconds.
    """
    if block is None:
        block = 64 if k <= 4 else 32
    out_i = torch.empty((q.shape[0], k), dtype=torch.int32, device=q.device)
    out_d = torch.empty((q.shape[0], k), dtype=torch.float32, device=q.device)
    if q.shape[0] > 0:
        _query_kernel[(triton.cdiv(q.shape[0], block),)](
            q, points, start, lengths, cand, out_i, out_d, q.shape[0],
            nlat, nlon, lat0, dlat, dlon,
            K=k, KP=triton.next_power_of_2(k), BLOCK=block, num_warps=num_warps,
        )  # fmt: skip
    return out_i, out_d
