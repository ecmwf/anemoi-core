# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton kernels for neighbourhood cross attention between two reduced grids, e.g. O1280 and O96.

Queries live on one reduced grid and keys and values on another (see
:class:`anemoi.models.layers.reduced_grid.ReducedGridCrossNeighbourhoodMask` for the rule). Each
query row is matched to the key row nearest in latitude through a small table, and the window's
key rows start from there; the longitude matching is the same as for self attention.

The kernels are those of :mod:`anemoi.models.triton.reduced_grid_attention` with the query and key
grids kept apart, and they reuse its helpers and tile tables. Query tiles (forward and dQ) are
tiles of the query grid; key tiles (dK and dV) are tiles of the key grid, and a second table gives
for every key row the first and last query row whose window includes it.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.triton.reduced_grid_attention import _MASKED
from anemoi.models.triton.reduced_grid_attention import _MAX_TILE_H
from anemoi.models.triton.reduced_grid_attention import _RCP_LN2
from anemoi.models.triton.reduced_grid_attention import _backward_configs
from anemoi.models.triton.reduced_grid_attention import _configs
from anemoi.models.triton.reduced_grid_attention import _grid_tables
from anemoi.models.triton.reduced_grid_attention import _launch_grid
from anemoi.models.triton.reduced_grid_attention import _matching
from anemoi.models.triton.reduced_grid_attention import _matching_shifted
from anemoi.models.triton.reduced_grid_attention import _near_in_row
from anemoi.models.triton.reduced_grid_attention import _packed_tiles
from anemoi.models.triton.reduced_grid_attention import _pick
from anemoi.models.triton.reduced_grid_attention import _rows_info
from anemoi.models.triton.reduced_grid_attention import _shift_table
from anemoi.models.triton.reduced_grid_attention import _spread
from anemoi.models.triton.reduced_grid_attention import _tile_points
from anemoi.models.triton.reduced_grid_attention import _tile_start
from anemoi.models.triton.reduced_grid_attention import _union
from anemoi.models.triton.reduced_grid_attention import _wrap

# Larger than any row index, for taking minima and maxima over rows.
_FAR = tl.constexpr(1 << 30)

_AUTOTUNE_KEY = [
    "Q_ROWS",
    "Q_POINTS",
    "K_ROWS",
    "K_POINTS",
    "HEAD_DIM",
    "KERNEL_H",
    "KERNEL_W",
    "DOT_PRECISION",
    "SHIFTED",
]


def _cross_tables(query_grid: ReducedGrid, key_grid: ReducedGrid, kernel_h: int, device: torch.device) -> dict:
    """Row tables linking the two grids, and the sizes of the per-tile row vectors.

    ``window_start[r]`` is the first key row seen from query row ``r``; ``see_first[k]`` and
    ``see_last[k]`` are the first and last query rows whose window includes key row ``k``
    (``see_first > see_last`` when there are none).

    The tables are kept for each pair of grids. Grids compare equal on their row lengths and
    shifts only, but the row latitudes decide which key rows a query row sees, so they are part
    of what the tables are kept by: parts of a grid with the same rows at other latitudes, such
    as bands of rows, get their own tables.
    """
    return _cross_tables_for(query_grid, key_grid, query_grid.row_latitudes, key_grid.row_latitudes, kernel_h, device)


@lru_cache(maxsize=512)
def _cross_tables_for(
    query_grid: ReducedGrid,
    key_grid: ReducedGrid,
    query_latitudes: tuple[float, ...],
    key_latitudes: tuple[float, ...],
    kernel_h: int,
    device: torch.device,
) -> dict:
    num_q_rows, num_k_rows = query_grid.num_rows, key_grid.num_rows
    start = (query_grid.nearest_rows(key_grid) - kernel_h // 2).clamp(0, num_k_rows - kernel_h)
    key_rows = torch.arange(num_k_rows)
    sees = (key_rows[None, :] >= start[:, None]) & (key_rows[None, :] < start[:, None] + kernel_h)
    has = sees.any(dim=0)
    first = torch.where(has, sees.int().argmax(dim=0), num_q_rows)
    last = torch.where(has, num_q_rows - 1 - sees.flip(0).int().argmax(dim=0), -1)

    # Most key rows a query tile reaches, and most query rows that see a key tile.
    widest_keys = max(
        int(start[min(r + _MAX_TILE_H - 1, num_q_rows - 1)] - start[r]) + kernel_h for r in range(num_q_rows)
    )
    widest_queries = 1
    for k in range(num_k_rows):
        tile = slice(k, min(k + _MAX_TILE_H, num_k_rows))
        seen = has[tile]
        if seen.any():
            widest_queries = max(widest_queries, int(last[tile][seen].max() - first[tile][seen].min()) + 1)

    def on_device(t):
        return t.to(dtype=torch.int32, device=device)

    return dict(
        window_start=on_device(start),
        see_first=on_device(first),
        see_last=on_device(last),
        KEY_ROWS=triton.next_power_of_2(widest_keys),
        QUERY_ROWS=triton.next_power_of_2(widest_queries),
    )


@triton.jit
def _wrap_twice(pos, n):
    """``pos`` taken round the globe into ``0 <= pos < n``, for ``-2 * n <= pos < 2 * n``."""
    pos = tl.where(pos < 0, pos + n, pos)
    return _wrap(pos, n)


@triton.jit
def _query_tile_keys(
    K_ROW_START,
    K_ROW_SHIFT,
    WINDOW_START,
    first_row,
    lo_h,
    hi_h,
    lengths_h,
    shift_h,
    SHIFTED: tl.constexpr,
    Q_ROWS: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """First key row seen by each query of a tile, and the key rows the tile reaches with their shifts and stretches."""
    RADIUS_W: tl.constexpr = KERNEL_W // 2
    rows_h = first_row + tl.arange(0, TILE_H)
    valid_h = rows_h < Q_ROWS
    window_h = tl.load(WINDOW_START + rows_h, mask=valid_h, other=0)
    q_window = _spread(window_h, TILE_H, TILE_W)
    k_first_row = tl.min(tl.where(valid_h, window_h, _FAR))
    num_k_rows = tl.max(tl.where(valid_h, window_h, -_FAR)) + KERNEL_H - k_first_row
    idx, valid, k_starts, k_lengths = _rows_info(K_ROW_START, k_first_row, num_k_rows, KEY_ROWS)

    nonempty = hi_h > lo_h
    n = tl.maximum(lengths_h, 1)
    if SHIFTED:
        k_shifts = tl.load(K_ROW_SHIFT + k_first_row + idx, mask=valid, other=0)
        firsts = _matching_shifted(lo_h[None, :], n[None, :], k_lengths[:, None], shift_h[None, :], k_shifts[:, None])
        lasts = _matching_shifted(
            hi_h[None, :] - 1, n[None, :], k_lengths[:, None], shift_h[None, :], k_shifts[:, None]
        )
        firsts = firsts - RADIUS_W
        lasts = lasts + RADIUS_W
    else:
        k_shifts = idx * 0
        firsts = _matching(lo_h[None, :], n[None, :], k_lengths[:, None]) - RADIUS_W
        lasts = _matching(hi_h[None, :] - 1, n[None, :], k_lengths[:, None]) + RADIUS_W
    first, length = _union(firsts, lasts, nonempty, k_lengths)
    blocks = tl.max(tl.where(valid, tl.cdiv(length, BLOCK_ITER), 0))
    return q_window, k_first_row, num_k_rows, idx, k_starts, k_lengths, k_shifts, first, length, blocks


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _xg_fwd(
    Q,
    K,
    V,
    OUT,
    M,  # row maximum of the scaled scores of each query (in log2 units), shape (batch * heads, query points)
    INV_L,  # inverse of the softmax sum of each query, same shape as M
    Q_ROW_START,
    K_ROW_START,
    Q_ROW_SHIFT,
    K_ROW_SHIFT,
    WINDOW_START,
    SEE_FIRST,
    SEE_LAST,
    QTILES,
    QTILE_OFFSETS,
    KTILES,
    KTILE_OFFSETS,
    sm_scale,
    Q_ROWS: tl.constexpr,
    Q_POINTS: tl.constexpr,
    K_ROWS: tl.constexpr,
    K_POINTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    DOT_PRECISION: tl.constexpr,  # "ieee" for float32 inputs, so their products are not rounded to TF32
    SHIFTED: tl.constexpr,  # either grid has rows shifted by half spacings (HEALPix); False compiles it away
    KEY_ROWS: tl.constexpr,  # most key rows one query tile reaches, rounded up to a power of 2
    QUERY_ROWS: tl.constexpr,  # most query rows that see one key tile, rounded up to a power of 2
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes the output of one tile of queries, for one batch entry and head."""
    RADIUS_W: tl.constexpr = KERNEL_W // 2
    bh = tl.program_id(1).to(tl.int64)
    q_base = bh * Q_POINTS * HEAD_DIM
    k_base = bh * K_POINTS * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    first_row, band = _tile_start(QTILES, QTILE_OFFSETS, TILE_H, TILE_W)
    q_rows, q_pos, q_valid, q_n, q_tok, lo_h, hi_h, lengths_h = _tile_points(
        Q_ROW_START, first_row, band, Q_ROWS, TILE_H, TILE_W
    )
    q = tl.load(Q + q_base + q_tok[:, None] * HEAD_DIM + d[None, :], mask=q_valid[:, None], other=0.0)
    if SHIFTED:
        tile_rows = first_row + tl.arange(0, TILE_H)
        shift_h = tl.load(Q_ROW_SHIFT + tile_rows, mask=tile_rows < Q_ROWS, other=0)
    else:
        shift_h = lengths_h * 0
    q_shift = _spread(shift_h, TILE_H, TILE_W)

    qk_scale = sm_scale * _RCP_LN2
    m_i = tl.full([TILE_H * TILE_W], _MASKED, dtype=tl.float32)
    l_i = tl.zeros([TILE_H * TILE_W], dtype=tl.float32)
    acc = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)

    q_window, k_first_row, num_k_rows, idx, k_starts, k_lengths, k_shifts, firsts, lengths, blocks = _query_tile_keys(
        K_ROW_START,
        K_ROW_SHIFT,
        WINDOW_START,
        first_row,
        lo_h,
        hi_h,
        lengths_h,
        shift_h,
        SHIFTED,
        Q_ROWS,
        KERNEL_H,
        KERNEL_W,
        TILE_H,
        TILE_W,
        KEY_ROWS,
        BLOCK_ITER,
    )
    offs = tl.arange(0, BLOCK_ITER)
    for it in range(0, num_k_rows * blocks):
        i = it // blocks
        k_row = k_first_row + i
        k_row_start = _pick(k_starts, idx, i)
        other_n = _pick(k_lengths, idx, i)
        first = _pick(firsts, idx, i)
        length = _pick(lengths, idx, i)
        t = (it % blocks) * BLOCK_ITER + offs
        k_valid = t < length
        k_pos = _wrap(first + t, other_n)
        k_tok = k_row_start + k_pos
        k = tl.load(K + k_base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + k_base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)

        row_seen = (k_row >= q_window) & (k_row < q_window + KERNEL_H)
        if SHIFTED:
            centre = _matching_shifted(q_pos, q_n, other_n, q_shift, _pick(k_shifts, idx, i))
        else:
            centre = _matching(q_pos, q_n, other_n)
        keep = _near_in_row(k_pos[None, :], centre[:, None], other_n, RADIUS_W)
        keep = keep & row_seen[:, None] & k_valid[None, :]
        qk = tl.where(keep, tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION) * qk_scale, _MASKED)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision=DOT_PRECISION)
        m_i = m_new

    inv_l = 1.0 / l_i
    acc = acc * inv_l[:, None]
    q_ptrs = q_base + q_tok[:, None] * HEAD_DIM + d[None, :]
    tl.store(OUT + q_ptrs, acc.to(OUT.dtype.element_ty), mask=q_valid[:, None])
    tl.store(M + bh * Q_POINTS + q_tok, m_i, mask=q_valid)
    tl.store(INV_L + bh * Q_POINTS + q_tok, inv_l, mask=q_valid)


@triton.autotune(configs=_backward_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _xg_bwd_dq(
    Q,
    K,
    V,
    OUT,
    DO,
    DQ,
    M,
    INV_L,
    DELTA,  # written here: sum over the head dimension of OUT * DO for each query, read by the dK/dV kernel
    LAM,  # written here: sum of the rounded ds over sum of the rounded probabilities per query, see below
    Q_ROW_START,
    K_ROW_START,
    Q_ROW_SHIFT,
    K_ROW_SHIFT,
    WINDOW_START,
    SEE_FIRST,
    SEE_LAST,
    QTILES,
    QTILE_OFFSETS,
    KTILES,
    KTILE_OFFSETS,
    sm_scale,
    Q_ROWS: tl.constexpr,
    Q_POINTS: tl.constexpr,
    K_ROWS: tl.constexpr,
    K_POINTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    SHIFTED: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes dQ for one tile of queries, visiting the same keys as the forward pass."""
    RADIUS_W: tl.constexpr = KERNEL_W // 2
    bh = tl.program_id(1).to(tl.int64)
    q_base = bh * Q_POINTS * HEAD_DIM
    k_base = bh * K_POINTS * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    first_row, band = _tile_start(QTILES, QTILE_OFFSETS, TILE_H, TILE_W)
    q_rows, q_pos, q_valid, q_n, q_tok, lo_h, hi_h, lengths_h = _tile_points(
        Q_ROW_START, first_row, band, Q_ROWS, TILE_H, TILE_W
    )
    q_ptrs = q_base + q_tok[:, None] * HEAD_DIM + d[None, :]
    q = tl.load(Q + q_ptrs, mask=q_valid[:, None], other=0.0)
    do = tl.load(DO + q_ptrs, mask=q_valid[:, None], other=0.0)
    out = tl.load(OUT + q_ptrs, mask=q_valid[:, None], other=0.0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(DELTA + bh * Q_POINTS + q_tok, delta, mask=q_valid)
    m = tl.load(M + bh * Q_POINTS + q_tok, mask=q_valid, other=0.0)
    inv_l = tl.load(INV_L + bh * Q_POINTS + q_tok, mask=q_valid, other=0.0)
    if SHIFTED:
        tile_rows = first_row + tl.arange(0, TILE_H)
        shift_h = tl.load(Q_ROW_SHIFT + tile_rows, mask=tile_rows < Q_ROWS, other=0)
    else:
        shift_h = lengths_h * 0
    q_shift = _spread(shift_h, TILE_H, TILE_W)

    qk_scale = sm_scale * _RCP_LN2
    dq = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)
    # The exact ds of a query sums to zero over its keys, which keeps dq blind to where the keys sit
    # as a group. Rounding ds to the input precision before the product with k leaves a small sum,
    # and dq picks up that sum times the mean key, which outweighs the true gradient when the keys
    # are far from the origin compared with their spread. The sums of the rounded ds and of the
    # rounded probabilities are kept, and lam = their ratio, so that ds - lam * p sums to zero; dq
    # and dk use that. See Chen et al., "Broken symmetry in BF16 attention" (GProj),
    # https://arxiv.org/abs/2609.34272.
    p_k = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)
    # The sums are kept per key column and added up after the loop, so each step only adds elementwise.
    ds_sums = tl.zeros([TILE_H * TILE_W, BLOCK_ITER], dtype=tl.float32)
    p_sums = tl.zeros([TILE_H * TILE_W, BLOCK_ITER], dtype=tl.float32)

    q_window, k_first_row, num_k_rows, idx, k_starts, k_lengths, k_shifts, firsts, lengths, blocks = _query_tile_keys(
        K_ROW_START,
        K_ROW_SHIFT,
        WINDOW_START,
        first_row,
        lo_h,
        hi_h,
        lengths_h,
        shift_h,
        SHIFTED,
        Q_ROWS,
        KERNEL_H,
        KERNEL_W,
        TILE_H,
        TILE_W,
        KEY_ROWS,
        BLOCK_ITER,
    )
    # Each row gets only the blocks its own stretch needs, rather than as many as the longest stretch,
    # so blocks whose keys would all be masked out are never computed; ends holds the running total of
    # blocks per row. Finding the row and block of each step costs a little index arithmetic, so this
    # gains most where the stretches differ in length (near the poles, between grids of different
    # resolution) and can be a few percent slower where they are about equally long.
    row_blocks = tl.where(idx < num_k_rows, tl.cdiv(lengths, BLOCK_ITER), 0)
    ends = tl.cumsum(row_blocks, 0)
    offs = tl.arange(0, BLOCK_ITER)
    for it in range(0, tl.sum(row_blocks)):
        i = tl.sum((ends <= it).to(tl.int32))
        block = it - _pick(ends - row_blocks, idx, i)
        k_row = k_first_row + i
        k_row_start = _pick(k_starts, idx, i)
        other_n = _pick(k_lengths, idx, i)
        first = _pick(firsts, idx, i)
        length = _pick(lengths, idx, i)
        t = block * BLOCK_ITER + offs
        k_valid = t < length
        k_pos = _wrap(first + t, other_n)
        k_tok = k_row_start + k_pos
        k = tl.load(K + k_base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + k_base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)

        row_seen = (k_row >= q_window) & (k_row < q_window + KERNEL_H)
        if SHIFTED:
            centre = _matching_shifted(q_pos, q_n, other_n, q_shift, _pick(k_shifts, idx, i))
        else:
            centre = _matching(q_pos, q_n, other_n)
        keep = _near_in_row(k_pos[None, :], centre[:, None], other_n, RADIUS_W)
        keep = keep & row_seen[:, None] & k_valid[None, :] & q_valid[:, None]
        qk = tl.where(keep, tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION) * qk_scale, _MASKED)
        p = tl.math.exp2(qk - m[:, None]) * inv_l[:, None]

        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        p = p.to(k.dtype)
        ds_sums += ds.to(tl.float32)
        p_sums += p.to(tl.float32)
        dq = tl.dot(ds, k, dq, input_precision=DOT_PRECISION)
        p_k = tl.dot(p, k, p_k, input_precision=DOT_PRECISION)

    ds_sum = tl.sum(ds_sums, 1)
    p_sum = tl.sum(p_sums, 1)
    lam = tl.where(p_sum > 0, ds_sum / p_sum, 0.0)
    tl.store(LAM + bh * Q_POINTS + q_tok, lam, mask=q_valid)
    dq = (dq - lam[:, None] * p_k) * sm_scale
    tl.store(DQ + q_ptrs, dq.to(DQ.dtype.element_ty), mask=q_valid[:, None])


@triton.autotune(configs=_backward_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _xg_bwd_dkdv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    M,
    INV_L,
    DELTA,
    LAM,
    Q_ROW_START,
    K_ROW_START,
    Q_ROW_SHIFT,
    K_ROW_SHIFT,
    WINDOW_START,
    SEE_FIRST,
    SEE_LAST,
    QTILES,
    QTILE_OFFSETS,
    KTILES,
    KTILE_OFFSETS,
    sm_scale,
    Q_ROWS: tl.constexpr,
    Q_POINTS: tl.constexpr,
    K_ROWS: tl.constexpr,
    K_POINTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    SHIFTED: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes dK and dV for one tile of keys, visiting every query whose window reaches it."""
    RADIUS_W: tl.constexpr = KERNEL_W // 2
    bh = tl.program_id(1).to(tl.int64)
    q_base = bh * Q_POINTS * HEAD_DIM
    k_base = bh * K_POINTS * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    first_row, band = _tile_start(KTILES, KTILE_OFFSETS, TILE_H, TILE_W)
    k_rows, k_pos, k_valid, k_n, k_tok, lo_h, hi_h, lengths_h = _tile_points(
        K_ROW_START, first_row, band, K_ROWS, TILE_H, TILE_W
    )
    k_ptrs = k_base + k_tok[:, None] * HEAD_DIM + d[None, :]
    k = tl.load(K + k_ptrs, mask=k_valid[:, None], other=0.0)
    v = tl.load(V + k_ptrs, mask=k_valid[:, None], other=0.0)
    if SHIFTED:
        tile_rows = first_row + tl.arange(0, TILE_H)
        shift_h = tl.load(K_ROW_SHIFT + tile_rows, mask=tile_rows < K_ROWS, other=0)
    else:
        shift_h = lengths_h * 0

    qk_scale = sm_scale * _RCP_LN2
    dk = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)

    # Query rows whose windows reach the tile's key rows, looked up once with their window starts.
    rows_h = first_row + tl.arange(0, TILE_H)
    valid_h = rows_h < K_ROWS
    see_first = tl.load(SEE_FIRST + rows_h, mask=valid_h, other=_FAR)
    see_last = tl.load(SEE_LAST + rows_h, mask=valid_h, other=-1)
    seen = valid_h & (see_last >= see_first)
    num_q_rows = tl.maximum(tl.max(tl.where(seen, see_last, -1)) - tl.min(tl.where(seen, see_first, _FAR)) + 1, 0)
    q_first_row = tl.where(num_q_rows > 0, tl.min(tl.where(seen, see_first, _FAR)), 0)
    idx, row_valid, q_starts, q_lengths = _rows_info(Q_ROW_START, q_first_row, num_q_rows, QUERY_ROWS)
    q_windows = tl.load(WINDOW_START + q_first_row + idx, mask=row_valid, other=0)

    # In each query row, the queries whose matching position in a tile row lies within RADIUS_W of the
    # tile's points there. The matching position only grows along a row, so inverting it gives one
    # stretch per tile row; only the tile rows inside the query row's window count. Two turns are
    # added so the divisions work on positive numbers.
    nonempty = hi_h > lo_h
    n_k = tl.maximum(lengths_h, 1)
    if SHIFTED:
        q_shifts = tl.load(Q_ROW_SHIFT + q_first_row + idx, mask=row_valid, other=0)
    else:
        q_shifts = idx * 0
    two_n_k = 2 * n_k
    low = 2 * (lo_h - RADIUS_W + 2 * n_k) - 1 + shift_h
    high = 2 * (hi_h - 1 + RADIUS_W + 2 * n_k) + 1 + shift_h
    q_offset = two_n_k[None, :] - 1 - q_shifts[:, None] * n_k[None, :]
    firsts_h = (low[None, :] * q_lengths[:, None] + q_offset) // two_n_k[None, :]
    lasts_h = (high[None, :] * q_lengths[:, None] + q_offset) // two_n_k[None, :] - 1
    tile_rows = first_row + tl.arange(0, TILE_H)
    sees = (
        nonempty[None, :]
        & (tile_rows[None, :] >= q_windows[:, None])
        & (tile_rows[None, :] < q_windows[:, None] + KERNEL_H)
    )
    lowest = tl.min(tl.where(sees, firsts_h, _FAR), axis=1) - 2 * q_lengths
    highest = tl.max(tl.where(sees, lasts_h, -_FAR), axis=1) - 2 * q_lengths
    whole = highest - lowest + 1 >= q_lengths
    firsts = tl.where(whole, 0, lowest)
    lengths = tl.maximum(tl.where(whole, q_lengths, highest - lowest + 1), 0)

    # Each row gets only the blocks its own stretch needs, rather than as many as the longest stretch,
    # so blocks whose keys would all be masked out are never computed; ends holds the running total of
    # blocks per row. Finding the row and block of each step costs a little index arithmetic, so this
    # gains most where the stretches differ in length (near the poles, between grids of different
    # resolution) and can be a few percent slower where they are about equally long.
    row_blocks = tl.where(row_valid, tl.cdiv(lengths, BLOCK_ITER), 0)
    ends = tl.cumsum(row_blocks, 0)
    offs = tl.arange(0, BLOCK_ITER)
    for it in range(0, tl.sum(row_blocks)):
        i = tl.sum((ends <= it).to(tl.int32))
        block = it - _pick(ends - row_blocks, idx, i)
        q_row_start = _pick(q_starts, idx, i)
        n = _pick(q_lengths, idx, i)
        q_window = _pick(q_windows, idx, i)
        first = _pick(firsts, idx, i)
        length = _pick(lengths, idx, i)
        t = block * BLOCK_ITER + offs
        q_valid = t < length
        q_pos = _wrap_twice(first + t, n)
        q_tok = q_row_start + q_pos
        q_ptrs = q_base + q_tok[:, None] * HEAD_DIM + d[None, :]
        q = tl.load(Q + q_ptrs, mask=q_valid[:, None], other=0.0)
        do = tl.load(DO + q_ptrs, mask=q_valid[:, None], other=0.0)
        m = tl.load(M + bh * Q_POINTS + q_tok, mask=q_valid, other=0.0)
        inv_l = tl.load(INV_L + bh * Q_POINTS + q_tok, mask=q_valid, other=0.0)
        delta = tl.load(DELTA + bh * Q_POINTS + q_tok, mask=q_valid, other=0.0)
        lam = tl.load(LAM + bh * Q_POINTS + q_tok, mask=q_valid, other=0.0)

        row_seen = (k_rows >= q_window) & (k_rows < q_window + KERNEL_H)
        # Matching position of each query in each tile row, repeated for the points of that row.
        if SHIFTED:
            centre_h = _matching_shifted(q_pos[None, :], n, n_k[:, None], _pick(q_shifts, idx, i), shift_h[:, None])
        else:
            centre_h = _matching(q_pos[None, :], n, n_k[:, None])
        centre = tl.reshape(
            tl.broadcast_to(centre_h[:, None, :], (TILE_H, TILE_W, BLOCK_ITER)), (TILE_H * TILE_W, BLOCK_ITER)
        )
        keep = _near_in_row(k_pos[:, None], centre, k_n[:, None], RADIUS_W)
        keep = keep & row_seen[:, None] & q_valid[None, :] & k_valid[:, None]
        # Scores laid out keys x queries.
        qk_t = tl.where(keep, tl.dot(k, tl.trans(q), input_precision=DOT_PRECISION) * qk_scale, _MASKED)
        p_t = tl.math.exp2(qk_t - m[None, :]) * inv_l[None, :]

        dp_t = tl.dot(v, tl.trans(do), input_precision=DOT_PRECISION)
        ds_t = p_t * (dp_t - delta[None, :])
        dv = tl.dot(p_t.to(do.dtype), do, dv, input_precision=DOT_PRECISION)
        # dk from ds - lam * p, with lam from the dQ kernel, so the rounded ds rows sum to zero there too.
        dk = tl.dot(ds_t.to(q.dtype), q, dk, input_precision=DOT_PRECISION)
        # This deviates from the paper (arXiv:2609.34272, appendix B, eq. 13), which rounds p and lam * q
        # separately: here -lam * p is rounded once and multiplied with q, which is already 16-bit.
        # The paper's version:
        # dk = tl.dot(p_t, (-lam[:, None] * q).to(q.dtype), dk, input_precision=DOT_PRECISION)
        dk = tl.dot((p_t * -lam[None, :]).to(q.dtype), q, dk, input_precision=DOT_PRECISION)

    dk = dk * sm_scale
    tl.store(DK + k_ptrs, dk.to(DK.dtype.element_ty), mask=k_valid[:, None])
    tl.store(DV + k_ptrs, dv.to(DV.dtype.element_ty), mask=k_valid[:, None])


class ReducedGridCrossAttentionTriton(torch.autograd.Function):
    """Neighbourhood cross attention between two reduced grids, computed with Triton kernels.

    Queries are ``(batch, heads, query points, head_dim)`` on ``query_grid``; keys and values are
    ``(batch, heads, key points, head_dim)`` on ``key_grid``, both stored as in
    :class:`anemoi.models.layers.reduced_grid.ReducedGrid`. ``head_dim`` must be a power of two of
    at least 16.
    """

    @staticmethod
    def forward(ctx, q, k, v, query_grid, key_grid, kernel_size, sm_scale):
        batch, heads, q_points, head_dim = q.shape
        assert q_points == query_grid.num_points, f"Expected {query_grid.num_points} queries, got {q_points}."
        assert k.shape == v.shape == (batch, heads, key_grid.num_points, head_dim), "k and v must match the key grid."
        assert (
            head_dim >= 16 and head_dim & (head_dim - 1) == 0
        ), f"head_dim must be a power of 2 >= 16, got {head_dim}."
        assert kernel_size[0] <= key_grid.num_rows, "kernel_size[0] must not exceed the number of key rows."

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)
        m = torch.empty((batch * heads, q_points), device=q.device, dtype=torch.float32)
        inv_l = torch.empty_like(m)
        q_row_starts, q_tiles = _grid_tables(query_grid.row_lengths, q.device)
        k_row_starts, k_tiles = _grid_tables(key_grid.row_lengths, q.device)
        cross = _cross_tables(query_grid, key_grid, kernel_size[0], q.device)
        tables = (
            q_row_starts,
            k_row_starts,
            _shift_table(query_grid.shifts, q.device),
            _shift_table(key_grid.shifts, q.device),
            cross["window_start"],
            cross["see_first"],
            cross["see_last"],
            *_packed_tiles(query_grid.row_lengths, q.device),
            *_packed_tiles(key_grid.row_lengths, q.device),
        )
        sizes = dict(
            Q_ROWS=query_grid.num_rows,
            Q_POINTS=q_points,
            K_ROWS=key_grid.num_rows,
            K_POINTS=key_grid.num_points,
            HEAD_DIM=head_dim,
            KERNEL_H=kernel_size[0],
            KERNEL_W=kernel_size[1],
            # Float32 inputs keep full precision in the matrix products; 16-bit inputs use tensor cores as usual.
            DOT_PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
            SHIFTED=query_grid.is_shifted or key_grid.is_shifted,
            KEY_ROWS=cross["KEY_ROWS"],
            QUERY_ROWS=cross["QUERY_ROWS"],
        )

        _xg_fwd[_launch_grid(q_tiles, batch * heads)](q, k, v, o, m, inv_l, *tables, sm_scale, **sizes)
        ctx.save_for_backward(q, k, v, o, m, inv_l)
        ctx.grids = (query_grid, key_grid)
        ctx.tables = tables
        ctx.q_tiles, ctx.k_tiles = q_tiles, k_tiles
        ctx.sizes = sizes
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, inv_l = ctx.saved_tensors
        batch, heads, _, _ = q.shape

        do = do.contiguous()
        delta, lam = torch.empty_like(m), torch.empty_like(m)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        # The dQ kernel also writes delta and lam, which the dK/dV kernel reads, so it runs first.
        _xg_bwd_dq[_launch_grid(ctx.q_tiles, batch * heads)](
            q, k, v, o, do, dq, m, inv_l, delta, lam, *ctx.tables, ctx.sm_scale, **ctx.sizes
        )
        _xg_bwd_dkdv[_launch_grid(ctx.k_tiles, batch * heads)](
            q, k, v, do, dk, dv, m, inv_l, delta, lam, *ctx.tables, ctx.sm_scale, **ctx.sizes
        )
        return dq, dk, dv, None, None, None, None


def reduced_grid_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_grid: ReducedGrid,
    key_grid: ReducedGrid,
    kernel_size: tuple[int, int],
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Neighbourhood cross attention between two reduced grids; see :class:`ReducedGridCrossAttentionTriton`.

    ``sm_scale`` defaults to ``1 / sqrt(head_dim)``, as in ``scaled_dot_product_attention``.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return ReducedGridCrossAttentionTriton.apply(q, k, v, query_grid, key_grid, tuple(kernel_size), sm_scale)
