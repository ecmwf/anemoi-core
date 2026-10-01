# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton kernels for neighbourhood attention on reduced grids such as octahedral O48, O96, ...

The grid is a set of latitude rows of different lengths, stored row by row (see
:class:`anemoi.models.layers.reduced_grid.ReducedGrid`). Each query attends to ``kernel_h`` rows
around its own row and, in each of them, to the ``kernel_w`` points nearest in longitude (see
:class:`anemoi.models.layers.reduced_grid.ReducedGridNeighbourhoodMask`). Longitudes wrap around;
a row shorter than the window contributes its whole ring, each point once.

The kernels follow the same flash attention scheme as the regular-grid kernels in
:mod:`anemoi.models.triton.neighbourhood_attention`. Each program takes a tile of ``TILE_H``
neighbouring rows and one band of longitude: in every row it holds the points whose longitude lies
in the band. The band edges come from the longest row of the group, so a row never has more than
``TILE_W`` points in a tile, and every point belongs to exactly one tile. Because the matching
longitude only grows along a row, the keys a tile needs in another row form a single stretch of
that row, which may wrap past longitude 0. The backward pass for dK and dV uses the same tiles for
the keys and walks the queries the same way, in the opposite direction.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from anemoi.models.layers.reduced_grid import ReducedGrid

# Stands in for minus infinity on masked scores; finite so fully masked rows stay free of NaNs.
_MASKED = tl.constexpr(-1.0e30)
# 1 / ln(2), so the softmax can use exp2.
_RCP_LN2 = tl.constexpr(1.4426950408889634)
# Larger than any position in any row, for taking minima and maxima over rows.
_FAR = tl.constexpr(1 << 30)
# (TILE_H, TILE_W) shapes the autotuner chooses from; the kernels pick the matching tile table.
_TILE_SHAPES = ((1, 32), (2, 16), (2, 32), (4, 16))
_MAX_TILE_H = max(h for h, _ in _TILE_SHAPES)


def _configs() -> list[triton.Config]:
    return [
        triton.Config({"TILE_H": th, "TILE_W": tw, "BLOCK_ITER": bi}, num_warps=4, num_stages=s)
        for th, tw in _TILE_SHAPES
        for bi in (16, 32)
        for s in (2, 3)
    ]


_AUTOTUNE_KEY = ["NUM_ROWS", "NUM_POINTS", "HEAD_DIM", "KERNEL_H", "KERNEL_W", "DOT_PRECISION", "SHIFTED"]


@lru_cache(maxsize=32)
def _grid_tables(row_lengths: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, dict]:
    """Row starts, and for each tile shape the (first row, band) of every tile."""
    row_starts = ReducedGrid(row_lengths).row_starts.to(dtype=torch.int32, device=device)
    tiles = {}
    for tile_h, tile_w in _TILE_SHAPES:
        pairs = []
        for first_row in range(0, len(row_lengths), tile_h):
            longest = max(row_lengths[first_row : first_row + tile_h])
            pairs += [(first_row, band) for band in range(-(-longest // tile_w))]
        tiles[(tile_h, tile_w)] = torch.tensor(pairs, dtype=torch.int32, device=device)
    return row_starts, tiles


@lru_cache(maxsize=32)
def _shift_table(row_shifts: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """The shift of every row in half spacings; only read by kernels compiled with ``SHIFTED``."""
    return torch.tensor(row_shifts, dtype=torch.int32, device=device)


@triton.jit
def _tile_start(T1x32, T2x16, T2x32, T4x16, TILE_H: tl.constexpr, TILE_W: tl.constexpr):
    """First row and longitude band of the tile handled by this program."""
    if TILE_H == 1:
        table = T1x32
    elif TILE_H == 4:
        table = T4x16
    elif TILE_W == 16:
        table = T2x16
    else:
        table = T2x32
    pid = tl.program_id(0)
    return tl.load(table + 2 * pid), tl.load(table + 2 * pid + 1)


@triton.jit
def _spread(per_row, TILE_H: tl.constexpr, TILE_W: tl.constexpr):
    """Repeat a value per tile row for each of the ``TILE_W`` points of that row."""
    return tl.reshape(tl.broadcast_to(per_row[:, None], (TILE_H, TILE_W)), (TILE_H * TILE_W,))


@triton.jit
def _tile_points(ROW_START, first_row, band, NUM_ROWS: tl.constexpr, TILE_H: tl.constexpr, TILE_W: tl.constexpr):
    """Points of a tile, and the first position, end position and length of its segment in each of its rows.

    Rows past the grid get empty segments. Point values are laid out row by row, ``TILE_W`` per row.
    """
    rows_h = first_row + tl.arange(0, TILE_H)
    valid_h = rows_h < NUM_ROWS
    starts_h = tl.load(ROW_START + rows_h, mask=valid_h, other=0)
    lengths_h = tl.load(ROW_START + rows_h + 1, mask=valid_h, other=0) - starts_h
    longest = tl.max(lengths_h)
    # Band edges scaled to each row and rounded up, so neighbouring bands share no point.
    lo_h = (band * TILE_W * lengths_h + longest - 1) // longest
    hi_h = tl.minimum(((band + 1) * TILE_W * lengths_h + longest - 1) // longest, lengths_h)

    offs = tl.arange(0, TILE_H * TILE_W) % TILE_W
    rows = _spread(rows_h, TILE_H, TILE_W)
    pos = _spread(lo_h, TILE_H, TILE_W) + offs
    valid = pos < _spread(hi_h, TILE_H, TILE_W)
    lengths = tl.maximum(_spread(lengths_h, TILE_H, TILE_W), 1)
    tok = _spread(starts_h, TILE_H, TILE_W) + pos
    return rows, pos, valid, lengths, tok, lo_h, hi_h, lengths_h


@triton.jit
def _rows_info(ROW_START, first_row, num_rows, ROWS: tl.constexpr):
    """Start and length of the ``num_rows`` rows from ``first_row``, as vectors of ``ROWS`` entries."""
    idx = tl.arange(0, ROWS)
    valid = idx < num_rows
    starts = tl.load(ROW_START + first_row + idx, mask=valid, other=0)
    lengths = tl.load(ROW_START + first_row + idx + 1, mask=valid, other=1) - starts
    return idx, valid, starts, lengths


@triton.jit
def _pick(values, idx, i):
    """Entry ``i`` of a small vector, taken from registers."""
    return tl.sum(tl.where(idx == i, values, 0))


@triton.jit
def _matching(j, n, other_n):
    """Position in a row of ``other_n`` points nearest in longitude to position ``j`` of a row of ``n`` points."""
    return (2 * j * other_n + n) // (2 * n)


@triton.jit
def _matching_shifted(j, n, other_n, shift, other_shift):
    """Like :func:`_matching` for rows shifted by ``shift`` and ``other_shift`` half spacings (HEALPix rings)."""
    return ((2 * j + shift) * other_n - other_shift * n + n) // (2 * n)


@triton.jit
def _row_window_start(r, KERNEL_H: tl.constexpr, NUM_ROWS: tl.constexpr):
    """First row of the rows seen from row ``r``, shifted to stay inside the grid."""
    return tl.minimum(tl.maximum(r - KERNEL_H // 2, 0), NUM_ROWS - KERNEL_H)


@triton.jit
def _rows_seeing(r, KERNEL_H: tl.constexpr, NUM_ROWS: tl.constexpr):
    """First and last row of the queries whose row window includes row ``r``."""
    RADIUS: tl.constexpr = KERNEL_H // 2
    lo = tl.where(r - KERNEL_H + 1 <= 0, 0, r - RADIUS)
    hi = tl.where(r >= NUM_ROWS - KERNEL_H, NUM_ROWS - 1, tl.minimum(r + RADIUS, NUM_ROWS - 1))
    return lo, hi


@triton.jit
def _near_in_row(pos_a, pos_b, n, RADIUS_W: tl.constexpr):
    """True where two positions ``0 <= pos < n`` of a row are at most ``RADIUS_W`` apart, going round the globe.

    Comparisons stand in for the remainder, which is slow on GPUs: the difference is less than one
    turn, so at most one turn has to be added or taken away.
    """
    half = n // 2
    offset = pos_a - pos_b
    offset = tl.where(offset > half, offset - n, offset)
    offset = tl.where(offset < -half, offset + n, offset)
    return tl.abs(offset) <= RADIUS_W


@triton.jit
def _wrap(pos, n):
    """``pos`` taken round the globe into ``0 <= pos < n``, for ``-n <= pos < 2 * n``."""
    return tl.where(pos < 0, pos + n, tl.where(pos >= n, pos - n, pos))


@triton.jit
def _union(firsts, lasts, nonempty, lengths):
    """One stretch covering the stretches of the non-empty tile rows (last axis), capped at a whole row.

    A stretch covering a whole row starts at position 0, so windows wider than a row never make a
    stretch start more than one turn before position 0.
    """
    first = tl.min(tl.where(nonempty[None, :], firsts, _FAR), axis=1)
    last = tl.max(tl.where(nonempty[None, :], lasts, -_FAR), axis=1)
    whole = last - first + 1 >= lengths
    return tl.where(whole, 0, first), tl.where(whole, lengths, last - first + 1)


@triton.jit
def _key_rows(
    ROW_START,
    ROW_SHIFT,
    first_row,
    lo_h,
    hi_h,
    lengths_h,
    shift_h,
    NUM_ROWS: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    TILE_H: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
    SHIFTED: tl.constexpr,
):
    """The key rows a query tile reaches, with their shifts, the stretch of keys needed in each, and blocks per row."""
    RADIUS_W: tl.constexpr = KERNEL_W // 2
    last_row = tl.minimum(first_row + TILE_H, NUM_ROWS) - 1
    k_first_row = _row_window_start(first_row, KERNEL_H, NUM_ROWS)
    num_k_rows = _row_window_start(last_row, KERNEL_H, NUM_ROWS) + KERNEL_H - k_first_row
    idx, valid, k_starts, k_lengths = _rows_info(ROW_START, k_first_row, num_k_rows, KEY_ROWS)

    nonempty = hi_h > lo_h
    n = tl.maximum(lengths_h, 1)
    if SHIFTED:
        k_shifts = tl.load(ROW_SHIFT + k_first_row + idx, mask=valid, other=0)
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
    return k_first_row, num_k_rows, idx, k_starts, k_lengths, k_shifts, first, length, blocks


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _rg_fwd(
    Q,
    K,
    V,
    OUT,
    M,  # row maximum of the scaled scores of each query (in log2 units), shape (batch * heads, points)
    INV_L,  # inverse of the softmax sum of each query, same shape as M
    ROW_START,
    ROW_SHIFT,
    T1x32,
    T2x16,
    T2x32,
    T4x16,
    sm_scale,
    NUM_ROWS: tl.constexpr,
    NUM_POINTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    DOT_PRECISION: tl.constexpr,  # "ieee" for float32 inputs, so their products are not rounded to TF32
    SHIFTED: tl.constexpr,  # rows are shifted by half spacings (HEALPix); False compiles the shift handling away
    KEY_ROWS: tl.constexpr,  # KERNEL_H + largest TILE_H - 1, rounded up to a power of 2
    QUERY_ROWS: tl.constexpr,  # 2 * KERNEL_H + largest TILE_H - 2, rounded up to a power of 2
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes the output of one tile of queries, for one batch entry and head."""
    RADIUS_W: tl.constexpr = KERNEL_W // 2
    bh = tl.program_id(1).to(tl.int64)
    base = bh * NUM_POINTS * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    first_row, band = _tile_start(T1x32, T2x16, T2x32, T4x16, TILE_H, TILE_W)
    q_rows, q_pos, q_valid, q_n, q_tok, lo_h, hi_h, lengths_h = _tile_points(
        ROW_START, first_row, band, NUM_ROWS, TILE_H, TILE_W
    )
    q = tl.load(Q + base + q_tok[:, None] * HEAD_DIM + d[None, :], mask=q_valid[:, None], other=0.0)
    q_window = _row_window_start(q_rows, KERNEL_H, NUM_ROWS)
    if SHIFTED:
        tile_rows = first_row + tl.arange(0, TILE_H)
        shift_h = tl.load(ROW_SHIFT + tile_rows, mask=tile_rows < NUM_ROWS, other=0)
    else:
        shift_h = lengths_h * 0
    q_shift = _spread(shift_h, TILE_H, TILE_W)

    qk_scale = sm_scale * _RCP_LN2
    m_i = tl.full([TILE_H * TILE_W], _MASKED, dtype=tl.float32)
    l_i = tl.zeros([TILE_H * TILE_W], dtype=tl.float32)
    acc = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)

    # The key rows are looked up once, so the loop below reads them from registers. Rows and blocks
    # of keys are walked in one loop, every row getting as many blocks as the widest one needs.
    k_first_row, num_k_rows, idx, k_starts, k_lengths, k_shifts, firsts, lengths, blocks = _key_rows(
        ROW_START,
        ROW_SHIFT,
        first_row,
        lo_h,
        hi_h,
        lengths_h,
        shift_h,
        NUM_ROWS,
        KERNEL_H,
        KERNEL_W,
        TILE_H,
        KEY_ROWS,
        BLOCK_ITER,
        SHIFTED,
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
        k = tl.load(K + base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)

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
    tl.store(OUT + base + q_tok[:, None] * HEAD_DIM + d[None, :], acc.to(OUT.dtype.element_ty), mask=q_valid[:, None])
    tl.store(M + bh * NUM_POINTS + q_tok, m_i, mask=q_valid)
    tl.store(INV_L + bh * NUM_POINTS + q_tok, inv_l, mask=q_valid)


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _rg_bwd_dq(
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
    ROW_START,
    ROW_SHIFT,
    T1x32,
    T2x16,
    T2x32,
    T4x16,
    sm_scale,
    NUM_ROWS: tl.constexpr,
    NUM_POINTS: tl.constexpr,
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
    base = bh * NUM_POINTS * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    first_row, band = _tile_start(T1x32, T2x16, T2x32, T4x16, TILE_H, TILE_W)
    q_rows, q_pos, q_valid, q_n, q_tok, lo_h, hi_h, lengths_h = _tile_points(
        ROW_START, first_row, band, NUM_ROWS, TILE_H, TILE_W
    )
    q_ptrs = base + q_tok[:, None] * HEAD_DIM + d[None, :]
    q = tl.load(Q + q_ptrs, mask=q_valid[:, None], other=0.0)
    do = tl.load(DO + q_ptrs, mask=q_valid[:, None], other=0.0)
    out = tl.load(OUT + q_ptrs, mask=q_valid[:, None], other=0.0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(DELTA + bh * NUM_POINTS + q_tok, delta, mask=q_valid)
    m = tl.load(M + bh * NUM_POINTS + q_tok, mask=q_valid, other=0.0)
    inv_l = tl.load(INV_L + bh * NUM_POINTS + q_tok, mask=q_valid, other=0.0)
    q_window = _row_window_start(q_rows, KERNEL_H, NUM_ROWS)
    if SHIFTED:
        tile_rows = first_row + tl.arange(0, TILE_H)
        shift_h = tl.load(ROW_SHIFT + tile_rows, mask=tile_rows < NUM_ROWS, other=0)
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
    ds_sum = tl.zeros([TILE_H * TILE_W], dtype=tl.float32)
    p_sum = tl.zeros([TILE_H * TILE_W], dtype=tl.float32)

    k_first_row, num_k_rows, idx, k_starts, k_lengths, k_shifts, firsts, lengths, blocks = _key_rows(
        ROW_START,
        ROW_SHIFT,
        first_row,
        lo_h,
        hi_h,
        lengths_h,
        shift_h,
        NUM_ROWS,
        KERNEL_H,
        KERNEL_W,
        TILE_H,
        KEY_ROWS,
        BLOCK_ITER,
        SHIFTED,
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
        k = tl.load(K + base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + base + k_tok[:, None] * HEAD_DIM + d[None, :], mask=k_valid[:, None], other=0.0)

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
        ds_sum += tl.sum(ds.to(tl.float32), 1)
        p_sum += tl.sum(p.to(tl.float32), 1)
        dq = tl.dot(ds, k, dq, input_precision=DOT_PRECISION)
        p_k = tl.dot(p, k, p_k, input_precision=DOT_PRECISION)

    lam = tl.where(p_sum > 0, ds_sum / p_sum, 0.0)
    tl.store(LAM + bh * NUM_POINTS + q_tok, lam, mask=q_valid)
    dq = (dq - lam[:, None] * p_k) * sm_scale
    tl.store(DQ + q_ptrs, dq.to(DQ.dtype.element_ty), mask=q_valid[:, None])


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _rg_bwd_dkdv(
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
    ROW_START,
    ROW_SHIFT,
    T1x32,
    T2x16,
    T2x32,
    T4x16,
    sm_scale,
    NUM_ROWS: tl.constexpr,
    NUM_POINTS: tl.constexpr,
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
    base = bh * NUM_POINTS * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    first_row, band = _tile_start(T1x32, T2x16, T2x32, T4x16, TILE_H, TILE_W)
    k_rows, k_pos, k_valid, k_n, k_tok, lo_h, hi_h, lengths_h = _tile_points(
        ROW_START, first_row, band, NUM_ROWS, TILE_H, TILE_W
    )
    k_ptrs = base + k_tok[:, None] * HEAD_DIM + d[None, :]
    k = tl.load(K + k_ptrs, mask=k_valid[:, None], other=0.0)
    v = tl.load(V + k_ptrs, mask=k_valid[:, None], other=0.0)
    if SHIFTED:
        tile_rows = first_row + tl.arange(0, TILE_H)
        shift_h = tl.load(ROW_SHIFT + tile_rows, mask=tile_rows < NUM_ROWS, other=0)
    else:
        shift_h = lengths_h * 0

    qk_scale = sm_scale * _RCP_LN2
    dk = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([TILE_H * TILE_W, HEAD_DIM], dtype=tl.float32)

    # The query rows are looked up once, with the stretch of queries in each that may see the tile.
    last_row = tl.minimum(first_row + TILE_H, NUM_ROWS) - 1
    q_first_row, _ = _rows_seeing(first_row, KERNEL_H, NUM_ROWS)
    _, q_last_row = _rows_seeing(last_row, KERNEL_H, NUM_ROWS)
    num_q_rows = q_last_row - q_first_row + 1
    idx, row_valid, q_starts, q_lengths = _rows_info(ROW_START, q_first_row, num_q_rows, QUERY_ROWS)
    nonempty = hi_h > lo_h
    n_k = tl.maximum(lengths_h, 1)
    # Adding two turns keeps the divisions on positive numbers; the stretch is a little wider than
    # needed and the exact test is left to _near_in_row.
    firsts_h = ((lo_h[None, :] - RADIUS_W - 1 + 2 * n_k[None, :]) * q_lengths[:, None]) // n_k[None, :]
    lasts_h = ((hi_h[None, :] + RADIUS_W + 2 * n_k[None, :]) * q_lengths[:, None]) // n_k[None, :] + 1
    if SHIFTED:
        # Half-spacing shifts move the matching position by up to one more point.
        q_shifts = tl.load(ROW_SHIFT + q_first_row + idx, mask=row_valid, other=0)
        firsts_h = firsts_h - 1
    else:
        q_shifts = idx * 0
    firsts, lengths = _union(firsts_h - 2 * q_lengths[:, None], lasts_h - 2 * q_lengths[:, None], nonempty, q_lengths)
    blocks = tl.max(tl.where(row_valid, tl.cdiv(lengths, BLOCK_ITER), 0))

    offs = tl.arange(0, BLOCK_ITER)
    for it in range(0, num_q_rows * blocks):
        i = it // blocks
        q_row = q_first_row + i
        q_row_start = _pick(q_starts, idx, i)
        n = _pick(q_lengths, idx, i)
        first = _pick(firsts, idx, i)
        length = _pick(lengths, idx, i)
        t = (it % blocks) * BLOCK_ITER + offs
        q_valid = t < length
        q_pos = _wrap(first + t, n)
        q_tok = q_row_start + q_pos
        q = tl.load(Q + base + q_tok[:, None] * HEAD_DIM + d[None, :], mask=q_valid[:, None], other=0.0)
        do = tl.load(DO + base + q_tok[:, None] * HEAD_DIM + d[None, :], mask=q_valid[:, None], other=0.0)
        m = tl.load(M + bh * NUM_POINTS + q_tok, mask=q_valid, other=0.0)
        inv_l = tl.load(INV_L + bh * NUM_POINTS + q_tok, mask=q_valid, other=0.0)
        delta = tl.load(DELTA + bh * NUM_POINTS + q_tok, mask=q_valid, other=0.0)
        lam = tl.load(LAM + bh * NUM_POINTS + q_tok, mask=q_valid, other=0.0)

        q_window = _row_window_start(q_row, KERNEL_H, NUM_ROWS)
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
        p_t = p_t.to(do.dtype)
        dv = tl.dot(p_t, do, dv, input_precision=DOT_PRECISION)
        # dk from ds - lam * p, with lam from the dQ kernel, so the rounded ds rows sum to zero there too.
        dk = tl.dot(ds_t.to(q.dtype), q, dk, input_precision=DOT_PRECISION)
        dk = tl.dot(p_t, (-lam[:, None] * q).to(q.dtype), dk, input_precision=DOT_PRECISION)

    dk = dk * sm_scale
    tl.store(DK + k_ptrs, dk.to(DK.dtype.element_ty), mask=k_valid[:, None])
    tl.store(DV + k_ptrs, dv.to(DV.dtype.element_ty), mask=k_valid[:, None])


def _launch_grid(tiles: dict, batch_heads: int):
    def grid(meta):
        return (tiles[(meta["TILE_H"], meta["TILE_W"])].shape[0], batch_heads)

    return grid


class ReducedGridAttentionTriton(torch.autograd.Function):
    """Neighbourhood attention on a reduced grid, computed with Triton kernels.

    Inputs are ``(batch, heads, points, head_dim)`` with the points stored as in
    :class:`anemoi.models.layers.reduced_grid.ReducedGrid`. ``head_dim`` must be a power of two
    of at least 16.
    """

    @staticmethod
    def forward(ctx, q, k, v, grid, kernel_size, sm_scale):
        row_lengths = grid.row_lengths
        batch, heads, num_points, head_dim = q.shape
        assert num_points == sum(row_lengths), f"Expected {sum(row_lengths)} points, got {num_points}."
        assert k.shape == q.shape and v.shape == q.shape, "q, k and v must have the same shape."
        assert (
            head_dim >= 16 and head_dim & (head_dim - 1) == 0
        ), f"head_dim must be a power of 2 >= 16, got {head_dim}."
        assert kernel_size[0] <= len(row_lengths), "kernel_size[0] must not exceed the number of rows."

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)
        m = torch.empty((batch * heads, num_points), device=q.device, dtype=torch.float32)
        inv_l = torch.empty_like(m)
        row_starts, tiles = _grid_tables(tuple(row_lengths), q.device)
        sizes = dict(
            NUM_ROWS=len(row_lengths),
            NUM_POINTS=num_points,
            HEAD_DIM=head_dim,
            KERNEL_H=kernel_size[0],
            KERNEL_W=kernel_size[1],
            # Float32 inputs keep full precision in the matrix products; 16-bit inputs use tensor cores as usual.
            DOT_PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
            SHIFTED=grid.is_shifted,
            KEY_ROWS=triton.next_power_of_2(kernel_size[0] + _MAX_TILE_H - 1),
            QUERY_ROWS=triton.next_power_of_2(2 * kernel_size[0] + _MAX_TILE_H - 2),
        )
        tables = (row_starts, _shift_table(grid.shifts, q.device), *(tiles[shape] for shape in _TILE_SHAPES))

        _rg_fwd[_launch_grid(tiles, batch * heads)](q, k, v, o, m, inv_l, *tables, sm_scale, **sizes)
        ctx.save_for_backward(q, k, v, o, m, inv_l)
        ctx.grid = grid
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
        row_starts, tiles = _grid_tables(ctx.grid.row_lengths, q.device)
        tables = (row_starts, _shift_table(ctx.grid.shifts, q.device), *(tiles[shape] for shape in _TILE_SHAPES))
        grid = _launch_grid(tiles, batch * heads)

        # The dQ kernel also writes delta and lam, which the dK/dV kernel reads, so it runs first.
        _rg_bwd_dq[grid](q, k, v, o, do, dq, m, inv_l, delta, lam, *tables, ctx.sm_scale, **ctx.sizes)
        _rg_bwd_dkdv[grid](q, k, v, do, dk, dv, m, inv_l, delta, lam, *tables, ctx.sm_scale, **ctx.sizes)
        return dq, dk, dv, None, None, None


def reduced_grid_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grid: ReducedGrid,
    kernel_size: tuple[int, int],
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Neighbourhood attention on a reduced grid; see :class:`ReducedGridAttentionTriton`.

    ``sm_scale`` defaults to ``1 / sqrt(head_dim)``, as in ``scaled_dot_product_attention``.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return ReducedGridAttentionTriton.apply(q, k, v, grid, tuple(kernel_size), sm_scale)
