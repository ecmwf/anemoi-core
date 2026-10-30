# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton kernels for neighbourhood attention on the refined icosahedron.

Every query attends to the keys within ``radius`` hops (see
:mod:`anemoi.models.layers.icosahedral_grid`). Each program takes a ``TILE_I`` x ``TILE_J`` tile of
one diamond. Its keys lie in the tile grown by ``radius`` hops; that region is listed once per tile
shape and read through the diamond's lookup table, so keys in neighbouring diamonds come for free.

The queries near the twelve vertices (the exceptions) are left to block kernels: 16 queries near
the same vertex read the union of their keys, with a bit mask of who attends to whom.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

from anemoi.models.layers.icosahedral_grid import IcosahedralShard
from anemoi.models.layers.icosahedral_grid import hexagon_offsets

# Stands in for minus infinity on masked scores; finite so fully masked rows stay free of NaNs.
_MASKED = tl.constexpr(-1.0e30)
# 1 / ln(2), so the softmax can use exp2.
_RCP_LN2 = tl.constexpr(1.4426950408889634)
# (TILE_I, TILE_J) shapes the autotuner chooses from; the kernels read their part of the packed tables.
_TILE_SHAPES = ((4, 4), (2, 8), (4, 8), (8, 4), (8, 8))


def _configs() -> list[triton.Config]:
    return [
        triton.Config({"TILE_I": ti, "TILE_J": tj, "BLOCK_ITER": bi}, num_warps=4, num_stages=s)
        for ti, tj in _TILE_SHAPES
        for bi in (16, 32, 64)
        for s in (2, 3)
    ]


_AUTOTUNE_KEY = ["N", "RADIUS", "NUM_QUERIES", "NUM_KEYS", "HEAD_DIM", "DOT_PRECISION", "WIDE"]


# Points per block of exception queries or corrected keys; tl.dot needs at least 16 rows.
_ROW_BLOCK = 16


@dataclass(frozen=True, eq=False)
class _Blocks:
    """Points handled by the block kernels, in blocks of ``_ROW_BLOCK``, with their partner points.

    A block's partners are the union of its points' lists, in chunks of 32; one 32-bit word per chunk
    and slot marks the slot's partners.
    """

    num_blocks: int
    rows: torch.Tensor
    "Point in each slot, -1 for empty slots, shape ``(num_blocks * _ROW_BLOCK,)``."
    replaced: torch.Tensor
    "Corrected keys only: 1 if the slot replaces the tiled kernels' gradient, 0 if it adds to it."
    chunk_starts: torch.Tensor
    "First chunk of each block, and the end of the last, shape ``(num_blocks + 1,)``."
    columns: torch.Tensor
    "Partner points, 32 per chunk, -1 where a chunk is not full."
    words: torch.Tensor
    "Per chunk and slot, bit ``c`` set if the slot's point is paired with column ``c``."


def _blocks(
    rows: np.ndarray,
    clusters: np.ndarray,
    starts: np.ndarray,
    lists: np.ndarray,
    replaced: np.ndarray,
    device: torch.device,
) -> _Blocks:
    """Group ``rows`` with their CSR ``lists`` into blocks of ``_ROW_BLOCK`` rows near the same vertex.

    Such rows share most of their partners, so each block reads few partner points.
    """
    order = np.lexsort((rows, clusters))
    slots, flags, chunk_starts, columns, words = [], [], [0], [], []
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and end - begin < _ROW_BLOCK and clusters[order[end]] == clusters[order[begin]]:
            end += 1
        members = order[begin:end]
        partners = [lists[starts[r] : starts[r + 1]] for r in members]
        union = np.unique(np.concatenate(partners))
        num_chunks = -(-len(union) // 32)
        padded = np.full(num_chunks * 32, -1, dtype=np.int64)
        padded[: len(union)] = union
        bits = np.zeros((num_chunks, _ROW_BLOCK), dtype=np.int64)
        for slot, partner in enumerate(partners):
            where = np.searchsorted(union, partner)
            np.bitwise_or.at(bits[:, slot], where // 32, np.left_shift(1, where % 32))
        slots.append(np.concatenate([rows[members], np.full(_ROW_BLOCK - len(members), -1)]))
        flags.append(np.concatenate([replaced[members], np.zeros(_ROW_BLOCK - len(members), dtype=bool)]))
        chunk_starts.append(chunk_starts[-1] + num_chunks)
        columns.append(padded)
        # The words hold 32 bits; reading them as signed 32-bit numbers keeps every bit.
        words.append(bits.astype(np.uint32).view(np.int32).ravel())
        begin = end

    def as_int(parts: list, dtype=np.int32) -> torch.Tensor:
        array = np.concatenate(parts).astype(dtype) if parts else np.zeros(1, dtype=dtype)
        return torch.as_tensor(array, dtype=torch.int32, device=device)

    return _Blocks(
        num_blocks=len(slots),
        rows=as_int(slots),
        replaced=as_int(flags),
        chunk_starts=torch.as_tensor(np.array(chunk_starts), dtype=torch.int32, device=device),
        columns=as_int(columns),
        words=as_int(words),
    )


@dataclass(frozen=True, eq=False)
class _DeviceTables:
    """The tables of one shard on one device, as the kernels read them."""

    keys: torch.Tensor
    queries: torch.Tensor
    query_tiles: torch.Tensor
    query_tile_offsets: torch.Tensor
    query_tile_counts: dict
    key_tiles: torch.Tensor
    key_tile_offsets: torch.Tensor
    key_tile_counts: dict
    regions: torch.Tensor
    region_stride: int
    exception_blocks: _Blocks
    corrected_blocks: _Blocks


def _region(tile_i: int, tile_j: int, radius: int) -> np.ndarray:
    """Positions within ``radius`` hops of a tile, relative to its first position, row by row."""
    offsets = hexagon_offsets(radius)
    i, j = np.meshgrid(np.arange(tile_i), np.arange(tile_j), indexing="ij")
    reach = (np.stack([i.ravel(), j.ravel()], axis=1)[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    return np.unique(reach, axis=0)


def _region_size(tile_i: int, tile_j: int, radius: int) -> int:
    """Number of positions in :func:`_region`."""
    return tile_i * tile_j + 2 * radius * (tile_i + tile_j) + 3 * radius * radius - radius


def _tiles(occupied: np.ndarray, tile_i: int, tile_j: int, radius: int, n: int) -> np.ndarray:
    """(diamond, row, column) of the first position of every tile that holds a point."""
    own = occupied[:, radius : radius + n, radius : radius + n]
    rows, cols = -(-n // tile_i), -(-n // tile_j)
    padded = np.zeros((own.shape[0], rows * tile_i, cols * tile_j), dtype=bool)
    padded[:, :n, :n] = own
    busy = padded.reshape(own.shape[0], rows, tile_i, cols, tile_j).any(axis=(2, 4))
    d, r, c = np.nonzero(busy)
    return np.stack([d, radius + r * tile_i, radius + c * tile_j], axis=1).astype(np.int32)


def _pack(parts: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The parts one after the other, and where each part starts."""
    starts = np.cumsum([0] + [len(p) for p in parts[:-1]])
    packed = np.concatenate(parts) if sum(len(p) for p in parts) else np.zeros((1, 3), dtype=np.int32)
    return (
        torch.as_tensor(np.ascontiguousarray(packed), dtype=torch.int32, device=device),
        torch.as_tensor(starts, dtype=torch.int32, device=device),
    )


# Keyed by the shard, which lives as long as the model component that planned it.
@lru_cache(maxsize=64)
def _device_tables(shard: IcosahedralShard, device: torch.device) -> _DeviceTables:
    radius, n = shard.radius, shard.n
    query_parts = [_tiles(shard.queries >= 0, ti, tj, radius, n) for ti, tj in _TILE_SHAPES]
    key_parts = [_tiles(shard.keys >= 0, ti, tj, radius, n) for ti, tj in _TILE_SHAPES]
    regions = [_region(ti, tj, radius) for ti, tj in _TILE_SHAPES]
    stride = max(len(r) for r in regions)
    packed_regions = np.zeros((len(regions), stride, 2), dtype=np.int32)
    for shape, region in enumerate(regions):
        assert len(region) == _region_size(*_TILE_SHAPES[shape], radius)
        packed_regions[shape, : len(region)] = region
    query_tiles, query_tile_offsets = _pack(query_parts, device)
    key_tiles, key_tile_offsets = _pack(key_parts, device)

    return _DeviceTables(
        keys=torch.as_tensor(shard.keys.ravel(), dtype=torch.int32, device=device),
        queries=torch.as_tensor(shard.queries.ravel(), dtype=torch.int32, device=device),
        query_tiles=query_tiles,
        query_tile_offsets=query_tile_offsets,
        query_tile_counts={shape: len(p) for shape, p in zip(_TILE_SHAPES, query_parts)},
        key_tiles=key_tiles,
        key_tile_offsets=key_tile_offsets,
        key_tile_counts={shape: len(p) for shape, p in zip(_TILE_SHAPES, key_parts)},
        regions=torch.as_tensor(packed_regions.ravel(), dtype=torch.int32, device=device),
        region_stride=stride,
        exception_blocks=_blocks(
            shard.exception_queries,
            shard.exception_clusters,
            shard.exception_key_starts,
            shard.exception_key_lists,
            np.zeros(len(shard.exception_queries), dtype=bool),
            device,
        ),
        corrected_blocks=_blocks(
            shard.corrected_keys,
            shard.corrected_clusters,
            shard.corrected_query_starts,
            shard.corrected_query_lists,
            shard.replaced,
            device,
        ),
    )


@triton.jit
def _shape_index(TILE_I: tl.constexpr, TILE_J: tl.constexpr):
    """Position of the tile shape in ``_TILE_SHAPES``."""
    if TILE_I == 4 and TILE_J == 4:
        shape = 0
    elif TILE_I == 2 and TILE_J == 8:
        shape = 1
    elif TILE_I == 4 and TILE_J == 8:
        shape = 2
    elif TILE_I == 8 and TILE_J == 4:
        shape = 3
    elif TILE_I == 8 and TILE_J == 8:
        shape = 4
    return shape


@triton.jit
def _tile(TILES, TILE_OFFSETS, TILE_I: tl.constexpr, TILE_J: tl.constexpr):
    """Diamond, first row and first column of the tile handled by this program."""
    pid = tl.load(TILE_OFFSETS + _shape_index(TILE_I, TILE_J)) + tl.program_id(0)
    return tl.load(TILES + 3 * pid), tl.load(TILES + 3 * pid + 1), tl.load(TILES + 3 * pid + 2)


@triton.jit
def _base(bh, heads, stride_b, stride_h):
    """Offset of the first point of program ``bh`` (= batch entry * heads + head) in a strided tensor."""
    return (bh // heads) * stride_b + (bh % heads) * stride_h


@triton.jit
def _rows(tok, stride, WIDE: tl.constexpr):
    """Offsets of the points ``tok``, ``stride`` elements apart; in 64 bits if ``WIDE``."""
    if WIDE:
        return tok.to(tl.int64) * stride
    else:
        return tok * stride


@triton.jit
def _tile_points(TABLE, d, row0, col0, N: tl.constexpr, RADIUS: tl.constexpr, SIZE: tl.constexpr, TILE_I, TILE_J):
    """Table row, table column and point (-1 if none) of every position of a tile."""
    t = tl.arange(0, TILE_I * TILE_J)
    rows = row0 + t // TILE_J
    cols = col0 + t % TILE_J
    inside = (rows < RADIUS + N) & (cols < RADIUS + N)
    tok = tl.load(TABLE + d * SIZE * SIZE + rows * SIZE + cols, mask=inside, other=-1)
    return rows, cols, tok


@triton.jit
def _region_size_jit(TILE_I: tl.constexpr, TILE_J: tl.constexpr, RADIUS: tl.constexpr):
    """:func:`_region_size` at compile time."""
    return TILE_I * TILE_J + 2 * RADIUS * (TILE_I + TILE_J) + 3 * RADIUS * RADIUS - RADIUS


@triton.jit
def _region_points(
    TABLE,
    REGIONS,
    d,
    row0,
    col0,
    it,
    SIZE: tl.constexpr,
    RADIUS: tl.constexpr,
    REGION_STRIDE: tl.constexpr,
    TILE_I: tl.constexpr,
    TILE_J: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Table row, table column and point (-1 if none) of block ``it`` of the tile's region."""
    start = _shape_index(TILE_I, TILE_J) * REGION_STRIDE
    o = it * BLOCK_ITER + tl.arange(0, BLOCK_ITER)
    listed = o < _region_size_jit(TILE_I, TILE_J, RADIUS)
    rows = row0 + tl.load(REGIONS + 2 * (start + o), mask=listed, other=0)
    cols = col0 + tl.load(REGIONS + 2 * (start + o) + 1, mask=listed, other=0)
    inside = listed & (rows >= 0) & (rows < SIZE) & (cols >= 0) & (cols < SIZE)
    tok = tl.load(TABLE + d * SIZE * SIZE + rows * SIZE + cols, mask=inside, other=-1)
    return rows, cols, tok


@triton.jit
def _near(rows_a, cols_a, rows_b, cols_b, RADIUS: tl.constexpr):
    """True where two lattice positions (axis 0 and axis 1) are at most ``RADIUS`` hops apart."""
    di = rows_b[None, :] - rows_a[:, None]
    dj = cols_b[None, :] - cols_a[:, None]
    return (tl.abs(di) <= RADIUS) & (tl.abs(dj) <= RADIUS) & (tl.abs(di + dj) <= RADIUS)


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _ico_fwd(
    Q,
    K,
    V,
    OUT,
    M,  # row maximum of the unscaled scores q . k of each query, shape (batch * heads, queries)
    INV_L,  # inverse of the softmax sum of each query, same shape as M
    KEYS,
    QUERIES,
    TILES,
    TILE_OFFSETS,
    REGIONS,
    sm_scale,
    heads,
    stride_qb,  # strides of Q over batch, head and point; each point's head_dim values are contiguous
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    N: tl.constexpr,
    RADIUS: tl.constexpr,
    SIZE: tl.constexpr,
    REGION_STRIDE: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,  # "ieee" for float32 inputs, so their products are not rounded to TF32
    WIDE: tl.constexpr,  # point offsets need 64 bits
    TILE_I: tl.constexpr,
    TILE_J: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes the output of one tile of queries, for one batch entry and head."""
    bh = tl.program_id(1).to(tl.int64)
    q_base = _base(bh, heads, stride_qb, stride_qh)
    k_base = _base(bh, heads, stride_kb, stride_kh)
    v_base = _base(bh, heads, stride_vb, stride_vh)
    o_base = _base(bh, heads, stride_ob, stride_oh)
    d = tl.arange(0, HEAD_DIM)

    diamond, row0, col0 = _tile(TILES, TILE_OFFSETS, TILE_I, TILE_J)
    q_rows, q_cols, q_tok = _tile_points(QUERIES, diamond, row0, col0, N, RADIUS, SIZE, TILE_I, TILE_J)
    q_valid = q_tok >= 0
    q_tok = tl.where(q_valid, q_tok, 0)
    q = tl.load(Q + q_base + _rows(q_tok, stride_qn, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0)

    qk_scale = sm_scale * _RCP_LN2
    m_i = tl.full([TILE_I * TILE_J], _MASKED, dtype=tl.float32)
    l_i = tl.zeros([TILE_I * TILE_J], dtype=tl.float32)
    acc = tl.zeros([TILE_I * TILE_J, HEAD_DIM], dtype=tl.float32)

    for it in range(0, tl.cdiv(_region_size_jit(TILE_I, TILE_J, RADIUS), BLOCK_ITER)):
        k_rows, k_cols, k_tok = _region_points(
            KEYS, REGIONS, diamond, row0, col0, it, SIZE, RADIUS, REGION_STRIDE, TILE_I, TILE_J, BLOCK_ITER
        )
        k_valid = k_tok >= 0
        k_tok = tl.where(k_valid, k_tok, 0)
        k = tl.load(K + k_base + _rows(k_tok, stride_kn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + v_base + _rows(k_tok, stride_vn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)

        keep = _near(q_rows, q_cols, k_rows, k_cols, RADIUS) & k_valid[None, :]
        qk = tl.where(keep, tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION), _MASKED)

        # Shift by the row maximum before scaling, as in reduced_grid_attention (KohakuFA, bug 2).
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2((qk - m_new[:, None]) * qk_scale)
        alpha = tl.math.exp2((m_i - m_new) * qk_scale)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision=DOT_PRECISION)
        m_i = m_new

    inv_l = 1.0 / l_i
    acc = acc * inv_l[:, None]
    o_offs = o_base + _rows(q_tok, stride_on, WIDE)[:, None] + d[None, :]
    tl.store(OUT + o_offs, acc.to(OUT.dtype.element_ty), mask=q_valid[:, None])
    tl.store(M + bh * NUM_QUERIES + q_tok, m_i, mask=q_valid)
    tl.store(INV_L + bh * NUM_QUERIES + q_tok, inv_l, mask=q_valid)


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _ico_bwd_dq(
    Q,
    K,
    V,
    OUT,
    DO,
    DQ,
    M,
    INV_L,
    DELTA,  # written here: sum over the head dimension of OUT * DO for each query
    LAM,  # written here: sum of the rounded ds over sum of the rounded probabilities per query, see below
    KEYS,
    QUERIES,
    TILES,
    TILE_OFFSETS,
    REGIONS,
    sm_scale,
    heads,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_don,
    stride_ob,  # strides of OUT and DQ, which have the same layout
    stride_oh,
    stride_on,
    N: tl.constexpr,
    RADIUS: tl.constexpr,
    SIZE: tl.constexpr,
    REGION_STRIDE: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    WIDE: tl.constexpr,
    TILE_I: tl.constexpr,
    TILE_J: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes dQ for one tile of queries, visiting the same keys as the forward pass."""
    bh = tl.program_id(1).to(tl.int64)
    q_base = _base(bh, heads, stride_qb, stride_qh)
    k_base = _base(bh, heads, stride_kb, stride_kh)
    v_base = _base(bh, heads, stride_vb, stride_vh)
    do_base = _base(bh, heads, stride_dob, stride_doh)
    o_base = _base(bh, heads, stride_ob, stride_oh)
    d = tl.arange(0, HEAD_DIM)

    diamond, row0, col0 = _tile(TILES, TILE_OFFSETS, TILE_I, TILE_J)
    q_rows, q_cols, q_tok = _tile_points(QUERIES, diamond, row0, col0, N, RADIUS, SIZE, TILE_I, TILE_J)
    q_valid = q_tok >= 0
    q_tok = tl.where(q_valid, q_tok, 0)
    o_offs = o_base + _rows(q_tok, stride_on, WIDE)[:, None] + d[None, :]
    q = tl.load(Q + q_base + _rows(q_tok, stride_qn, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0)
    do = tl.load(DO + do_base + _rows(q_tok, stride_don, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0)
    out = tl.load(OUT + o_offs, mask=q_valid[:, None], other=0.0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(DELTA + bh * NUM_QUERIES + q_tok, delta, mask=q_valid)
    m = tl.load(M + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
    inv_l = tl.load(INV_L + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)

    qk_scale = sm_scale * _RCP_LN2
    dq = tl.zeros([TILE_I * TILE_J, HEAD_DIM], dtype=tl.float32)
    # Make the rounded ds sum to zero (GProj), as in reduced_grid_attention. See Chen et al.,
    # "Broken symmetry in BF16 attention", https://arxiv.org/abs/2609.34272.
    p_k = tl.zeros([TILE_I * TILE_J, HEAD_DIM], dtype=tl.float32)
    ds_sums = tl.zeros([TILE_I * TILE_J, BLOCK_ITER], dtype=tl.float32)
    p_sums = tl.zeros([TILE_I * TILE_J, BLOCK_ITER], dtype=tl.float32)

    for it in range(0, tl.cdiv(_region_size_jit(TILE_I, TILE_J, RADIUS), BLOCK_ITER)):
        k_rows, k_cols, k_tok = _region_points(
            KEYS, REGIONS, diamond, row0, col0, it, SIZE, RADIUS, REGION_STRIDE, TILE_I, TILE_J, BLOCK_ITER
        )
        k_valid = k_tok >= 0
        k_tok = tl.where(k_valid, k_tok, 0)
        k = tl.load(K + k_base + _rows(k_tok, stride_kn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + v_base + _rows(k_tok, stride_vn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)

        keep = _near(q_rows, q_cols, k_rows, k_cols, RADIUS) & k_valid[None, :] & q_valid[:, None]
        qk = tl.where(keep, tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION), _MASKED)
        p = tl.math.exp2((qk - m[:, None]) * qk_scale) * inv_l[:, None]

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
    tl.store(LAM + bh * NUM_QUERIES + q_tok, lam, mask=q_valid)
    dq = (dq - lam[:, None] * p_k) * sm_scale
    tl.store(DQ + o_offs, dq.to(DQ.dtype.element_ty), mask=q_valid[:, None])


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY)
@triton.jit
def _ico_bwd_dkdv(
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
    KEYS,
    QUERIES,
    TILES,
    TILE_OFFSETS,
    REGIONS,
    sm_scale,
    heads,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_don,
    stride_gb,  # strides of DK and DV, which have the same layout
    stride_gh,
    stride_gn,
    N: tl.constexpr,
    RADIUS: tl.constexpr,
    SIZE: tl.constexpr,
    REGION_STRIDE: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    WIDE: tl.constexpr,
    TILE_I: tl.constexpr,
    TILE_J: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
):
    """Each program computes dK and dV for one tile of keys, from the queries within ``radius`` hops.

    Hops are symmetric, so this is the same region as in the forward pass.
    """
    bh = tl.program_id(1).to(tl.int64)
    q_base = _base(bh, heads, stride_qb, stride_qh)
    k_base = _base(bh, heads, stride_kb, stride_kh)
    v_base = _base(bh, heads, stride_vb, stride_vh)
    do_base = _base(bh, heads, stride_dob, stride_doh)
    g_base = _base(bh, heads, stride_gb, stride_gh)
    d = tl.arange(0, HEAD_DIM)

    diamond, row0, col0 = _tile(TILES, TILE_OFFSETS, TILE_I, TILE_J)
    k_rows, k_cols, k_tok = _tile_points(KEYS, diamond, row0, col0, N, RADIUS, SIZE, TILE_I, TILE_J)
    k_valid = k_tok >= 0
    k_tok = tl.where(k_valid, k_tok, 0)
    k = tl.load(K + k_base + _rows(k_tok, stride_kn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
    v = tl.load(V + v_base + _rows(k_tok, stride_vn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)

    qk_scale = sm_scale * _RCP_LN2
    dk = tl.zeros([TILE_I * TILE_J, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([TILE_I * TILE_J, HEAD_DIM], dtype=tl.float32)

    for it in range(0, tl.cdiv(_region_size_jit(TILE_I, TILE_J, RADIUS), BLOCK_ITER)):
        q_rows, q_cols, q_tok = _region_points(
            QUERIES, REGIONS, diamond, row0, col0, it, SIZE, RADIUS, REGION_STRIDE, TILE_I, TILE_J, BLOCK_ITER
        )
        q_valid = q_tok >= 0
        q_tok = tl.where(q_valid, q_tok, 0)
        q = tl.load(Q + q_base + _rows(q_tok, stride_qn, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0)
        do = tl.load(
            DO + do_base + _rows(q_tok, stride_don, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0
        )
        m = tl.load(M + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
        inv_l = tl.load(INV_L + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
        delta = tl.load(DELTA + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
        lam = tl.load(LAM + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)

        # Scores laid out keys x queries.
        keep = _near(k_rows, k_cols, q_rows, q_cols, RADIUS) & q_valid[None, :] & k_valid[:, None]
        qk_t = tl.where(keep, tl.dot(k, tl.trans(q), input_precision=DOT_PRECISION), _MASKED)
        p_t = tl.math.exp2((qk_t - m[None, :]) * qk_scale) * inv_l[None, :]

        dp_t = tl.dot(v, tl.trans(do), input_precision=DOT_PRECISION)
        ds_t = p_t * (dp_t - delta[None, :])
        dv = tl.dot(p_t.to(do.dtype), do, dv, input_precision=DOT_PRECISION)
        # dk from ds - lam * p, with lam from the dQ kernel.
        dk = tl.dot(ds_t.to(q.dtype), q, dk, input_precision=DOT_PRECISION)
        dk = tl.dot((p_t * -lam[None, :]).to(q.dtype), q, dk, input_precision=DOT_PRECISION)

    dk = dk * sm_scale
    g_offs = g_base + _rows(k_tok, stride_gn, WIDE)[:, None] + d[None, :]
    tl.store(DK + g_offs, dk.to(DK.dtype.element_ty), mask=k_valid[:, None])
    tl.store(DV + g_offs, dv.to(DV.dtype.element_ty), mask=k_valid[:, None])


@triton.jit
def _block_rows(ROWS, block, ROW_BLOCK: tl.constexpr):
    """Slots of a block, the point in each (0 for empty slots) and which slots hold one."""
    slot = block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    tok = tl.load(ROWS + slot)
    return slot, tl.where(tok >= 0, tok, 0), tok >= 0


@triton.jit
def _block_chunk(COLUMNS, WORDS, chunk, ROW_BLOCK: tl.constexpr):
    """Points of a chunk of 32 columns (0 if empty), which columns are used, and which pairs attend."""
    c = tl.arange(0, 32)
    tok = tl.load(COLUMNS + chunk * 32 + c)
    words = tl.load(WORDS + chunk * ROW_BLOCK + tl.arange(0, ROW_BLOCK))
    pairs = ((words[:, None] >> c[None, :]) & 1) != 0
    return tl.where(tok >= 0, tok, 0), tok >= 0, pairs


@triton.jit
def _ico_block_fwd(
    Q,
    K,
    V,
    OUT,
    M,
    INV_L,
    ROWS,  # query position of the exception in each slot, -1 for empty slots
    CHUNK_STARTS,
    COLUMNS,  # key positions, 32 per chunk, -1 for empty columns
    WORDS,  # for each chunk and slot, a bit per column: whether the slot's query attends to it
    sm_scale,
    heads,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    NUM_QUERIES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    WIDE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    """Each program computes the output of one block of exception queries, for one batch entry and head."""
    block = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    k_base = _base(bh, heads, stride_kb, stride_kh)
    v_base = _base(bh, heads, stride_vb, stride_vh)
    d = tl.arange(0, HEAD_DIM)
    _, q_tok, q_valid = _block_rows(ROWS, block, ROW_BLOCK)
    q = tl.load(
        Q + _base(bh, heads, stride_qb, stride_qh) + _rows(q_tok, stride_qn, WIDE)[:, None] + d[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )

    qk_scale = sm_scale * _RCP_LN2
    m_i = tl.full([ROW_BLOCK], _MASKED, dtype=tl.float32)
    l_i = tl.zeros([ROW_BLOCK], dtype=tl.float32)
    acc = tl.zeros([ROW_BLOCK, HEAD_DIM], dtype=tl.float32)
    for chunk in range(tl.load(CHUNK_STARTS + block), tl.load(CHUNK_STARTS + block + 1)):
        k_tok, k_valid, pairs = _block_chunk(COLUMNS, WORDS, chunk, ROW_BLOCK)
        k = tl.load(K + k_base + _rows(k_tok, stride_kn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + v_base + _rows(k_tok, stride_vn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
        qk = tl.where(pairs, tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION), _MASKED)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2((qk - m_new[:, None]) * qk_scale)
        alpha = tl.math.exp2((m_i - m_new) * qk_scale)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision=DOT_PRECISION)
        m_i = m_new

    inv_l = 1.0 / l_i
    o_offs = _base(bh, heads, stride_ob, stride_oh) + _rows(q_tok, stride_on, WIDE)[:, None] + d[None, :]
    tl.store(OUT + o_offs, (acc * inv_l[:, None]).to(OUT.dtype.element_ty), mask=q_valid[:, None])
    tl.store(M + bh * NUM_QUERIES + q_tok, m_i, mask=q_valid)
    tl.store(INV_L + bh * NUM_QUERIES + q_tok, inv_l, mask=q_valid)


@triton.jit
def _ico_block_dq(
    Q,
    K,
    V,
    OUT,
    DO,
    DQ,
    M,
    INV_L,
    DELTA,  # written here for the exceptions
    LAM,  # written here for the exceptions, as in the tiled dQ kernel
    ROWS,
    CHUNK_STARTS,
    COLUMNS,
    WORDS,
    sm_scale,
    heads,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_don,
    stride_ob,  # strides of OUT and DQ, which have the same layout
    stride_oh,
    stride_on,
    NUM_QUERIES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    WIDE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    """Each program computes dQ of one block of exception queries, as the tiled dQ kernel does for a tile."""
    block = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    k_base = _base(bh, heads, stride_kb, stride_kh)
    v_base = _base(bh, heads, stride_vb, stride_vh)
    d = tl.arange(0, HEAD_DIM)
    _, q_tok, q_valid = _block_rows(ROWS, block, ROW_BLOCK)
    o_offs = _base(bh, heads, stride_ob, stride_oh) + _rows(q_tok, stride_on, WIDE)[:, None] + d[None, :]
    q = tl.load(
        Q + _base(bh, heads, stride_qb, stride_qh) + _rows(q_tok, stride_qn, WIDE)[:, None] + d[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + _base(bh, heads, stride_dob, stride_doh) + _rows(q_tok, stride_don, WIDE)[:, None] + d[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    out = tl.load(OUT + o_offs, mask=q_valid[:, None], other=0.0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(DELTA + bh * NUM_QUERIES + q_tok, delta, mask=q_valid)
    m = tl.load(M + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
    inv_l = tl.load(INV_L + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)

    qk_scale = sm_scale * _RCP_LN2
    dq = tl.zeros([ROW_BLOCK, HEAD_DIM], dtype=tl.float32)
    p_k = tl.zeros([ROW_BLOCK, HEAD_DIM], dtype=tl.float32)
    ds_sums = tl.zeros([ROW_BLOCK, 32], dtype=tl.float32)
    p_sums = tl.zeros([ROW_BLOCK, 32], dtype=tl.float32)
    for chunk in range(tl.load(CHUNK_STARTS + block), tl.load(CHUNK_STARTS + block + 1)):
        k_tok, k_valid, pairs = _block_chunk(COLUMNS, WORDS, chunk, ROW_BLOCK)
        k = tl.load(K + k_base + _rows(k_tok, stride_kn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
        v = tl.load(V + v_base + _rows(k_tok, stride_vn, WIDE)[:, None] + d[None, :], mask=k_valid[:, None], other=0.0)
        keep = pairs & q_valid[:, None]
        qk = tl.where(keep, tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION), _MASKED)
        p = tl.math.exp2((qk - m[:, None]) * qk_scale) * inv_l[:, None]
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
    tl.store(LAM + bh * NUM_QUERIES + q_tok, lam, mask=q_valid)
    dq = (dq - lam[:, None] * p_k) * sm_scale
    tl.store(DQ + o_offs, dq.to(DQ.dtype.element_ty), mask=q_valid[:, None])


@triton.jit
def _ico_block_dkdv(
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
    ROWS,  # key position of the corrected key in each slot, -1 for empty slots
    REPLACED,  # 1 where the slot's key takes all its gradient from here, 0 where it is added to the tiled kernel's
    CHUNK_STARTS,
    COLUMNS,  # query positions, 32 per chunk, -1 for empty columns
    WORDS,
    sm_scale,
    heads,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_don,
    stride_gb,  # strides of DK and DV, which have the same layout
    stride_gh,
    stride_gn,
    NUM_QUERIES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    WIDE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    """Each program computes dK and dV of one block of corrected keys from the queries of its columns."""
    block = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    q_base = _base(bh, heads, stride_qb, stride_qh)
    do_base = _base(bh, heads, stride_dob, stride_doh)
    d = tl.arange(0, HEAD_DIM)
    slot, k_tok, k_valid = _block_rows(ROWS, block, ROW_BLOCK)
    k = tl.load(
        K + _base(bh, heads, stride_kb, stride_kh) + _rows(k_tok, stride_kn, WIDE)[:, None] + d[None, :],
        mask=k_valid[:, None],
        other=0.0,
    )
    v = tl.load(
        V + _base(bh, heads, stride_vb, stride_vh) + _rows(k_tok, stride_vn, WIDE)[:, None] + d[None, :],
        mask=k_valid[:, None],
        other=0.0,
    )

    qk_scale = sm_scale * _RCP_LN2
    dk = tl.zeros([ROW_BLOCK, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([ROW_BLOCK, HEAD_DIM], dtype=tl.float32)
    for chunk in range(tl.load(CHUNK_STARTS + block), tl.load(CHUNK_STARTS + block + 1)):
        q_tok, q_valid, pairs = _block_chunk(COLUMNS, WORDS, chunk, ROW_BLOCK)
        q = tl.load(Q + q_base + _rows(q_tok, stride_qn, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0)
        do = tl.load(
            DO + do_base + _rows(q_tok, stride_don, WIDE)[:, None] + d[None, :], mask=q_valid[:, None], other=0.0
        )
        m = tl.load(M + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
        inv_l = tl.load(INV_L + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
        delta = tl.load(DELTA + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)
        lam = tl.load(LAM + bh * NUM_QUERIES + q_tok, mask=q_valid, other=0.0)

        # Scores laid out keys x queries.
        keep = pairs & k_valid[:, None]
        qk_t = tl.where(keep, tl.dot(k, tl.trans(q), input_precision=DOT_PRECISION), _MASKED)
        p_t = tl.math.exp2((qk_t - m[None, :]) * qk_scale) * inv_l[None, :]
        dp_t = tl.dot(v, tl.trans(do), input_precision=DOT_PRECISION)
        ds_t = p_t * (dp_t - delta[None, :])
        dv = tl.dot(p_t.to(do.dtype), do, dv, input_precision=DOT_PRECISION)
        # dk from ds - lam * p, as the tiled dK/dV kernel uses for the same queries.
        dk = tl.dot(ds_t.to(q.dtype), q, dk, input_precision=DOT_PRECISION)
        dk = tl.dot((p_t * -lam[None, :]).to(q.dtype), q, dk, input_precision=DOT_PRECISION)

    g_offs = _base(bh, heads, stride_gb, stride_gh) + _rows(k_tok, stride_gn, WIDE)[:, None] + d[None, :]
    adds = k_valid & (tl.load(REPLACED + slot) == 0)
    dk = dk * sm_scale + tl.load(DK + g_offs, mask=adds[:, None], other=0.0).to(tl.float32)
    dv = dv + tl.load(DV + g_offs, mask=adds[:, None], other=0.0).to(tl.float32)
    tl.store(DK + g_offs, dk.to(DK.dtype.element_ty), mask=k_valid[:, None])
    tl.store(DV + g_offs, dv.to(DV.dtype.element_ty), mask=k_valid[:, None])


def _launch_grid(counts: dict, batch_heads: int):
    def grid(meta):
        return (counts[(meta["TILE_I"], meta["TILE_J"])], batch_heads)

    return grid


def _strides(t: torch.Tensor) -> tuple[int, int, int]:
    """Strides of a ``(batch, heads, points, head_dim)`` tensor over batch, head and point."""
    return t.stride(0), t.stride(1), t.stride(2)


def _rows_readable(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the kernels read it: any layout whose head_dim values of each point are contiguous."""
    return t if t.stride(-1) == 1 else t.contiguous()


def _point_major(batch: int, heads: int, points: int, head_dim: int, like: torch.Tensor) -> torch.Tensor:
    """Empty ``(batch, heads, points, head_dim)`` tensor in the model's point-major layout, so no copy is needed later."""
    return torch.empty((batch, points, heads, head_dim), dtype=like.dtype, device=like.device).transpose(1, 2)


def _needs_wide(*tensors: torch.Tensor) -> bool:
    """Whether the offset of some point from the first point of its batch entry and head reaches 2**31."""
    return any((t.shape[2] - 1) * t.stride(2) + t.shape[3] >= 2**31 for t in tensors)


def _dot_precision(dtype: torch.dtype) -> str:
    """How the kernels' matrix products treat float32 inputs, following torch.set_float32_matmul_precision."""
    if dtype == torch.float32 and torch.get_float32_matmul_precision() == "highest":
        return "ieee"
    return "tf32"


class IcosahedralAttentionTriton(torch.autograd.Function):
    """Neighbourhood attention on the refined icosahedron, for the queries and keys of one shard.

    ``q`` has ``shard.num_queries`` points, ``k`` and ``v`` have ``shard.num_keys``. Inputs may use any
    layout where each point's head_dim values are contiguous; outputs are point-major.
    """

    @staticmethod
    def forward(ctx, q, k, v, shard, sm_scale):
        batch, heads, num_queries, head_dim = q.shape
        assert num_queries == shard.num_queries, f"Expected {shard.num_queries} queries, got {num_queries}."
        assert k.shape[2] == shard.num_keys, f"Expected {shard.num_keys} keys, got {k.shape[2]}."
        assert v.shape == k.shape and k.shape[:2] == q.shape[:2] and k.shape[3] == head_dim
        assert (
            head_dim >= 16 and head_dim & (head_dim - 1) == 0
        ), f"head_dim must be a power of 2 >= 16, got {head_dim}."

        q, k, v = (_rows_readable(t) for t in (q, k, v))
        tables = _device_tables(shard, q.device)
        o = _point_major(batch, heads, num_queries, head_dim, q)
        m = torch.empty((batch * heads, num_queries), device=q.device, dtype=torch.float32)
        inv_l = torch.empty_like(m)
        sizes = dict(
            N=shard.n,
            RADIUS=shard.radius,
            SIZE=shard.n + 2 * shard.radius,
            REGION_STRIDE=tables.region_stride,
            NUM_QUERIES=num_queries,
            NUM_KEYS=shard.num_keys,
            HEAD_DIM=head_dim,
            DOT_PRECISION=_dot_precision(q.dtype),
        )
        wide = _needs_wide(q, k, v, o)
        _ico_fwd[_launch_grid(tables.query_tile_counts, batch * heads)](
            q,
            k,
            v,
            o,
            m,
            inv_l,
            tables.keys,
            tables.queries,
            tables.query_tiles,
            tables.query_tile_offsets,
            tables.regions,
            sm_scale,
            heads,
            *_strides(q),
            *_strides(k),
            *_strides(v),
            *_strides(o),
            WIDE=wide,
            **sizes,
        )
        blocks = dict(
            NUM_QUERIES=num_queries, HEAD_DIM=head_dim, DOT_PRECISION=sizes["DOT_PRECISION"], ROW_BLOCK=_ROW_BLOCK
        )
        exceptions = tables.exception_blocks
        if exceptions.num_blocks:
            _ico_block_fwd[(exceptions.num_blocks, batch * heads)](
                q,
                k,
                v,
                o,
                m,
                inv_l,
                exceptions.rows,
                exceptions.chunk_starts,
                exceptions.columns,
                exceptions.words,
                sm_scale,
                heads,
                *_strides(q),
                *_strides(k),
                *_strides(v),
                *_strides(o),
                WIDE=wide,
                **blocks,
            )

        ctx.save_for_backward(q, k, v, o, m, inv_l)
        ctx.shard = shard
        ctx.sizes = sizes
        ctx.blocks = blocks
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, inv_l = ctx.saved_tensors
        batch, heads, _, _ = q.shape
        sm_scale = ctx.sm_scale
        tables = _device_tables(ctx.shard, q.device)

        do = _rows_readable(do)
        delta, lam = torch.empty_like(m), torch.empty_like(m)
        # Every key is written once: by the tile that holds it, or, for the exceptions (among them
        # the two points outside the diamonds), by the block kernel.
        dq = _point_major(*q.shape, q)
        dk, dv = _point_major(*k.shape, k), _point_major(*v.shape, v)
        inputs = (*_strides(q), *_strides(k), *_strides(v), *_strides(do))
        wide = _needs_wide(q, k, v, o, do, dq, dk, dv)

        # The dQ kernel also writes delta and lam, which the dK/dV kernel reads, so it runs first.
        _ico_bwd_dq[_launch_grid(tables.query_tile_counts, batch * heads)](
            q,
            k,
            v,
            o,
            do,
            dq,
            m,
            inv_l,
            delta,
            lam,
            tables.keys,
            tables.queries,
            tables.query_tiles,
            tables.query_tile_offsets,
            tables.regions,
            sm_scale,
            heads,
            *inputs,
            *_strides(dq),
            WIDE=wide,
            **ctx.sizes,
        )

        exceptions = tables.exception_blocks
        if exceptions.num_blocks:
            _ico_block_dq[(exceptions.num_blocks, batch * heads)](
                q,
                k,
                v,
                o,
                do,
                dq,
                m,
                inv_l,
                delta,
                lam,
                exceptions.rows,
                exceptions.chunk_starts,
                exceptions.columns,
                exceptions.words,
                sm_scale,
                heads,
                *inputs,
                *_strides(dq),
                WIDE=wide,
                **ctx.blocks,
            )

        _ico_bwd_dkdv[_launch_grid(tables.key_tile_counts, batch * heads)](
            q,
            k,
            v,
            do,
            dk,
            dv,
            m,
            inv_l,
            delta,
            lam,
            tables.keys,
            tables.queries,
            tables.key_tiles,
            tables.key_tile_offsets,
            tables.regions,
            sm_scale,
            heads,
            *inputs,
            *_strides(dk),
            WIDE=wide,
            **ctx.sizes,
        )

        # Exception keys get all their gradient here; keys next to an exception query get that
        # query's part added. Runs after both dQ kernels, which write delta and lam.
        corrected = tables.corrected_blocks
        if corrected.num_blocks:
            _ico_block_dkdv[(corrected.num_blocks, batch * heads)](
                q,
                k,
                v,
                do,
                dk,
                dv,
                m,
                inv_l,
                delta,
                lam,
                corrected.rows,
                corrected.replaced,
                corrected.chunk_starts,
                corrected.columns,
                corrected.words,
                sm_scale,
                heads,
                *inputs,
                *_strides(dk),
                WIDE=wide,
                **ctx.blocks,
            )

        return dq, dk, dv, None, None


def icosahedral_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shard: IcosahedralShard,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Neighbourhood attention on the refined icosahedron; see :class:`IcosahedralAttentionTriton`.

    ``sm_scale`` defaults to ``1 / sqrt(head_dim)``, as in ``scaled_dot_product_attention``.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return IcosahedralAttentionTriton.apply(q, k, v, shard, sm_scale)
