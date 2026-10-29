# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton kernels for projecting a field from one node set onto another with a table of weights.

Every target point ``t`` gets the weighted sum of the values at its source points:

    y[t] = sum_k w[t, k] * x[idx[t, k]]

with ``idx`` and ``w`` of shape ``(target points, longest row)`` and the number of entries of
each row, as given by :meth:`anemoi.models.layers.graph_provider.ProjectionGraphProvider.get_table`.

The forward kernel takes a block of target points and a block of channels of one field, and goes
through the target points' source points several at a time: it gathers a tile of source values,
weights them and adds them up, so that many loads are in flight at once, and writes the sums
straight into the result.

The backward kernel takes a block of target points of one field and, for each tile of source
points, goes through all the channels, sending each target point's weighted gradient back to its
source points with atomic additions, since several target points share a source point. The
additions arrive in whatever order the GPU runs them, so repeated runs can differ in the last bits.

Both kernels read only the stored entries of each row: a block of target points goes as far as
its longest row, and the slots past each row's length are masked out. The padding at the end of
short rows is never read or written, so rows of uneven length cost about what their entries do,
and a NaN or infinity in the field reaches only the target points whose rows hold it, as with
the sparse matrix product.

Sums are kept in float32, or in float64 for float64 fields. The block sizes are fixed, with one
forward tile for tables of long rows and one for narrow tables (chosen on an A100), so each kernel
compiles once per field dtype, row tile width and, for the forward kernel, channel block size.
"""

import torch
import triton
import triton.language as tl

from anemoi.models.layers.projection_backend import accumulation_dtype
from anemoi.models.triton.utils import torch_dtype_to_triton

# Forward tiles as (target points, table slots, warps), chosen on an A100: tables of rows with
# about 128 entries take the first, tables at most _SHORT_ROWS_WIDTH wide (interpolation from a
# coarser grid, a few entries per row) the second.
_FWD_TILE = (8, 8, 2)
_FWD_SHORT_ROWS_TILE = (16, 4, 2)
_SHORT_ROWS_WIDTH = 32
_FWD_MAX_CHANNELS = 128
_BWD_POINTS, _BWD_MAX_NEIGHBOURS, _BWD_CHANNELS, _BWD_WARPS = 8, 4, 32, 4


def _block(size: int, largest: int, smallest: int = 1) -> int:
    return min(largest, max(smallest, triton.next_power_of_2(size)))


@triton.jit
def _project_fwd(
    x_ptr,
    out_ptr,
    neighbours_ptr,
    weights_ptr,
    lengths_ptr,
    num_sources,
    num_targets,
    channels,
    num_neighbours,
    ACC: tl.constexpr,
    BLOCK_POINTS: tl.constexpr,
    BLOCK_NEIGHBOURS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    points = (tl.program_id(0) * BLOCK_POINTS + tl.arange(0, BLOCK_POINTS)).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    field = tl.program_id(2).to(tl.int64)
    point_mask = points < num_targets
    col_mask = cols < channels
    lengths = tl.load(lengths_ptr + points, mask=point_mask, other=0)

    acc = tl.zeros([BLOCK_POINTS, BLOCK_CHANNELS], dtype=ACC)
    # The block goes as far as its longest row; the slots past each row's length are masked out.
    for k0 in range(0, tl.max(lengths, axis=0), BLOCK_NEIGHBOURS):
        ks = k0 + tl.arange(0, BLOCK_NEIGHBOURS)
        tile_mask = point_mask[:, None] & (ks[None, :] < lengths[:, None])
        table = points[:, None] * num_neighbours + ks[None, :]
        neighbour = tl.load(neighbours_ptr + table, mask=tile_mask, other=0).to(tl.int64)
        weight = tl.load(weights_ptr + table, mask=tile_mask, other=0.0).to(ACC)
        values = tl.load(
            x_ptr + (field * num_sources + neighbour[:, :, None]) * channels + cols[None, None, :],
            mask=tile_mask[:, :, None] & col_mask[None, None, :],
            other=0.0,
        )
        acc += tl.sum(weight[:, :, None] * values.to(ACC), axis=1)

    tl.store(
        out_ptr + (field * num_targets + points[:, None]) * channels + cols[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=point_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _project_bwd(
    grad_out_ptr,
    grad_x_ptr,
    neighbours_ptr,
    weights_ptr,
    lengths_ptr,
    num_sources,
    num_targets,
    channels,
    num_neighbours,
    ACC: tl.constexpr,
    BLOCK_POINTS: tl.constexpr,
    BLOCK_NEIGHBOURS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    points = (tl.program_id(0) * BLOCK_POINTS + tl.arange(0, BLOCK_POINTS)).to(tl.int64)
    field = tl.program_id(1).to(tl.int64)
    point_mask = points < num_targets
    lengths = tl.load(lengths_ptr + points, mask=point_mask, other=0)

    for k0 in range(0, tl.max(lengths, axis=0), BLOCK_NEIGHBOURS):
        ks = k0 + tl.arange(0, BLOCK_NEIGHBOURS)
        tile_mask = point_mask[:, None] & (ks[None, :] < lengths[:, None])
        table = points[:, None] * num_neighbours + ks[None, :]
        neighbour = tl.load(neighbours_ptr + table, mask=tile_mask, other=0).to(tl.int64)
        weight = tl.load(weights_ptr + table, mask=tile_mask, other=0.0).to(ACC)
        for c0 in range(0, channels, BLOCK_CHANNELS):
            cols = c0 + tl.arange(0, BLOCK_CHANNELS)
            col_mask = cols < channels
            grad = tl.load(
                grad_out_ptr + (field * num_targets + points[:, None]) * channels + cols[None, :],
                mask=point_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(ACC)
            tl.atomic_add(
                grad_x_ptr + (field * num_sources + neighbour[:, :, None]) * channels + cols[None, None, :],
                weight[:, :, None] * grad[:, None, :],
                mask=tile_mask[:, :, None] & col_mask[None, None, :],
                sem="relaxed",
            )


class SparseProjectionTriton(torch.autograd.Function):
    """Projection of a ``(batch, source points, channels)`` field with the Triton kernels."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, neighbours: torch.Tensor, weights: torch.Tensor, row_lengths: torch.Tensor):
        x = x.contiguous()
        neighbours = neighbours.contiguous()
        weights = weights.contiguous()
        row_lengths = row_lengths.contiguous()
        batch, num_sources, channels = x.shape
        num_targets, num_neighbours = neighbours.shape
        out = torch.empty(batch, num_targets, channels, dtype=x.dtype, device=x.device)
        block_points, block_slots, warps = _FWD_SHORT_ROWS_TILE if num_neighbours <= _SHORT_ROWS_WIDTH else _FWD_TILE
        block_channels = _block(channels, _FWD_MAX_CHANNELS, smallest=16)
        grid = (triton.cdiv(num_targets, block_points), triton.cdiv(channels, block_channels), batch)
        # The kernels run on fields with at least one channel; with the grid split by channels, a
        # process can get an empty share.
        if out.numel() > 0:
            _project_fwd[grid](
                x,
                out,
                neighbours,
                weights,
                row_lengths,
                num_sources,
                num_targets,
                channels,
                num_neighbours,
                ACC=torch_dtype_to_triton(accumulation_dtype(x.dtype)),
                BLOCK_POINTS=block_points,
                BLOCK_NEIGHBOURS=_block(num_neighbours, block_slots),
                BLOCK_CHANNELS=block_channels,
                num_warps=warps,
            )
        ctx.save_for_backward(neighbours, weights, row_lengths)
        ctx.num_sources = num_sources
        ctx.dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        neighbours, weights, row_lengths = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        batch, num_targets, channels = grad_out.shape
        acc_dtype = accumulation_dtype(ctx.dtype)
        grad_x = torch.zeros(batch, ctx.num_sources, channels, dtype=acc_dtype, device=grad_out.device)
        grid = (triton.cdiv(num_targets, _BWD_POINTS), batch)
        # The kernels run on fields with at least one channel; with the grid split by channels, a
        # process can get an empty share.
        if grad_x.numel() > 0:
            _project_bwd[grid](
                grad_out,
                grad_x,
                neighbours,
                weights,
                row_lengths,
                ctx.num_sources,
                num_targets,
                channels,
                neighbours.shape[1],
                ACC=torch_dtype_to_triton(acc_dtype),
                BLOCK_POINTS=_BWD_POINTS,
                BLOCK_NEIGHBOURS=_block(neighbours.shape[1], _BWD_MAX_NEIGHBOURS),
                BLOCK_CHANNELS=_BWD_CHANNELS,
                num_warps=_BWD_WARPS,
            )
        return grad_x.to(ctx.dtype), None, None, None


def project_triton(
    x: torch.Tensor, neighbours: torch.Tensor, weights: torch.Tensor, row_lengths: torch.Tensor
) -> torch.Tensor:
    """Projection of a ``(batch, source points, channels)`` field with the Triton kernels.

    See :class:`anemoi.models.layers.sparse_projector.SparseProjector` for the backend that uses it.
    """
    return SparseProjectionTriton.apply(x, neighbours, weights, row_lengths)
