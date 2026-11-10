# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton kernels for Gaussian smoothing over a neighbour list.

See :mod:`anemoi.models.layers.gaussian_smoothing` for the maths. The forward kernel computes
the weights of a tile of neighbours from the unit vectors and accumulates the weighted values;
it also stores the per-point weight sums for the backward pass. The backward kernel recomputes
the weights and scatters the gradients with atomic adds, so results can differ in the last bits
between runs.

Accumulation is in float32 (float64 for float64 inputs). Block sizes are fixed (tuned on an A100
at O320), so there is no autotuning.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from anemoi.models.layers.projection_backend import accumulation_dtype
from anemoi.models.triton.utils import torch_dtype_to_triton

_FWD_POINTS, _FWD_NEIGHBOURS, _FWD_MAX_CHANNELS, _FWD_WARPS = 8, 16, 128, 4
_BWD_POINTS, _BWD_NEIGHBOURS, _BWD_CHANNELS, _BWD_WARPS = 8, 4, 32, 4


def _block_channels(channels: int, largest: int) -> int:
    return min(largest, max(16, triton.next_power_of_2(channels)))


@triton.jit
def _load_unit_vectors(unit_vectors_ptr, points, mask, ACC: tl.constexpr):
    """x, y and z of the unit vectors of ``points``."""
    ux = tl.load(unit_vectors_ptr + points * 3 + 0, mask=mask, other=0.0).to(ACC)
    uy = tl.load(unit_vectors_ptr + points * 3 + 1, mask=mask, other=0.0).to(ACC)
    uz = tl.load(unit_vectors_ptr + points * 3 + 2, mask=mask, other=0.0).to(ACC)
    return ux, uy, uz


@triton.jit
def _neighbour_weights(unit_vectors_ptr, neighbour, tile_mask, px, py, pz, scale, ACC: tl.constexpr):
    """Unnormalised Gaussian weights of a tile of neighbours."""
    nx, ny, nz = _load_unit_vectors(unit_vectors_ptr, neighbour, tile_mask, ACC)
    dx = nx - px[:, None]
    dy = ny - py[:, None]
    dz = nz - pz[:, None]
    chord = tl.sqrt(dx * dx + dy * dy + dz * dz)
    angle = 2.0 * libdevice.asin(tl.minimum(chord * 0.5, 1.0))
    return tl.where(tile_mask, tl.exp(-angle * angle * scale), 0.0)


@triton.jit
def _gaussian_smooth_fwd(
    x_ptr,
    out_ptr,
    totals_ptr,
    neighbours_ptr,
    unit_vectors_ptr,
    scale_ptr,
    num_points,
    channels,
    num_neighbours,
    ACC: tl.constexpr,
    BLOCK_POINTS: tl.constexpr,
    BLOCK_NEIGHBOURS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    points = (tl.program_id(0) * BLOCK_POINTS + tl.arange(0, BLOCK_POINTS)).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    field = tl.program_id(2).to(tl.int64) * num_points * channels
    point_mask = points < num_points
    col_mask = cols < channels
    scale = tl.load(scale_ptr).to(ACC)
    px, py, pz = _load_unit_vectors(unit_vectors_ptr, points, point_mask, ACC)

    acc = tl.zeros([BLOCK_POINTS, BLOCK_CHANNELS], dtype=ACC)
    total = tl.zeros([BLOCK_POINTS], dtype=ACC)
    for k0 in range(0, num_neighbours, BLOCK_NEIGHBOURS):
        ks = k0 + tl.arange(0, BLOCK_NEIGHBOURS)
        tile_mask = point_mask[:, None] & (ks < num_neighbours)[None, :]
        neighbour = tl.load(
            neighbours_ptr + points[:, None] * num_neighbours + ks[None, :], mask=tile_mask, other=0
        ).to(tl.int64)
        weight = _neighbour_weights(unit_vectors_ptr, neighbour, tile_mask, px, py, pz, scale, ACC)
        total += tl.sum(weight, axis=1)
        values = tl.load(
            x_ptr + field + neighbour[:, :, None] * channels + cols[None, None, :],
            mask=tile_mask[:, :, None] & col_mask[None, None, :],
            other=0.0,
        )
        acc += tl.sum(weight[:, :, None] * values.to(ACC), axis=1)

    total = tl.where(point_mask, total, 1.0)
    out = acc / total[:, None]
    tl.store(
        out_ptr + field + points[:, None] * channels + cols[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=point_mask[:, None] & col_mask[None, :],
    )
    # The weight sums are the same for every field and channel block, so one program stores them.
    if tl.program_id(1) == 0 and tl.program_id(2) == 0:
        tl.store(totals_ptr + points, total, mask=point_mask)


@triton.jit
def _gaussian_smooth_bwd(
    grad_out_ptr,
    grad_x_ptr,
    totals_ptr,
    neighbours_ptr,
    unit_vectors_ptr,
    scale_ptr,
    num_points,
    channels,
    num_neighbours,
    ACC: tl.constexpr,
    BLOCK_POINTS: tl.constexpr,
    BLOCK_NEIGHBOURS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    points = (tl.program_id(0) * BLOCK_POINTS + tl.arange(0, BLOCK_POINTS)).to(tl.int64)
    field = tl.program_id(1).to(tl.int64) * num_points * channels
    point_mask = points < num_points
    scale = tl.load(scale_ptr).to(ACC)
    px, py, pz = _load_unit_vectors(unit_vectors_ptr, points, point_mask, ACC)
    total = tl.load(totals_ptr + points, mask=point_mask, other=1.0)

    for k0 in range(0, num_neighbours, BLOCK_NEIGHBOURS):
        ks = k0 + tl.arange(0, BLOCK_NEIGHBOURS)
        tile_mask = point_mask[:, None] & (ks < num_neighbours)[None, :]
        neighbour = tl.load(
            neighbours_ptr + points[:, None] * num_neighbours + ks[None, :], mask=tile_mask, other=0
        ).to(tl.int64)
        weight = _neighbour_weights(unit_vectors_ptr, neighbour, tile_mask, px, py, pz, scale, ACC) / total[:, None]
        for c0 in range(0, channels, BLOCK_CHANNELS):
            cols = c0 + tl.arange(0, BLOCK_CHANNELS)
            col_mask = cols < channels
            grad = tl.load(
                grad_out_ptr + field + points[:, None] * channels + cols[None, :],
                mask=point_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(ACC)
            tl.atomic_add(
                grad_x_ptr + field + neighbour[:, :, None] * channels + cols[None, None, :],
                weight[:, :, None] * grad[:, None, :],
                mask=tile_mask[:, :, None] & col_mask[None, None, :],
                sem="relaxed",
            )


class GaussianSmoothingTriton(torch.autograd.Function):
    """Gaussian smoothing of a ``(batch, num_points, channels)`` field."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, neighbours: torch.Tensor, unit_vectors: torch.Tensor, sigma: float):
        acc_dtype = accumulation_dtype(x.dtype)
        x = x.contiguous()
        neighbours = neighbours.contiguous()
        unit_vectors = unit_vectors.contiguous()
        batch, num_points, channels = x.shape
        scale = torch.full((1,), 1.0 / (2.0 * sigma**2), dtype=acc_dtype, device=x.device)
        totals = torch.empty(num_points, dtype=acc_dtype, device=x.device)
        out = torch.empty_like(x)
        block_channels = _block_channels(channels, _FWD_MAX_CHANNELS)
        grid = (triton.cdiv(num_points, _FWD_POINTS), triton.cdiv(channels, block_channels), batch)
        # A rank can hold zero channels when the grid is split by channels.
        if out.numel() > 0:
            with torch.cuda.device(out.device):
                _gaussian_smooth_fwd[grid](
                    x,
                    out,
                    totals,
                    neighbours,
                    unit_vectors,
                    scale,
                    num_points,
                    channels,
                    neighbours.shape[1],
                    ACC=torch_dtype_to_triton(acc_dtype),
                    BLOCK_POINTS=_FWD_POINTS,
                    BLOCK_NEIGHBOURS=_FWD_NEIGHBOURS,
                    BLOCK_CHANNELS=block_channels,
                    num_warps=_FWD_WARPS,
                )
        ctx.save_for_backward(neighbours, unit_vectors, scale, totals)
        ctx.dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        neighbours, unit_vectors, scale, totals = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        batch, num_points, channels = grad_out.shape
        grad_x = torch.zeros(grad_out.shape, dtype=scale.dtype, device=grad_out.device)
        grid = (triton.cdiv(num_points, _BWD_POINTS), batch)
        # A rank can hold zero channels when the grid is split by channels.
        if grad_x.numel() > 0:
            with torch.cuda.device(grad_x.device):
                _gaussian_smooth_bwd[grid](
                    grad_out,
                    grad_x,
                    totals,
                    neighbours,
                    unit_vectors,
                    scale,
                    num_points,
                    channels,
                    neighbours.shape[1],
                    ACC=torch_dtype_to_triton(scale.dtype),
                    BLOCK_POINTS=_BWD_POINTS,
                    BLOCK_NEIGHBOURS=_BWD_NEIGHBOURS,
                    BLOCK_CHANNELS=_BWD_CHANNELS,
                    num_warps=_BWD_WARPS,
                )
        return grad_x.to(ctx.dtype), None, None, None


def gaussian_smooth_triton(
    x: torch.Tensor, neighbours: torch.Tensor, unit_vectors: torch.Tensor, sigma: float
) -> torch.Tensor:
    """See :func:`anemoi.models.layers.gaussian_smoothing.gaussian_smooth`."""
    return GaussianSmoothingTriton.apply(x, neighbours, unit_vectors, sigma)
