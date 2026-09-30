# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton kernel for the turn of rotary position embeddings.

Computes the same as :func:`anemoi.models.layers.spherical_rotary.apply_rotary`: channel ``i`` and
channel ``head_dim // 2 + i`` of every point are turned by the angle of that point and pair, for the
first ``n`` pairs, and the other channels pass through. One pass reads each value once and writes it
once; the turn is worked out in float32 (float64 for float64 inputs) and stored in the input's dtype.
The backward pass is the same turn by the opposite angle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _turn(
    X,
    COS,
    SIN,
    OUT,
    NUM_POINTS,
    HEADS,
    NUM_TURNED,
    stride_xb,
    stride_xh,
    stride_xn,
    stride_ob,
    stride_oh,
    stride_on,
    SIGN: tl.constexpr,
    HALF: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    ODD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    # One program turns BLOCK_N points of one (batch, head) pair.
    point_block = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HEADS
    h = bh % HEADS
    compute = tl.float64 if WIDE else tl.float32

    n = point_block * BLOCK_N + tl.arange(0, BLOCK_N)
    c = tl.arange(0, HALF_BLOCK)
    point_ok = n < NUM_POINTS
    in_half = point_ok[:, None] & (c < HALF)[None, :]
    turned = point_ok[:, None] & (c < NUM_TURNED)[None, :]

    x_point = X + b * stride_xb + h * stride_xh + n * stride_xn
    o_point = OUT + b * stride_ob + h * stride_oh + n * stride_on
    x_row = x_point[:, None]
    o_row = o_point[:, None]
    first = tl.load(x_row + c[None, :], mask=in_half, other=0.0).to(compute)
    second = tl.load(x_row + HALF + c[None, :], mask=in_half, other=0.0).to(compute)

    # Pairs beyond NUM_TURNED keep angle zero: cosine 1, sine 0.
    table = n[:, None] * NUM_TURNED + c[None, :]
    cos = tl.load(COS + table, mask=turned, other=1.0).to(compute)
    sin = tl.load(SIN + table, mask=turned, other=0.0).to(compute)
    if SIGN < 0:
        sin = -sin

    out_first = first * cos - second * sin
    out_second = first * sin + second * cos
    tl.store(o_row + c[None, :], out_first.to(OUT.dtype.element_ty), mask=in_half)
    tl.store(o_row + HALF + c[None, :], out_second.to(OUT.dtype.element_ty), mask=in_half)

    if ODD:
        # With an odd number of channels the last one has no partner and passes through.
        last = tl.load(x_point + 2 * HALF, mask=point_ok, other=0.0)
        tl.store(o_point + 2 * HALF, last, mask=point_ok)


def _as_four_dims(x: torch.Tensor) -> torch.Tensor:
    """View ``x`` as ``(batch, heads, points, channels)``, adding leading dimensions of size one."""
    if x.dim() < 2 or x.dim() > 4:
        raise ValueError(f"Expected 2 to 4 dimensions (..., points, channels), got shape {tuple(x.shape)}.")
    while x.dim() < 4:
        x = x.unsqueeze(0)
    return x


def _launch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sign: int) -> torch.Tensor:
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty_like(x)
    x4, out4 = _as_four_dims(x), _as_four_dims(out)
    batch, heads, num_points, head_dim = x4.shape
    half = head_dim // 2
    half_block = triton.next_power_of_2(half)
    # About 8192 values of each of the four tiles per program keeps registers in check for any head size.
    block_n = max(16, min(128, 8192 // half_block))
    grid = (triton.cdiv(num_points, block_n), batch * heads)
    _turn[grid](
        x4,
        cos,
        sin,
        out4,
        num_points,
        heads,
        cos.shape[1],
        *x4.stride()[:3],
        *out4.stride()[:3],
        SIGN=sign,
        HALF=half,
        HALF_BLOCK=half_block,
        ODD=head_dim % 2 == 1,
        BLOCK_N=block_n,
        WIDE=x.dtype == torch.float64,
        num_warps=4,
    )
    return out


class SphericalRotaryTriton(torch.autograd.Function):
    """Rotary turn of ``x`` (shape ``(..., points, channels)``) with a Triton kernel, forward and backward.

    ``cos`` and ``sin`` are float32 (or float64) tables of shape ``(points, n)`` with ``n`` at most
    ``channels // 2``; they carry no gradient. The last dimension of ``x`` must be contiguous (it is
    made so if not); the other dimensions may have any strides, as after moving the heads dimension.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("The Triton rotary kernel needs a GPU tensor; use backend 'torch' on other devices.")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(
                f"The Triton rotary kernel supports float16, bfloat16, float32 and float64, got {x.dtype}."
            )
        if cos.shape != sin.shape or cos.shape[0] != x.shape[-2] or 2 * cos.shape[1] > x.shape[-1]:
            raise ValueError(
                f"Tables of shape {tuple(cos.shape)} do not fit {x.shape[-2]} points of {x.shape[-1]} channels."
            )
        cos, sin = cos.contiguous(), sin.contiguous()
        ctx.save_for_backward(cos, sin)
        return _launch(x, cos, sin, sign=1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        # A turn keeps lengths, so its gradient is the turn by the opposite angle.
        return _launch(grad, cos, sin, sign=-1), None, None


def spherical_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary turn of ``x`` with the Triton kernel; see :class:`SphericalRotaryTriton`."""
    return SphericalRotaryTriton.apply(x, cos, sin)
