# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

# check if triton is installed
# If pytorch is installed on CPU then torch is not available
try:
    import triton
    import triton.language as tl
except ImportError:
    raise ValueError(
        "Error. 'triton' is not installed. the RMSNorm triton implementation can not be used. Full error: {e}"
    )

import logging

from anemoi.models.triton.utils import build_masks_and_offsets

LOGGER = logging.getLogger(__name__)


@triton.jit
def _layer_norm_fwd(x, C):
    """Normalises x in-place along the channel dimension

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_layer_norm_fwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        C: int
    returns:
        x_norm: (H_pad,C_pad)
    """
    eps: tl.constexpr = 1e-5  # small numerical stability factor, same ep as pytorch LN

    # mask padded values to 0 so they don't contribute to mean and variance
    C_pad: tl.constexpr = x.shape[-1]
    mask = tl.arange(0, C_pad) < C
    x = tl.where(mask, x, 0.0)

    mean_x = tl.sum(x, axis=1, keep_dims=True) / C
    var_x = tl.sum(x * x, axis=1, keep_dims=True) / C - mean_x * mean_x
    x = (x - mean_x) * tl.rsqrt(var_x + eps)

    # Mask again after normalization (padded values are no longer 0 after subtracting mean)
    x = tl.where(mask, x, 0.0)
    return x


@triton.jit
def _layer_norm_bwd(x, dout, C: tl.constexpr):
    """Computes the backward pass of LayerNorm

    x is the original unnormalised input.

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_layer_norm_bwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        dout: (H_pad,C_pad)
        C: int
    returns:
        dx: (H_pad,C_pad)
    """
    eps: tl.constexpr = 1e-5  # small numerical stability factor, same ep as pytorch LN

    # mask padded values to 0, so they don't contribute to mean and variance
    C_pad: tl.constexpr = x.shape[-1]
    mask = tl.arange(0, C_pad) < C
    x = tl.where(mask, x, 0.0)
    dout = tl.where(mask, dout, 0.0)

    mean_x = tl.sum(x, axis=1, keep_dims=True) / C
    var_x = tl.sum(x * x, axis=1, keep_dims=True) / C - mean_x * mean_x
    inv_std = tl.rsqrt(var_x + eps)
    x_hat = (x - mean_x) * inv_std

    c1 = tl.sum(dout, axis=1, keep_dims=True) / C
    c2 = tl.sum(dout * x_hat, axis=1, keep_dims=True) / C
    grad_x = inv_std * (dout - c1 - x_hat * c2)

    return grad_x


@triton.jit
def _rms_norm_fwd(x, C):
    """Normalises x in-place along the channel dimension

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_rms_norm_fwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        C: int
    returns:
        x_norm: (H_pad,C_pad)
    """
    eps: tl.constexpr = 1e-7  # small numerical stability factor

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    x = x * inv_rms

    return x


@triton.jit
def _norm_fwd_standalone(x_ptr, out_ptr, H: tl.constexpr, C: tl.constexpr, norm_type: tl.constexpr):
    """Normalises x in-place along the channel dimension

    Wrapper which handles the loading and storing for "_rms_norm_fwd" or "_layer_norm_fwd".
    Picks norm implementation based on norm_type argument (1 for RMSNorm, 2 for LayerNorm)

    x_ptr is a pointer is the original unnormalised input.

    x is loaded and the computed output is stored.

    inputs:
        x_ptr: (H_pad,C_pad)
        out_ptr: (H_pad,C_pad)
        H: int
        C: int
        norm_type: int (1 for RMSNorm, 2 for LayerNorm)
    returns:
        None
    """
    B = tl.program_id(0)

    # load x tensors
    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    _, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)
    x = tl.load(x_ptr + B * H * C + H_C_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    if norm_type == 1:
        x = _rms_norm_fwd(x, C)
    elif norm_type == 2:
        x = _layer_norm_fwd(x, C)

    tl.store(out_ptr + B * H * C + H_C_off, x.reshape(H_pad * C_pad), mask=H_C_mask)


@triton.jit
def _rms_norm_bwd(x, dout, C: tl.constexpr):
    """Computes the backward pass of RMSNorm

    x is the original unnormalised input.

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_rms_norm_bwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        dout: (H_pad,C_pad)
        C: int
    returns:
        dx: (H_pad,C_pad)
    """
    eps: tl.constexpr = 1e-7  # small numerical stability factor

    # Set padded values to 0 explictly
    # Needed because we sum along that axis
    C_pad: tl.constexpr = x.shape[-1]
    mask = tl.arange(0, C_pad) < C
    x = tl.where(mask, x, 0.0)
    dout = tl.where(mask, dout, 0.0)

    # recompute forward pass scaling factor
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)

    # first term
    x_hat = x * inv_rms
    dot = tl.sum(dout * x_hat, axis=1, keep_dims=True) / C
    # second term
    dx = (dout - x_hat * dot) * inv_rms

    return dx


@triton.jit
def _norm_bwd_standalone(
    x_ptr,
    dout_ptr,
    dx_ptr,
    H: tl.constexpr,
    C: tl.constexpr,
    norm_type: tl.constexpr,
):
    """Computes the backward pass of normalisation

    Wrapper which handles the loading and storing for "_rms_norm_bwd" or "_layer_norm_bwd".
    Picks norm implementation based on norm_type argument (1 for RMSNorm, 2 for LayerNorm)

    x_ptr is a pointer is the original unnormalised input.

    x and dout are loaded and dx is stored.

    inputs:
        x_ptr: (H_pad,C_pad)
        dout_ptr: (H_pad,C_pad)
        dx_ptr: (H_pad)
        H: int
        C: int
        norm_type: int (1 for RMSNorm, 2 for LayerNorm)
    returns:
        None
    """
    B = tl.program_id(0)

    # load x tensors
    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    _, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)
    x = tl.load(x_ptr + B * H * C + H_C_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    dout = tl.load(dout_ptr + B * H * C + H_C_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    if norm_type == 1:  # RMSNorm
        dx = _rms_norm_bwd(x, dout, C)
    elif norm_type == 2:  # LayerNorm
        dx = _layer_norm_bwd(x, dout, C)

    tl.store(
        dx_ptr + B * H * C + H_C_off,
        dx.reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )


class _NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, norm_type: int):

        B, H, C = x.shape
        x = x.contiguous()

        ctx.norm_type = norm_type

        out = torch.empty_like(x)

        _norm_fwd_standalone[(B,)](x, out, H, C, norm_type=ctx.norm_type)
        ctx.save_for_backward(x)
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        (x,) = ctx.saved_tensors
        B, H, C = x.shape

        dx = torch.empty_like(x)

        grid = B

        _norm_bwd_standalone[(grid,)](x, dout, dx, H, C, norm_type=ctx.norm_type)

        return dx, None  # None for norm_type which is not a tensor


class RMSNorm(torch.nn.Module):
    """Interface to triton RMSNorm.

    Meant for testing purposes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, H, C), got {tuple(x.shape)}")

        return _NormFunction.apply(x, 1)


class LayerNorm(torch.nn.Module):
    """Interface to triton LayerNorm.
    Intended for testing purposes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, H, C), got {tuple(x.shape)}")

        return _NormFunction.apply(x, 2)
