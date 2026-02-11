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

from anemoi.models.triton.utils import build_masks_and_offsets


@triton.jit
def _layer_norm_fwd(x, w, C, elementwise_affine: tl.constexpr):
    """Normalises x in-place along the channel dimension

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_layer_norm_fwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        w: (C_pad) or None
        C: int
        elementwise_affine: bool - whether LayerNorm has a learnable elementwise affine
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
    # Optionally apply elementwise affine
    if elementwise_affine:
        x = x * w

    return x

@triton.jit
def _layer_norm_bwd(x, w, dout, C: tl.constexpr, elementwise_affine: tl.constexpr):
    """Computes the backward pass of LayerNorm

    x is the original unnormalised input.

    w is an optional learnable elementwise affine.

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_layer_norm_bwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        w: (C_pad) or None, depending on elementwise_affine
        dout: (H_pad,C_pad)
        C: int
        elementwise_affine: bool - whether LayerNorm has a learnable elementwise affine
    returns:
        dx: (H_pad,C_pad)
        dw: (C_pad), depending on elementwise_affine
    """
    eps: tl.constexpr = 1e-5  # small numerical stability factor, same ep as pytorch LN
    
    # mask padded values to 0, so they don't contribute to mean and variance
    C_pad: tl.constexpr = x.shape[-1]
    mask = tl.arange(0, C_pad) < C
    x = tl.where(mask, x, 0.0)
    wdy = dout
    if elementwise_affine:
        wdy = dout * w
    wdy = tl.where(mask, wdy, 0.0)
 
    mean_x = tl.sum(x, axis=1, keep_dims=True) / C
    var_x = tl.sum(x * x, axis=1, keep_dims=True) / C - mean_x * mean_x
    inv_std = tl.rsqrt(var_x + eps)
    x_hat = (x - mean_x) * inv_std

    c1 = tl.sum(wdy, axis=1, keep_dims=True)/ C
    c2 = tl.sum(wdy * x_hat, axis=1, keep_dims=True) / C
    grad_x = inv_std * (dout - c1 - x_hat * c2)

    # dL/dW
    if elementwise_affine:
        dw = tl.sum(
            dout * x_hat,
            axis=0,
        )
        return grad_x, dw
    else:
        # can't return None for dw, since triton inlined functions can't return None.
        # so return a reference to existing tensor which will be ignored by the caller
        return grad_x, 0

@triton.jit
def _rms_norm_fwd(x, w, C, elementwise_affine: tl.constexpr):
    """Normalises x in-place along the channel dimension

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_rms_norm_fwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        w: (C_pad) or None
        C: int
        elementwise_affine: bool - whether RMSNorm has a learnable elementwise affine
    returns:
        x_norm: (H_pad,C_pad)
    """
    eps: tl.constexpr = 1e-7  # small numerical stability factor

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    x = x * inv_rms

    # Optionally apply elementwise affine
    if elementwise_affine:
        x = x * w

    return x


@triton.jit
def _norm_fwd_standalone(x_ptr, out_ptr, w_ptr, H: tl.constexpr, C: tl.constexpr, norm_type: tl.constexpr, elementwise_affine: tl.constexpr):
    """Normalises x in-place along the channel dimension

    Wrapper which handles the loading and storing for "_rms_norm_fwd" or "_layer_norm_fwd".
    Picks norm implementation based on norm_type argument (1 for RMSNorm, 2 for LayerNorm)

    x_ptr is a pointer is the original unnormalised input.

    w_ptr is a pointer to an optional learnable elementwise affine.

    x is loaded and the computed output is stored.
    If w_ptr is given, then w is loaded

    inputs:
        x_ptr: (H_pad,C_pad)
        out_ptr: (H_pad,C_pad)
        w_ptr: (C_pad) or None
        H: int
        C: int
        norm_type: int (1 for RMSNorm, 2 for LayerNorm)
        elementwise_affine: bool - whether normalisation has a learnable elementwise affine
    returns:
        None
    """
    B = tl.program_id(0)

    # load x and w tensors
    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    _, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)
    C_pad_off = tl.arange(0, C_pad)
    C_mask = C_pad_off < C
    x = tl.load(x_ptr + B * H * C + H_C_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    w = None
    if w_ptr is not None:
        w = tl.load(w_ptr + C_pad_off, mask=C_mask).to(tl.float32)

    if norm_type == 1:
        x = _rms_norm_fwd(x, w, C, elementwise_affine)
    elif norm_type == 2:
        x = _layer_norm_fwd(x, w, C, elementwise_affine)

    tl.store(out_ptr + B * H * C + H_C_off, x.reshape(H_pad * C_pad), mask=H_C_mask)


@triton.jit
def _rms_norm_bwd(x, w, dout, C: tl.constexpr, elementwise_affine: tl.constexpr):
    """Computes the backward pass of RMSNorm

    x is the original unnormalised input.

    w is an optional learnable elementwise affine.

    This function only performs arithmetic, not any loading or storing
    so that it can be called inside larger kernels which handles the loads and stores.
    See "_rms_norm_bwd_standalone" for a version with loads and stores

    inputs:
        x: (H_pad,C_pad)
        w: (C_pad) or None, depending on elementwise_affine
        dout: (H_pad,C_pad)
        C: int
        elementwise_affine: bool - whether RMSNorm has a learnable elementwise affine
    returns:
        dx: (H_pad,C_pad)
        dw: (C_pad), depending on elementwise_affine
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
    wdy = dout
    if elementwise_affine:
        wdy = dout * w

    dot = tl.sum(wdy * x_hat, axis=1, keep_dims=True) / C
    # second term
    dx = (wdy - x_hat * dot) * inv_rms

    # dL/dW
    if elementwise_affine:
        dw = tl.sum(
            dout * x_hat,
            axis=0,
        )
        return dx, dw
    else:
        # can't return None for dw, since triton inlined functions can't return None.
        # so return a reference to existing tensor which will be ignored by the caller
        return dx, 0


@triton.jit
def _norm_bwd_standalone(
    x_ptr, dout_ptr, w_ptr, dx_ptr, dw_partial_ptr, H: tl.constexpr, C: tl.constexpr, norm_type: tl.constexpr, elementwise_affine: tl.constexpr
):
    """Computes the backward pass of normalisation

    Wrapper which handles the loading and storing for "_rms_norm_bwd" or "_layer_norm_bwd".
    Picks norm implementation based on norm_type argument (1 for RMSNorm, 2 for LayerNorm)

    x_ptr is a pointer is the original unnormalised input.

    w_ptr is a pointer to an optional learnable elementwise affine.

    x and dout are loaded and dx is stored.
    If w_ptr is given, then w is loaded and dw stored.

    A parallel reduction strategy is used efficiently accumulate dW when elementwise_affine is True.

    inputs:
        x_ptr: (H_pad,C_pad)
        dout_ptr: (H_pad,C_pad)
        w_ptr: (C_pad) or None
        dx_ptr: (H_pad)
        dw_partial_ptr: (grid_size, C_pad) or None
        H: int
        C: int
        norm_type: int (1 for RMSNorm, 2 for LayerNorm)
        elementwise_affine: bool - whether RMSNorm has a learnable elementwise affine
    returns:
        None
    """
    B = tl.program_id(0)

    # load x and w tensors
    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    _, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)
    C_pad_off = tl.arange(0, C_pad)
    C_mask = C_pad_off < C
    x = tl.load(x_ptr + B * H * C + H_C_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    dout = tl.load(dout_ptr + B * H * C + H_C_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    w = None
    if elementwise_affine:
        w = tl.load(w_ptr + C_pad_off, mask=C_mask).to(tl.float32)

    if norm_type == 1: # RMSNorm
        dx, dw = _rms_norm_bwd(x, w, dout, C, elementwise_affine)
    elif norm_type == 2: # LayerNorm
        dx, dw = _layer_norm_bwd(x, w, dout, C, elementwise_affine)

    tl.store(
        dx_ptr + B * H * C + H_C_off,
        dx.reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )
    if elementwise_affine:
        tl.store(dw_partial_ptr + B * C + C_pad_off, dw, mask=C_mask)


@triton.jit
def _rms_norm_bwd_accumulate_dw(dw_partial_ptr, dw_ptr, N: tl.constexpr, H: tl.constexpr, C: tl.constexpr):
    """inputs:
        dw_partial_ptr: (N,C_pad)
        dw_ptr: (C_pad)
        N: int
        H: int
        C: int
    returns:
        None
    """

    C_pad: tl.constexpr = triton.next_power_of_2(C)
    C_pad_off = tl.arange(0, C_pad)
    C_mask = C_pad_off < C

    acc = tl.zeros((C_pad,), dtype=tl.float32)
    for curr in range(0, N):
        acc += tl.load(
            dw_partial_ptr + curr * C + C_pad_off,
            mask=C_mask,
        )

    tl.store(dw_ptr + C_pad_off, acc, mask=C_mask)


class _NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, norm_type: int):

        B, H, C = x.shape
        x = x.contiguous()

        ctx.elementwise_affine = weight is not None
        ctx.norm_type = norm_type

        out = torch.empty_like(x)

        _norm_fwd_standalone[(B,)](x, out, weight, H, C, norm_type=ctx.norm_type, elementwise_affine=ctx.elementwise_affine)
        ctx.save_for_backward(x, weight)
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        x, weight = ctx.saved_tensors
        B, H, C = x.shape

        dx = torch.empty_like(x)

        grid = B

        dw = None
        dw_partial = None
        if ctx.elementwise_affine:
            dw_partial = torch.zeros(
                (
                    grid,
                    C,
                ),
                device=x.device,
                dtype=torch.float32,
            )

        _norm_bwd_standalone[(grid,)](x, dout, weight, dx, dw_partial, H, C, norm_type=ctx.norm_type, elementwise_affine=ctx.elementwise_affine)

        # if using elementwise affine, we need to accumulate dw across the batch dimension, since each block computes a partial dw for its own batch element
        if ctx.elementwise_affine:
            if B > 1:
                dw = torch.zeros((C,), device=x.device, dtype=torch.float32)
                if ctx.norm_type == 1:
                    _rms_norm_bwd_accumulate_dw[1,](dw_partial, dw, B, H, C)
                elif ctx.norm_type == 2:
                    raise NotImplementedError("dw accumulation for LayerNorm is not implemented. Please reach out if you want this feature implemented.")
            else:
                dw = torch.squeeze(dw_partial, 0)

        return dx, dw, None


class RMSNorm(torch.nn.Module):
    """Interface to triton RMSNorm.
    
        Meant for testing purposes as the accumulation for dw when using elementwise_affine is not efficient.
    """

    def __init__(self, dim: int, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected input of shape (B, H, {self.dim}), got {tuple(x.shape)}")

        return _NormFunction.apply(x, self.weight, 1)
    
class LayerNorm(torch.nn.Module):
    """Interface to triton LayerNorm.
     Meant for testing purposes as the accumulation for dw when using elementwise_affine is not implemented.
    """

    def __init__(self, dim: int, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected input of shape (B, H, {self.dim}), got {tuple(x.shape)}")

        return _NormFunction.apply(x, self.weight, 2)
