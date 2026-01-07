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
        "Error. The 'triton' backend was selected for the GraphTransformer but Triton is not installed. To use this backend please install Triton. Otherwise, select a different backend for the GraphTransformer in the models config."
    )
from anemoi.models.triton.utils import torch_dtype_to_triton

@triton.jit
def _rms_norm_fwd(x, w, C):
    """Normalises x in-place along the head dimension
    inputs:
        x: (H_pad,C_pad)
        w: (H_pad, C_pad) or None
    returns:
        x_norm: (H_pad,C_pad)
    """
    eps: tl.constexpr = 0.0  # small numerical stability factor
    
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    x = x * inv_rms
    
    #Optionally apply elementwise affine
    if w is not None:
        x = x * w
    
    return x


@triton.jit
def _rms_norm_fwd_standalone(x_ptr, out_ptr, w_ptr, H: tl.constexpr, C: tl.constexpr):
    """Normalises x in-place along the head dimension
    
    Not fused so has to do extra work loading the X and W value and storing the result  
    
    inputs:
        x_ptr: (H_pad,C_pad)
        out_ptr: (H_pad, C_pad)
        w_ptr: (H_pad, C_pad) or None
    returns:
        None 
    """
    
    # load x and w tensors
    off = tl.arange(0, H * C)
    x = tl.load(x_ptr + off).to(tl.float32).reshape((H,C))
    w = None
    if w_ptr is not None:
        off_C =  tl.arange(0, C)
        w = tl.load(w_ptr + off_C).to(tl.float32).reshape((C))
        
    # compute rms scale factor
    x = _rms_norm_fwd(x, w, C)
    
    tl.store(
        out_ptr + off,
        x.reshape(H*C)
    )

@triton.jit
def _rms_norm_bwd(x, w, dout, C):
    """computes grad_x given x and grad_out
    x is the original unnormalised input

    inputs:
        x: (H_pad,C_pad)
        grad_out: (H_pad,C_pad)
        C: int
    returns:
        grad_x: (H_pad,C_pad)
    """
    eps: tl.constexpr = 0.0  # small numerical stability factor

    #inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    #x_hat = x * inv_rms

    # first term
    #dx = dout * inv_rms
    # second term
    #dot = tl.sum(dout * x, axis=1, keep_dims=True)
    #dx = dx - (x * inv_rms * inv_rms * inv_rms) * (dot / C)
    
    #dw = tl.sum(dout * x_hat, axis=0, keep_dims=True) #TODO check axis, in reference its axis 0
    
    
    #inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)

    # first term
    #dx = dout * inv_rms
    # second term
    #dot = tl.sum(dout * x, axis=1, keep_dims=True)
    #dx = dx - (x * inv_rms * inv_rms * inv_rms) * (dot / C)
    
    #dw = None
    #TODO could make this optional
    #compute_dw: tl.constexpr = True
    #if compute_dw:
    #    x_hat = x * inv_rms
    #    dw = tl.sum(dout * x_hat, axis=0, keep_dims=True)
    
    #recompute forward pass scaling factor
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    
    # first term
    x_hat = x * inv_rms
    wdy = dout
    if w is not None: 
        wdy = dout * w
    dot = tl.sum(wdy * x_hat, axis=1, keep_dims=True)/C 
    #second term
    dx = (wdy - x_hat * dot) * inv_rms

    # dL/dW
    #TODO could make optional 
    dw = tl.sum(dout * x_hat, axis=0,)

    return dx , dw

@triton.jit
def _rms_norm_bwd_standalone(x_ptr, dout_ptr, w_ptr, dx_ptr, dw_ptr, H: tl.constexpr, C: tl.constexpr):
    """computes grad_x given x and grad_out
    x is the original unnormalised input
    
    Not fused so it has to do extra work of loading and storing tensors

    inputs:
        x_ptr: (H_pad,C_pad)
        dout_ptr: (H_pad,C_pad)
        w_ptr: (H_pad, C_pad) or None
        H: int
        C: int
    returns:
        None
    """
    # load x and w tensors
    off = tl.arange(0, H * C)
    C_off = tl.arange(0,C)
    x = tl.load(x_ptr + off).to(tl.float32).reshape((H,C))
    dout = tl.load(dout_ptr + off).to(tl.float32).reshape((H,C))
    
    w = None
    if w_ptr is not None:
        off_C =  tl.arange(0, C)
        w = tl.load(w_ptr + off_C).to(tl.float32).reshape((C))
    
    #compute_dw = w_ptr is None
    
    dx, dw = _rms_norm_bwd(x, w, dout, C)
    
    tl.store(dx_ptr + off, dx.reshape(H*C,))
    if w_ptr is not None:
        tl.store(dw_ptr + C_off, dw.reshape(C,))
    
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        H, C = x.shape
        x = x.contiguous()

        out = torch.empty_like(x)

        _rms_norm_fwd_standalone[(1,)](
            x, out, weight, H, C
        )

        ctx.save_for_backward(x, weight)
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        x, weight = ctx.saved_tensors
        H, C = x.shape

        dx = torch.empty_like(x)

        dw = None
        if weight is not None:
            # dw is per-feature
            dw = torch.zeros((C,), device=x.device, dtype=torch.float32)

        _rms_norm_bwd_standalone[(1,)](
            x, dout, weight, dx, dw, H, C
        )

        return dx, dw

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input of shape (H, {self.dim}), got {tuple(x.shape)}"
            )

        return RMSNormFunction.apply(x, self.weight)


#https://veitner.bearblog.dev/backprop-through-rmsnorm/
def rmsnorm_bwd_ref(x, w, dout, C):
    """Reference implementation for RMSNorm backward pass."""
    
    eps: tl.constexpr = 0.0  # small numerical stability factor
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    
    # first term
    x_hat = x * inv_rms
    wdy = dout * w
    dot = tl.sum(wdy * x_hat, axis=1, keep_dims=True)/C #c1
    dx = x_hat - (x * inv_rms * inv_rms * inv_rms) * dot

    # dL/dW
    dw = (dout * x_hat).sum(dim=0)
    return dx, dw