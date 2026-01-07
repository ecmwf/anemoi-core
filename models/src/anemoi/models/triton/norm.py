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
        w = tl.load(w_ptr + off).to(tl.float32).reshape((H,C))
        
    # compute rms scale factor
    x = _rms_norm_fwd(x, w, C)
    
    tl.store(
        out_ptr + off,
        x.reshape(H*C)
    )

@triton.jit
def _rms_norm_bwd(x, dout, C):
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

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    x_hat = x * inv_rms

    # first term
    dx = dout * inv_rms
    # second term
    dot = tl.sum(dout * x, axis=1, keep_dims=True)
    dx = dx - (x * inv_rms * inv_rms * inv_rms) * (dot / C)
    
    dw = tl.sum(dout * x_hat, axis=0, keep_dims=True) #TODO check axis, in reference its axis 0

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
    
    dx, dw = _rms_norm_bwd(x, dout, C)
    
    tl.store(dx_ptr + off, dx.reshape(H*C,))
    if dw_ptr is not None:
        tl.store(dw_ptr + C_off, dw.reshape(C,))
    
    

class RMSNormFunction(torch.autograd.Function):
    """Wrapper function for the Triton RMS kernels.
    
    The triton kernels were designed to be called inside the Triton GT kernel. 
    This wrapper is primarily intended for correctness testing purposes."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("Error. The Triton RMS Norm is only supported with the 'cuda' backend but it is not available.")

    @staticmethod
    def forward(ctx, x, w=None):
        """Args:
        x: [H, C]
        elementwise_affine: bool
        """
        H, C = x.shape

        # Ensure contiguous memory layout for Triton
        x = x.contiguous()
        
        out = torch.empty_like(x)
        #if elementwise_affine:
        #    w = torch.ones(H, C, device=x.device, dtype=torch.float32)
        #else:
        #    w = None

        #ctx.elementwise_affine = elementwise_affine

        _rms_norm_fwd_standalone[(1,)](x, out, w, H, C)

        # Save tensors for backward
        ctx.save_for_backward(x, out, w)
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        x, out, w= ctx.saved_tensors
        H, C = x.shape

        # Allocate grads and intermediates
        dX = torch.empty_like(x)
        #dW = torch.empty_like(x)
        dW = None
        if w is not None:
            dW= torch.empty(H,C, dtype=torch.float32, device="cuda")
            #dW = torch.empty_like(x)
        
        #if w is not None:
        #    dW = torch.empty_like(w)
            
        _rms_norm_bwd_standalone[(1,)](x, dout, w, dX, dW, H, C)
        
        return dX, dW

#https://veitner.bearblog.dev/backprop-through-rmsnorm/
def rmsnorm_bwd_ref(x, w, dout, C):
    """Reference implementation for RMSNorm backward pass."""
    
    eps: tl.constexpr = 0.0  # small numerical stability factor
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    
    # first term
    x_hat = x * inv_rms
    wdy = dout * w
    dot = tl.sum(wdy * x_hat, axis=1, keep_dims=True)/C #c1
    dx =  - (x * inv_rms * inv_rms * inv_rms) * dot

    # dL/dW
    dw = (dout * x_hat).sum(dim=0)
    return dx, dw