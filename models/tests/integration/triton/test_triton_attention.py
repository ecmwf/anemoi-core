# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# These tests for a triton fused attention algorithm are adapted from
# the fused attention example from the Triton-lang github repo (MIT license) (Credits: OpenAI kernel team)
# The tests have been extended to support sliding window and to test real-world problem sizes against flash attention if its available

import math

import einops
import pytest
import torch
import triton

from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.attention import TritonAttention
    from anemoi.models.triton.attention import is_hip

try:
    from flash_attn import flash_attn_func

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize(
    "N_CTX", [1024, 40320] if HAS_FLASH else [1024]
)  # test larger (o96) config if FLASH_ATTN is available to compute reference
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("warp_specialize", [False])
@pytest.mark.parametrize(
    "window", [0, 512, 1120] if HAS_FLASH else [0, 512]
)  # test larger (o96) config if FLASH_ATTN is available to compute reference
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
def test_triton_attention(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, window, mode, dtype=torch.float16):
    """Compares Triton flash attention against a naive torch implementation, and optionally flash attention
    
    Since flash attention is more memory efficient, installing it allows larger problem sizes
    to be tested (in this case, an o96 processor setup).
    """
    attention = TritonAttention.apply
    
    if not is_triton_available():
        pytest.skip("Triton not available")
    
    if window > 0 and causal:
        pytest.skip("Causal and sliding window together not supported")
    torch.manual_seed(20)
    try:
        DEVICE = triton.runtime.driver.active.get_active_torch_device()
    except RuntimeError:
        pytest.skip("No GPU detected")

    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    sm_scale = 1 / math.sqrt(q.size(-1))
    # reference implementation
    ref_dtype = dtype
    
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    
    # Compute reference values
    if not HAS_FLASH:
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        
        # Optionally mask values
        if causal:
            # Create causal mask
            p[:, :, M == 0] = float("-inf")
        if window != 0:
            # Create sliding window mask
            positions = torch.arange(N_CTX, device="cuda")
            mask = abs(positions[:, None] - positions[None, :]) <= window
            p[:, :, ~mask] = float("-inf")
            
        p = torch.softmax(p.float(), dim=-1)
        p = p.to(ref_dtype)
        ref_out = torch.matmul(p, v).half()

        if mode == "bwd":
            dout = torch.randn_like(q)
            ref_out.backward(dout)
            ref_dq, q.grad = q.grad.clone(), None
            ref_dv, v.grad = v.grad.clone(), None
            ref_dk, k.grad = k.grad.clone(), None
    else:
        # Flash attention references
        q_flash, k_flash, v_flash = (einops.rearrange(t, "b h s d -> b s h d") for t in (q, k, v))
        q_flash.retain_grad()
        k_flash.retain_grad()
        v_flash.retain_grad()
        flash_window = (-1, -1) if window == 0 else (window, window)
        ref_out = flash_attn_func(
            q_flash, k_flash, v_flash, causal=causal, window_size=flash_window, softmax_scale=sm_scale
        )

        if mode == "bwd":
            dout = torch.randn_like(q)
            dout_flash = einops.rearrange(dout, "b s h d -> b h s d")
            ref_out.backward(dout_flash)
            ref_dq, q.grad = q_flash.grad.clone(), None
            ref_dv, v.grad = v_flash.grad.clone(), None
            ref_dk, k.grad = k_flash.grad.clone(), None

            # rearrange for later comparison w triton version
            ref_dq = einops.rearrange(ref_dq, "b s h d -> b h s d")
            ref_dv = einops.rearrange(ref_dv, "b s h d -> b h s d")
            ref_dk = einops.rearrange(ref_dk, "b s h d -> b h s d")
        ref_out = einops.rearrange(ref_out, "b s h d -> b h s d")

    # Compute triton values
    tri_out = attention(q, k, v, causal, window, sm_scale, warp_specialize).half()
    
    if mode == "fwd":
        atol = 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if is_hip() and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)
