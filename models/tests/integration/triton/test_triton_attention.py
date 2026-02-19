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

HAS_FLASH= True


def attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
    window_size: int = -1,
) -> torch.Tensor:
    """
    Reference implementation for global attention with variable-length sequences.
    
    Args:
        q: Query tensor of shape [Total_Q_tokens, H, D]
        k: Key tensor of shape [Total_K_tokens, H, D]
        v: Value tensor of shape [Total_K_tokens, H, D]
        cu_seqlens_q: Cumulative sequence lengths for queries, shape [B+1]
        cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [B+1]
        sm_scale: Softmax scale (typically 1/sqrt(d))
        causal: Whether to apply causal masking
        window_size: Sliding window size (-1 for no window)
    
    Returns:
        Output tensor of shape [Total_Q_tokens, H, D]
    """
    device = q.device
    dtype = q.dtype
    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    
    # Collect outputs for each sequence
    outputs = []
    
    for b in range(batch_size):
        # Extract the current sequence
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        
        q_seq = q[q_start:q_end]  # [seq_len_q, H, D]
        k_seq = k[k_start:k_end]  # [seq_len_k, H, D]
        v_seq = v[k_start:k_end]  # [seq_len_k, H, D]
        
        seq_len_q = q_end - q_start
        seq_len_k = k_end - k_start
        
        # Reshape for batch matrix multiplication
        # [seq_len, H, D] -> [H, seq_len, D]
        q_seq = q_seq.transpose(0, 1)
        k_seq = k_seq.transpose(0, 1)
        v_seq = v_seq.transpose(0, 1)
        
        # Compute attention scores: [H, seq_len_q, D] @ [H, D, seq_len_k] -> [H, seq_len_q, seq_len_k]
        scores = torch.matmul(q_seq, k_seq.transpose(1, 2)) * sm_scale
        
        # Apply masks
        if causal:
            # Causal mask: each query position can only attend to positions <= its own position
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        
        if window_size != -1:
            # Sliding window mask
            positions_q = torch.arange(seq_len_q, device=device)
            positions_k = torch.arange(seq_len_k, device=device)
            window_mask = torch.abs(positions_q[:, None] - positions_k[None, :]) > window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0), float("-inf"))
        
        # Apply softmax
        attn_weights = torch.softmax(scores.float(), dim=-1).to(dtype)
        
        # Compute output: [H, seq_len_q, seq_len_k] @ [H, seq_len_k, D] -> [H, seq_len_q, D]
        out_seq = torch.matmul(attn_weights, v_seq)
        
        # Reshape back: [H, seq_len_q, D] -> [seq_len_q, H, D]
        out_seq = out_seq.transpose(0, 1)
        
        outputs.append(out_seq)
    
    # Concatenate all sequences
    output = torch.cat(outputs, dim=0)  # [Total_Q_tokens, H, D]
    
    return output


#def test_triton_attention_deterministic():
#    """ Computes the same test case 50 times in a row and checks that the output matches to ensure that the implementation is deterministic. """
#    raise NotImplementedError("TODO(cathal): implement this test.")

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("Z", [1, 2]) #4, 8, 16])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize(
    "N_CTX", [1, 2, 4, 8, 16, 18, 32, 33, 34, 38, 42, 48, 52, 64, 65, 68] 
    #"N_CTX", [32]
    # BLOCK_FIXED is locked to 128 for pytests, so 128 is the smallest possible context length
)  # test larger (o96) config if FLASH_ATTN is available to compute reference
@pytest.mark.parametrize("HEAD_DIM", [64])
@pytest.mark.parametrize("causal", [False])  # TODO(cathal) fix 0.0% mismatch for causal=True for some configurations
@pytest.mark.parametrize(
    "window",
    [False]
    #[0]
)  # test larger (o96) config if FLASH_ATTN is available to compute reference
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_triton_attention(Z, H, N_CTX, HEAD_DIM, causal, window, mode, dtype):
    """Compares Triton flash attention against a naive torch implementation, and optionally flash attention

    Since flash attention is more memory efficient, installing it allows larger problem sizes
    to be tested (in this case, an o96 processor setup).
    """
    attention = TritonAttention.apply
    
    if N_CTX > 2048 and not HAS_FLASH:
        pytest.skip("N_CTX > 2048 will cause OOM for naive pytorch reference implementation, so we skip these tests when flash attention is not available.")

    if not is_triton_available():
        pytest.skip("Triton not available")

    if window and causal:
        pytest.skip("Causal and sliding window together not supported")
    torch.manual_seed(42)
    try:
        DEVICE = triton.runtime.driver.active.get_active_torch_device()
    except RuntimeError:
        pytest.skip("No GPU detected")

    q = torch.rand((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).requires_grad_()
    k = torch.rand((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).requires_grad_()
    v = torch.rand((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).requires_grad_()
    sm_scale = 1 / math.sqrt(q.size(-1))
    # reference implementation
    ref_dtype = dtype

    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    
    window_size = -1
    if window:
        window_size = int(torch.randint(0, N_CTX, (1,))[0])
        

    # Compute reference values
    if not HAS_FLASH:
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

        # Optionally mask values
        if causal:
            # Create causal mask
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
            p[:, :, M == 0] = float("-inf")
        if window_size != -1:
            # Create sliding window mask
            positions = torch.arange(N_CTX, device="cuda")
            mask = abs(positions[:, None] - positions[None, :]) <= window_size  
            p[:, :, ~mask] = float("-inf")

        p = torch.softmax(p.float(), dim=-1)
        p = p.to(ref_dtype)
        ref_out = torch.matmul(p, v).to(dtype)

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
        flash_window = (-1, -1) if not window else (window_size, window_size)
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
    tri_out = attention(q, k, v, causal, window_size, sm_scale).to(dtype)

    # Set tolerances based on dtype precision
    # bfloat16 has 7 mantissa bits vs float16's 10 bits, so ~8x less precision
    if dtype == torch.bfloat16:
        atol = 5e-3
        rtol = 1e-2
    else:
        atol = 1e-3
        rtol = 0.0
    
    if mode == "fwd":
        try:
            torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=rtol)
        except AssertionError:
            # Diagnostic information to help locate where the mismatch comes from.
            with torch.no_grad():
                diff = (tri_out - ref_out).abs()

                # Max error per (batch, head) to see if only some batches/heads are affected.
                # Shape: [Z, H]
                per_bh_max = diff.amax(dim=(-1, -2))
                print("[triton-attn debug] max abs error per (batch, head):", per_bh_max.detach().cpu())

                # Global max and its location
                max_err = diff.max()
                max_idx = (diff == max_err).nonzero(as_tuple=False)[0]
                z, h, t, d = [int(x) for x in max_idx]
                print(
                    "[triton-attn debug] global max abs error:",
                    float(max_err.detach().cpu()),
                    "at (batch, head, token, dim)=",
                    (z, h, t, d),
                )

                # Print a small slice around the offending token to inspect cross-batch/head behaviour.
                print("[triton-attn debug] tri_out[z, h, t, :8] =",
                      tri_out[z, h, t, :8].detach().cpu())
                print("[triton-attn debug] ref_out[z, h, t, :8] =",
                      ref_out[z, h, t, :8].detach().cpu())

            # Re-raise so the test still fails, but with extra context.
            raise
        return

    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=rtol)
    
    # Backward pass may have additional hardware-specific requirements
    bwd_rtol = rtol
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if is_hip() and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        bwd_rtol = max(1e-2, bwd_rtol)

    torch.testing.assert_close(tri_dq, ref_dq, atol=atol, rtol=bwd_rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=atol, rtol=bwd_rtol)
    torch.testing.assert_close(tri_dv, ref_dv, atol=atol, rtol=bwd_rtol)
