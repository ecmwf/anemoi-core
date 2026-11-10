# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gradients of the Triton sliding-window attention kernel when the attention logits are large.

Large logits make attention nearly one-hot. Flash attention kernels then return gradients that are
wrong by orders of magnitude while the output still looks right (KohakuBlueleaf, KohakuFA,
https://github.com/KohakuBlueleaf/KohakuFA, docs/precision.md). It names three causes:

1. the row maximum and the row sum saved as one float32 number, m * scale + log2(l), which rounds
   log2(l) away at large m;
2. the exponent worked out as s * scale - m * scale, which leaves the row's largest probability
   slightly off 1 instead of shifting first, (s - m) * scale;
3. scale * dS rounded to float16 before the dQ and dK products instead of dS, which pushes small
   score gradients below float16's normal range.

Each test below fails on a copy of the kernel with one of these put back in. The kernel is compared
with attention in float64 on the same rounded inputs, and with what dense attention in float32 or an
exact emulation of a correct 16-bit kernel reaches on those inputs. Float32 inputs run with PyTorch's
float32 matmul precision at "highest", which the kernel follows with full float32 matrix products.
"""

import contextlib
import math

import pytest
import torch

from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.attention import TritonAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_triton_available(), reason="CUDA and Triton needed"),
]

MASKINGS = [(False, -1), (True, -1), (False, 32)]  # (causal, window): global, causal, sliding window
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# A multiple of the largest block, and a length that takes the kernel's path for a partly filled last block.
LENGTHS = [256, 203]


@contextlib.contextmanager
def _float32_matmul_precision(precision):
    """PyTorch's float32 matmul precision set to ``precision`` inside the block, put back afterwards."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.fixture(autouse=True)
def _full_float32_matmuls():
    with _float32_matmul_precision("highest"):
        yield


def _visible(n, causal, window):
    """Which keys (columns) each query (row) sees."""
    positions = torch.arange(n, device="cuda")
    offsets = positions[:, None] - positions[None, :]
    mask = torch.ones(n, n, dtype=torch.bool, device="cuda")
    if causal:
        mask &= offsets >= 0
    if window >= 0:
        mask &= offsets.abs() <= window
    return mask


def _dense(qkv, grad_out, sm_scale, mask, dtype):
    """Dense attention computed in ``dtype``; returns out, dq, dk, dv."""
    q, k, v = (t.detach().to(dtype).requires_grad_() for t in qkv)
    scores = (q @ k.transpose(-1, -2)) * sm_scale
    out = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v
    out.backward(grad_out.to(dtype))
    return [out.detach(), q.grad, k.grad, v.grad]


def _kernel(qkv, grad_out, sm_scale, causal, window):
    q, k, v = (t.detach().clone().requires_grad_() for t in qkv)
    out = TritonAttention.apply(q, k, v, causal, window, sm_scale)
    out.backward(grad_out)
    return [out.detach(), q.grad, k.grad, v.grad]


def _error(result, reference):
    """Relative error over the whole tensor."""
    return ((result.double() - reference).norm() / reference.norm()).item()


def _clustered_inputs(logit_size, dtype, n=512, heads=4, head_dim=64, jitter=3e-2, seed=0):
    """KohakuFA's benchmark inputs (benchmarks/precision.py): queries and keys around 16 shared directions.

    Most rows are nearly one-hot, keys from the same direction nearly tie, and the largest scaled
    logit is about ``logit_size``.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    centres = torch.randn(1, heads, 16, head_dim, device="cuda", generator=generator)
    centres = centres / centres.norm(dim=-1, keepdim=True)
    pick = torch.randint(0, 16, (1, heads, n), device="cuda", generator=generator)
    base = torch.gather(centres, 2, pick[..., None].expand(-1, -1, -1, head_dim))
    radius = math.sqrt(logit_size * math.sqrt(head_dim))
    q = radius * (base + jitter * torch.randn(base.shape, device="cuda", generator=generator))
    k = radius * (base + jitter * torch.randn(base.shape, device="cuda", generator=generator))
    v = torch.randn(base.shape, device="cuda", generator=generator)
    grad_out = torch.randn(base.shape, device="cuda", generator=generator)
    return [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("causal,window", MASKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_two_keys_close_to_a_tie_at_large_scores(dtype, causal, window, n):
    """One query sees two keys whose scaled scores are about 1.2e6 but only about 1 apart.

    The raw scores q . k are exact in float32. Subtracting the larger one before scaling keeps their
    gap exact, and keeping the row maximum and the row sum apart keeps the sum exact; scaling first,
    or saving one number m * scale + log2(l), changes the probabilities by a few percent.
    """
    head_dim = 32  # 1 / sqrt(32) is not a power of two, so scaling the scores rounds them
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.zeros(1, 2, n, head_dim, device="cuda")
    k = torch.zeros_like(q)
    query, first, second = n // 2, n // 2 - 1, n // 2 - 3
    q[:, :, query, 0], q[:, :, query, 1] = 1024.0, 1.0
    k[:, :, first, 0] = 6400.0  # raw score 6553600
    k[:, :, second, 0], k[:, :, second, 1] = 6400.0, -6.0  # raw score 6 lower, scaled about 1.06 lower
    v = torch.randn(q.shape, device="cuda", generator=generator)
    grad_out = torch.randn(q.shape, device="cuda", generator=generator)
    qkv, grad_out = [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)
    sm_scale = head_dim**-0.5

    reference = _dense(qkv, grad_out, sm_scale, _visible(n, causal, window), torch.float64)
    results = _kernel(qkv, grad_out, sm_scale, causal, window)
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype]
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        scale = ref.abs().max()
        torch.testing.assert_close(result.double() / scale, ref / scale, rtol=0, atol=tolerance, msg=name)


@pytest.mark.parametrize("causal,window", MASKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_kohakufa_huge_logits(dtype, causal, window):
    """KohakuFA's regression test (tests/test_attention.py::test_huge_logits) with its inputs and bounds.

    Rows are nearly one-hot with scores around 1e7. The backward's probabilities must match the
    forward's, so dV is close to exact and dK close. dQ is left out as there: with clustered keys of
    norm about 3e3, dQ cancels to far below its terms.
    """
    generator = torch.Generator(device="cuda").manual_seed(2)
    centres = torch.randn(1, 2, 8, 64, device="cuda", generator=generator)
    pick = torch.randint(0, 8, (1, 2, 269), device="cuda", generator=generator)
    base = torch.gather(centres, 2, pick[..., None].expand(-1, -1, -1, 64))
    q = 400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=generator))
    k = 400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=generator))
    v = torch.randn(base.shape, device="cuda", generator=generator)
    grad_out = torch.randn(base.shape, device="cuda", generator=generator)
    qkv, grad_out = [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)

    reference = _dense(qkv, grad_out, 64**-0.5, _visible(269, causal, window), torch.float64)
    results = _kernel(qkv, grad_out, 64**-0.5, causal, window)
    for name, result, ref, bound in zip(("dk", "dv"), results[2:], reference[2:], (0.995, 0.9999)):
        cosine = torch.nn.functional.cosine_similarity(result.double().flatten(), ref.flatten(), dim=0)
        assert cosine > bound, name


@pytest.mark.parametrize("logit_size", [1e2, 1e4, 1e5])
@pytest.mark.parametrize("causal,window", MASKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gradients_track_dense_float32(dtype, causal, window, logit_size):
    """On KohakuFA's benchmark inputs the kernel stays within a few times the error of dense float32 attention.

    Errors that grow with the logits, from how the softmax is recomputed in the backward, show up
    here as factors of tens to hundreds by logits of 1e5.
    """
    qkv, grad_out = _clustered_inputs(logit_size, dtype)
    mask = _visible(grad_out.shape[2], causal, window)
    reference = _dense(qkv, grad_out, 1 / 8, mask, torch.float64)
    plain = _dense(qkv, grad_out, 1 / 8, mask, torch.float32)
    results = _kernel(qkv, grad_out, 1 / 8, causal, window)
    for name, result, ref, base in zip(("dq", "dk", "dv"), results[1:], reference[1:], plain[1:]):
        assert torch.isfinite(result).all(), name
        assert _error(result, ref) <= 2.5 * _error(base.to(dtype), ref) + 1e-6, name


@pytest.mark.parametrize("causal,window", MASKINGS)
def test_float16_small_output_gradients(causal, window):
    """With a small upstream gradient, as under a moderate loss scale, many score gradients dS sit near
    the bottom of float16's normal range.

    Rounding dS to float16 is unavoidable; rounding scale * dS loses log2(1 / scale) more bits. The
    kernel's dQ and dK stay within twice the error of an exact emulation that rounds only dS.
    """
    qkv, grad_out = _clustered_inputs(1e2, torch.float16, jitter=3e-2)
    grad_out = (grad_out.float() * 2.0**-15).half()
    q, k, v = (t.double() for t in qkv)
    mask = _visible(q.shape[2], causal, window)
    reference = _dense(qkv, grad_out, 1 / 8, mask, torch.float64)

    p = ((q @ k.transpose(-1, -2)) / 8).masked_fill(~mask, float("-inf")).softmax(-1)
    dp = grad_out.double() @ v.transpose(-1, -2)
    ds = (p * (dp - (p * dp).sum(-1, keepdim=True))).half().double()
    floor = [((ds @ k) / 8).half(), ((ds.transpose(-1, -2) @ q) / 8).half()]

    results = _kernel(qkv, grad_out, 1 / 8, causal, window)
    for name, result, ref, best in zip(("dq", "dk"), results[1:3], reference[1:3], floor):
        assert _error(result, ref) <= 2 * _error(best, ref), name


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("causal,window", MASKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_scores_far_below_zero(dtype, causal, window, n):
    """Every query's raw scores q . k are about -1.5e8, which softmax does not care about.

    The keys a query does not see must still count for nothing, however low the scores of the keys
    it does see.
    """
    head_dim = 32
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.zeros(1, 2, n, head_dim, device="cuda")
    k = torch.zeros_like(q)
    q[..., 0], q[..., 1] = 16384.0, 1.0
    k[..., 0] = -9216.0
    # Raw scores -150994944 + 16 j are exact in float32; neighbouring ones are about 2.8 apart once scaled.
    k[..., 1] = 16.0 * torch.randint(-8, 9, k.shape[:-1], device="cuda", generator=generator)
    v = torch.randn(q.shape, device="cuda", generator=generator)
    grad_out = torch.randn(q.shape, device="cuda", generator=generator)
    qkv, grad_out = [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)
    sm_scale = head_dim**-0.5

    reference = _dense(qkv, grad_out, sm_scale, _visible(n, causal, window), torch.float64)
    results = _kernel(qkv, grad_out, sm_scale, causal, window)
    # In float32, dq of keys that share an offset 100 times their spread is the limit (about 1e-3).
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-3}[dtype]
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        assert _error(result, ref) <= tolerance, name


@pytest.mark.parametrize("causal,window", MASKINGS)
def test_float32_follows_matmul_precision(causal, window):
    """With float32 inputs the kernel's matrix products follow PyTorch's float32 matmul precision.

    At "highest" its gradients are as accurate as dense float32 attention; at "high" PyTorch allows
    TF32, whose 10-bit mantissa makes them about a thousand times less accurate.
    """
    qkv, grad_out = _clustered_inputs(1e2, torch.float32)
    mask = _visible(grad_out.shape[2], causal, window)
    reference = _dense(qkv, grad_out, 1 / 8, mask, torch.float64)
    plain = _dense(qkv, grad_out, 1 / 8, mask, torch.float32)
    full = _kernel(qkv, grad_out, 1 / 8, causal, window)
    with _float32_matmul_precision("high"):
        tf32 = _kernel(qkv, grad_out, 1 / 8, causal, window)
    for name, result, rounded, ref, base in zip(("out", "dq", "dk", "dv"), full, tf32, reference, plain):
        assert _error(result, ref) <= 2.5 * _error(base, ref), name
        assert _error(rounded, ref) >= 100 * _error(base, ref), name
