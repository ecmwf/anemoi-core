# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gradients of the Triton neighbourhood attention kernels when the attention logits are large.

Large logits make attention nearly one-hot. Flash attention kernels then return gradients that are
wrong by orders of magnitude while the output still looks right, because of how the softmax is
recomputed in float32 in the backward (KohakuBlueleaf, KohakuFA, https://github.com/KohakuBlueleaf/KohakuFA).
Here the neighbourhood self and cross attention kernels are compared with dense attention in float64
on the same connections, and their errors with those of dense attention computed in float32, which is
what correct float32 arithmetic gives.
"""

import math

import pytest
import torch

from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.reduced_grid import ReducedGridCrossNeighbourhoodMask
from anemoi.models.layers.reduced_grid import ReducedGridNeighbourhoodMask
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.reduced_grid_attention import reduced_grid_attention
    from anemoi.models.triton.reduced_grid_cross_attention import reduced_grid_cross_attention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_triton_available(), reason="CUDA and Triton needed"),
]

HEADS = 2
# 1 / sqrt(32) is not a power of two, so scaling the scores rounds them.
HEAD_DIM = 32
SELF_GRID, SELF_KERNEL = ReducedGrid.octahedral(16), (5, 7)
QUERY_GRID, KEY_GRID, CROSS_KERNEL = ReducedGrid.octahedral(8), ReducedGrid.octahedral(16), (5, 5)
KINDS = ["self", "cross"]


def _mask(kind: str) -> torch.Tensor:
    """Which keys each query sees, (queries, keys)."""
    if kind == "cross":
        rule = ReducedGridCrossNeighbourhoodMask(QUERY_GRID, KEY_GRID, CROSS_KERNEL)
        num_q, num_k = QUERY_GRID.num_points, KEY_GRID.num_points
    else:
        rule = ReducedGridNeighbourhoodMask(SELF_GRID, SELF_KERNEL)
        num_q = num_k = SELF_GRID.num_points
    rule = rule.to(torch.device("cuda"))
    q_idx, k_idx = torch.arange(num_q, device="cuda"), torch.arange(num_k, device="cuda")
    return rule(None, None, q_idx[:, None], k_idx[None, :])


def _inputs(kind: str, logit_size: float, dtype: torch.dtype):
    """q, k, v of shape (heads, points, head_dim), scaled so the scaled scores have standard deviation logit_size."""
    mask = _mask(kind)
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(n):
        return torch.randn(HEADS, n, HEAD_DIM, generator=generator, device="cuda", dtype=torch.float64)

    num_q, num_k = mask.shape
    q, k = randn(num_q) * math.sqrt(logit_size), randn(num_k) * math.sqrt(logit_size)
    v, grad_out = randn(num_k), randn(num_q)
    return mask, [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)


def _dense(mask, qkv, grad_out, dtype):
    """Dense attention over ``mask`` computed in ``dtype``; returns out, dq, dk, dv."""
    q, k, v = (t.detach().to(dtype).requires_grad_() for t in qkv)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    out = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
    out.backward(grad_out.to(dtype))
    return [out.detach(), q.grad, k.grad, v.grad]


def _kernel(kind, qkv, grad_out):
    """The Triton kernel for ``kind``; returns out, dq, dk, dv in the layout of ``qkv``."""
    q, k, v = (t.detach().clone().requires_grad_() for t in qkv)
    if kind == "self":
        out = reduced_grid_attention(q[None], k[None], v[None], SELF_GRID, SELF_KERNEL)[0]
    else:
        out = reduced_grid_cross_attention(q[None], k[None], v[None], QUERY_GRID, KEY_GRID, CROSS_KERNEL)[0]
    out.backward(grad_out)
    return [out.detach(), q.grad, k.grad, v.grad]


def _error(result, reference):
    return (result.double() - reference).norm().item()


@pytest.mark.parametrize("kind", KINDS)
def test_two_keys_close_to_a_tie_at_large_scores(kind):
    """One query sees two keys whose scaled scores are about 1e6 apart from zero but only about 1 from each other.

    The raw scores q . k are exact in float32. Subtracting the larger one before scaling keeps their gap
    exact; scaling first would round both scores to steps of about 0.1 and change the second key's
    probability by several percent.
    """
    mask, (q, k, v), grad_out = _inputs(kind, 1.0, torch.float32)
    query = mask.shape[0] // 2
    first, second = mask[query].nonzero().flatten()[:2].tolist()
    q, k = torch.zeros_like(q), torch.zeros_like(k)
    q[:, query, 0] = 1024.0
    k[:, first, 0] = 6400.0  # raw score 6553600, scaled about 1.16e6
    k[:, second, 0] = 6400.0 - 6.0 / 1024  # raw score 6 lower, scaled about 1.06 lower
    qkv = [q, k, v]

    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(kind, qkv, grad_out)
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        scale = ref.abs().max()
        torch.testing.assert_close(result.double() / scale, ref / scale, rtol=0, atol=1e-5, msg=name)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("logit_size", [1e2, 1e3, 1e4, 3e4])
def test_float32_gradients_track_dense_float32(kind, logit_size):
    """Up to logits of a few 1e4 the kernels stay within a few times the error of plain float32 arithmetic.

    Errors that grow with the logits, from how the softmax is recomputed in the backward, would
    show up here as factors of tens to thousands.
    """
    mask, qkv, grad_out = _inputs(kind, logit_size, torch.float32)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    plain = _dense(mask, qkv, grad_out, torch.float32)
    results = _kernel(kind, qkv, grad_out)
    for name, result, ref, base in zip(("out", "dq", "dk", "dv"), results, reference, plain):
        assert torch.isfinite(result).all(), name
        # A floor far below the gradients' size, for when the float32 reference happens to be exact.
        floor = 1e-6 * ref.norm().item()
        assert _error(result, ref) <= 5 * _error(base, ref) + floor, name


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("logit_size", [1e2, 1e4, 3e4])
def test_bfloat16_gradients_at_large_logits(kind, logit_size):
    """With bfloat16 inputs the gradients stay at bfloat16 accuracy as the logits grow."""
    mask, qkv, grad_out = _inputs(kind, logit_size, torch.bfloat16)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(kind, qkv, grad_out)
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        assert _error(result, ref) <= 1e-2 * ref.norm().item(), name


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradients_stay_small_at_extreme_logits(kind, dtype):
    """At logits around 1e6 attention is exactly one-hot and the exact dq and dk nearly vanish.

    No float32 arithmetic recovers them, but the kernels must not blow them up: their errors stay
    far below the size of the gradient of the values.
    """
    mask, qkv, grad_out = _inputs(kind, 1e6, dtype)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(kind, qkv, grad_out)
    size = reference[3].norm().item()
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        assert torch.isfinite(result).all(), name
        assert _error(result, ref) <= 1e-2 * size, name
