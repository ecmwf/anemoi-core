# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gradients of the Triton graph transformer attention when the attention logits are large.

Large logits make attention nearly one-hot. Flash attention kernels then return gradients that are
wrong by orders of magnitude while the output still looks right, because of how the softmax is
recomputed in float32 in the backward (KohakuBlueleaf, KohakuFA, https://github.com/KohakuBlueleaf/KohakuFA).
Here the kernel is compared with dense attention in float64 on the same connections, and its errors
with those of dense attention computed in float32, which is what correct float32 arithmetic gives.
"""

import math

import pytest
import torch

from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.gt import graph_transformer_attention_conv

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_triton_available(), reason="CUDA and Triton needed"),
]

HEADS = 2
# 1 / sqrt(32) is not a power of two, so scaling the scores rounds them.
HEAD_DIM = 32
NUM_NODES = 1600
# Each node attends to the 35 nodes around it on a ring.
NEIGHBOURS = torch.arange(-17, 18)


def _mask() -> torch.Tensor:
    """Which source nodes each destination node sees, (destinations, sources)."""
    dst = torch.arange(NUM_NODES, device="cuda")
    src = torch.remainder(dst[:, None] + NEIGHBOURS.to("cuda"), NUM_NODES)
    mask = torch.zeros(NUM_NODES, NUM_NODES, dtype=torch.bool, device="cuda")
    mask[dst[:, None], src] = True
    return mask


def _inputs(logit_size: float, dtype: torch.dtype):
    """q, k, v of shape (heads, nodes, head_dim), scaled so the scaled scores have standard deviation logit_size."""
    mask = _mask()
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn():
        return torch.randn(HEADS, NUM_NODES, HEAD_DIM, generator=generator, device="cuda", dtype=torch.float64)

    q, k = randn() * math.sqrt(logit_size), randn() * math.sqrt(logit_size)
    v, grad_out = randn(), randn()
    return mask, [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)


def _dense(mask, qkv, grad_out, dtype):
    """Dense attention over ``mask`` computed in ``dtype``; returns out, dq, dk, dv."""
    q, k, v = (t.detach().to(dtype).requires_grad_() for t in qkv)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    out = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
    out.backward(grad_out.to(dtype))
    return [out.detach(), q.grad, k.grad, v.grad]


def _kernel(mask, qkv, grad_out):
    """The Triton kernel, one edge per pair the mask allows and zero edge features; returns out, dq, dk, dv."""
    q, k, v = (t.detach().clone().requires_grad_() for t in qkv)
    dst, src = mask.nonzero(as_tuple=True)
    csc, _, reverse = edge_index_to_csc(
        torch.stack([src, dst]), num_nodes=(NUM_NODES, NUM_NODES), reverse=True, edges_are_dst_sorted=True
    )
    edges = torch.zeros(len(src), HEADS, HEAD_DIM, device="cuda", dtype=q.dtype)
    nodes_first = [t.transpose(0, 1) for t in (q, k, v)]
    out = graph_transformer_attention_conv(*nodes_first, edges, csc, reverse).transpose(0, 1)
    out.backward(grad_out)
    return [out.detach(), q.grad, k.grad, v.grad]


def _error(result, reference):
    return (result.double() - reference).norm().item()


def test_two_keys_close_to_a_tie_at_large_scores():
    """One node sees two keys whose scaled scores are about 1e6 apart from zero but only about 1 from each other.

    The raw scores q . k are exact in float32. Subtracting the larger one before scaling keeps their gap
    exact; scaling first would round both scores to steps of about 0.1 and change the second key's
    probability by several percent.
    """
    mask, (q, k, v), grad_out = _inputs(1.0, torch.float32)
    node = NUM_NODES // 2
    first, second = mask[node].nonzero().flatten()[:2].tolist()
    q, k = torch.zeros_like(q), torch.zeros_like(k)
    q[:, node, 0] = 1024.0
    k[:, first, 0] = 6400.0  # raw score 6553600, scaled about 1.16e6
    k[:, second, 0] = 6400.0 - 6.0 / 1024  # raw score 6 lower, scaled about 1.06 lower
    qkv = [q, k, v]

    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(mask, qkv, grad_out)
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        scale = ref.abs().max()
        torch.testing.assert_close(result.double() / scale, ref / scale, rtol=0, atol=1e-5, msg=name)


@pytest.mark.parametrize("logit_size", [1e2, 1e3, 1e4, 3e4])
def test_float32_gradients_track_dense_float32(logit_size):
    """Up to logits of a few 1e4 the kernel stays within a few times the error of plain float32 arithmetic.

    Errors that grow with the logits, from how the softmax is recomputed in the backward, would
    show up here as factors of tens to thousands.
    """
    mask, qkv, grad_out = _inputs(logit_size, torch.float32)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    plain = _dense(mask, qkv, grad_out, torch.float32)
    results = _kernel(mask, qkv, grad_out)
    for name, result, ref, base in zip(("out", "dq", "dk", "dv"), results, reference, plain):
        assert torch.isfinite(result).all(), name
        # A floor far below the gradients' size, for when the float32 reference happens to be exact.
        floor = 1e-6 * ref.norm().item()
        assert _error(result, ref) <= 5 * _error(base, ref) + floor, name


@pytest.mark.parametrize("logit_size", [1e2, 1e4, 3e4])
def test_bfloat16_gradients_at_large_logits(logit_size):
    """With bfloat16 inputs the gradients stay at bfloat16 accuracy as the logits grow."""
    mask, qkv, grad_out = _inputs(logit_size, torch.bfloat16)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(mask, qkv, grad_out)
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        assert _error(result, ref) <= 1e-2 * ref.norm().item(), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradients_stay_small_at_extreme_logits(dtype):
    """At logits around 1e6 attention is exactly one-hot and the exact dq and dk nearly vanish.

    No float32 arithmetic recovers them, but the kernel must not blow them up: its errors stay
    far below the size of the gradient of the values.
    """
    mask, qkv, grad_out = _inputs(1e6, dtype)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(mask, qkv, grad_out)
    size = reference[3].norm().item()
    for name, result, ref in zip(("out", "dq", "dk", "dv"), results, reference):
        assert torch.isfinite(result).all(), name
        assert _error(result, ref) <= 1e-2 * size, name


def _clustered_inputs(logit_size: float, jitter: float, dtype: torch.dtype, clusters: int = 16):
    """Queries and keys around a few shared directions, so most rows are nearly one-hot and some keys nearly tie.

    The construction of KohakuFA's precision benchmark: every query and key picks one of ``clusters`` unit
    directions, adds Gaussian noise of size ``jitter`` and is scaled so the largest scaled logit is about
    ``logit_size``.
    """
    mask = _mask()
    generator = torch.Generator(device="cuda").manual_seed(0)
    centers = torch.randn(HEADS, clusters, HEAD_DIM, generator=generator, device="cuda", dtype=torch.float64)
    centers = centers / centers.norm(dim=-1, keepdim=True)
    radius = math.sqrt(logit_size * math.sqrt(HEAD_DIM))

    def around():
        pick = torch.randint(0, clusters, (HEADS, NUM_NODES), generator=generator, device="cuda")
        base = torch.gather(centers, 1, pick[..., None].expand(-1, -1, HEAD_DIM))
        noise = torch.randn(base.shape, generator=generator, device="cuda", dtype=torch.float64)
        return radius * (base + jitter * noise)

    q, k = around(), around()
    v = torch.randn(HEADS, NUM_NODES, HEAD_DIM, generator=generator, device="cuda", dtype=torch.float64)
    grad_out = torch.randn(HEADS, NUM_NODES, HEAD_DIM, generator=generator, device="cuda", dtype=torch.float64)
    return mask, [t.to(dtype) for t in (q, k, v)], grad_out.to(dtype)


def _floor(mask, qkv, grad_out):
    """dq and dk of exact attention from float32-rounded raw scores, rounded to the input precision.

    The best a kernel that adds up the scores in float32 can do: at large logits the rounding of the
    scores alone moves near-tie probabilities.
    """
    q, k, v = (t.double() for t in qkv)
    scores = (q @ k.transpose(-1, -2)).float().double() / math.sqrt(HEAD_DIM)
    p = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    dp = grad_out.double() @ v.transpose(-1, -2)
    ds = p * (dp - (p * dp).sum(-1, keepdim=True)) / math.sqrt(HEAD_DIM)
    return (ds @ k).to(qkv[0].dtype), (ds.transpose(-1, -2) @ q).to(qkv[0].dtype)


@pytest.mark.parametrize(("dtype", "logit_size"), [(torch.bfloat16, 1e5), (torch.bfloat16, 1e6), (torch.float16, 1e5)])
def test_nearly_one_hot_gradients_reach_the_float32_floor(dtype, logit_size):
    """With nearly one-hot rows, dq and dk stay within a few times the best that float32 scores allow.

    The backward forms D = dO . O from the saved output. Its rounding error, times the weighted mean
    key, would dominate dq and dk here if the score gradients were not corrected to sum to zero.
    """
    mask, qkv, grad_out = _clustered_inputs(logit_size, 3e-2, dtype)
    reference = _dense(mask, qkv, grad_out, torch.float64)
    results = _kernel(mask, qkv, grad_out)
    for name, result, ref, floor in zip(("dq", "dk"), results[1:3], reference[1:3], _floor(mask, qkv, grad_out)):
        assert torch.isfinite(result).all(), name
        assert _error(result, ref) <= 3 * _error(floor, ref), name
