# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch

from anemoi.models.layers.icosahedral_grid import IcosahedralNeighbourhood
from anemoi.models.triton.icosahedral_attention import icosahedral_attention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

CASES = [
    # resolution, radius, head dim
    (1, 2, 16),  # every point is an exception
    (2, 1, 16),
    (3, 2, 32),
    (4, 3, 64),
    (4, 5, 32),
    (5, 7, 32),
]


def _qkv(num_points, head_dim, dtype, batch=2, heads=3):
    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (batch, heads, num_points, head_dim)
    return [torch.randn(shape, generator=generator, device="cuda", dtype=dtype) for _ in range(3)]


def _triton(tables, qkv, grad_out, own=None):
    """Output and gradients of the points ``own`` (all by default) attending through one shard."""
    shard = tables.shard(own)
    keys = torch.as_tensor(np.concatenate([np.arange(shard.own.start, shard.own.stop), shard.halo]), device="cuda")
    q, k, v = qkv
    leaves = [q[:, :, shard.own].detach().clone(), k[:, :, keys].detach().clone(), v[:, :, keys].detach().clone()]
    for t in leaves:
        t.requires_grad_(True)
    out = icosahedral_attention(*leaves, shard)
    out.backward(grad_out[:, :, shard.own])
    return out.detach(), keys, [t.grad for t in leaves]


def _reference(tables, qkv, grad_out):
    qkv = [t.detach().double().clone().requires_grad_(True) for t in qkv]
    out = torch.nn.functional.scaled_dot_product_attention(*qkv, attn_mask=tables.mask(device="cuda"))
    out.backward(grad_out.double())
    return [out.detach()] + [t.grad for t in qkv]


def _assert_close(result, reference, tolerance, name):
    scale = reference.abs().max()
    torch.testing.assert_close(result.double() / scale, reference / scale, rtol=tolerance, atol=tolerance, msg=name)


@pytest.mark.parametrize(("resolution", "radius", "head_dim"), CASES)
@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
def test_triton_matches_dense_mask(resolution, radius, head_dim, dtype, tolerance):
    """Forward and gradients agree with the dense-mask reference computed in float64."""
    tables = IcosahedralNeighbourhood(resolution, radius)
    qkv = _qkv(tables.num_points, head_dim, dtype)
    grad_out = torch.randn_like(qkv[0])

    out, _, (dq, dk, dv) = _triton(tables, qkv, grad_out)
    for name, result, reference in zip(("out", "dq", "dk", "dv"), (out, dq, dk, dv), _reference(tables, qkv, grad_out)):
        _assert_close(result, reference, tolerance, name)


@pytest.mark.parametrize("sizes", [[321, 321], [3, 600, 39], [100, 150, 200, 192]])
def test_triton_split_matches_dense_mask(sizes):
    """Each shard's outputs and query gradients, and the key gradients added up over the shards, match the whole grid."""
    tables = IcosahedralNeighbourhood(3, 3)
    assert sum(sizes) == tables.num_points
    qkv = _qkv(tables.num_points, 32, torch.float32)
    grad_out = torch.randn_like(qkv[0])
    out, dq = torch.zeros_like(qkv[0]), torch.zeros_like(qkv[0])
    dk, dv = torch.zeros_like(qkv[1]), torch.zeros_like(qkv[2])
    starts = np.cumsum([0] + sizes)
    for start, stop in zip(starts[:-1], starts[1:]):
        own = slice(int(start), int(stop))
        out_s, keys, (dq_s, dk_s, dv_s) = _triton(tables, qkv, grad_out, own)
        out[:, :, own], dq[:, :, own] = out_s, dq_s
        dk[:, :, keys] += dk_s
        dv[:, :, keys] += dv_s
    for name, result, reference in zip(("out", "dq", "dk", "dv"), (out, dq, dk, dv), _reference(tables, qkv, grad_out)):
        _assert_close(result, reference, 1e-4, name)


def test_triton_reads_point_major_views():
    """Queries, keys and values stored point by point, as views of the model's features, give the same result."""
    tables = IcosahedralNeighbourhood(4, 3)
    batch, heads, head_dim = 2, 4, 32
    features = torch.randn(batch, tables.num_points, 3, heads, head_dim, device="cuda")
    q, k, v = (features[:, :, i].transpose(1, 2) for i in range(3))
    shard = tables.shard()
    torch.testing.assert_close(
        icosahedral_attention(q, k, v, shard),
        icosahedral_attention(q.contiguous(), k.contiguous(), v.contiguous(), shard),
    )
