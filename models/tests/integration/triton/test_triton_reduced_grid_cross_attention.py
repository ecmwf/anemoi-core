# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttentionWrapper
from anemoi.models.layers.reduced_grid import ReducedGrid

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

CASES = [
    # query grid, key grid, kernel size, head dim
    ("O8", "O16", (5, 7), 16),  # queries on the coarser grid, as in an encoder
    ("O16", "O8", (3, 5), 32),  # queries on the finer grid, as in a decoder
    ("O12", "O24", (7, 27), 32),  # window wider than the 20-point rows next to the poles
    ("O24", "O6", (3, 5), 64),  # four times finer queries
    ("O16", "O16", (7, 13), 32),  # the same grid on both sides
    ("O16", "O8", (3, 61), 32),  # window wider than three polar rows of the key grid
    ("H4", "H8", (5, 7), 32),  # HEALPix to HEALPix
    ("H16", "H4", (3, 5), 32),
    ("H8", "O8", (3, 5), 32),  # HEALPix queries, octahedral keys
    ("O8", "H8", (5, 13), 64),  # octahedral queries, HEALPix keys
]


def _family(grid: ReducedGrid) -> str:
    return "healpix" if grid.is_shifted else "octahedral"


def self_attention(grid: ReducedGrid, kernel_size, backend: str) -> NeighbourhoodAttentionWrapper:
    """Neighbourhood self attention on ``grid``, with the points in grid order."""
    neighbourhood = GridNeighbourhood(_family(grid), tuple(kernel_size), backend, grid, grid, is_self_attention=True)
    return NeighbourhoodAttentionWrapper(neighbourhood)


def cross_attention(
    query_grid: ReducedGrid, key_grid: ReducedGrid, kernel_size, backend: str
) -> NeighbourhoodAttentionWrapper:
    """Neighbourhood cross attention from ``query_grid`` to ``key_grid``, with the points in grid order."""
    neighbourhood = GridNeighbourhood(_family(key_grid), tuple(kernel_size), backend, query_grid, key_grid)
    return NeighbourhoodAttentionWrapper(neighbourhood)


def _grid(name: str) -> ReducedGrid:
    """``"O16"`` is the octahedral grid O16, ``"H8"`` the HEALPix grid with nside 8 in ring ordering."""
    return ReducedGrid.octahedral(int(name[1:])) if name[0] == "O" else ReducedGrid.healpix(int(name[1:]))


def _tensors(query_grid, key_grid, head_dim, dtype, batch=2, heads=3):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(batch, heads, query_grid.num_points, head_dim, generator=generator, device="cuda", dtype=dtype)
    k, v = (
        torch.randn(batch, heads, key_grid.num_points, head_dim, generator=generator, device="cuda", dtype=dtype)
        for _ in range(2)
    )
    return q, k, v


def _run(attention, qkv, grad_out):
    qkv = [t.detach().clone().requires_grad_(True) for t in qkv]
    out = attention(*qkv, qkv[0].shape[0])
    out.backward(grad_out)
    return [out.detach()] + [t.grad for t in qkv]


@pytest.mark.parametrize(("query_name", "key_name", "kernel_size", "head_dim"), CASES)
@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
def test_triton_matches_dense_mask(query_name, key_name, kernel_size, head_dim, dtype, tolerance):
    """Forward and gradients agree with the dense-mask reference computed in float64."""
    query_grid, key_grid = _grid(query_name), _grid(key_name)
    qkv = _tensors(query_grid, key_grid, head_dim, dtype)
    grad_out = torch.randn_like(qkv[0])

    triton = cross_attention(query_grid, key_grid, kernel_size, backend="triton")
    dense = cross_attention(query_grid, key_grid, kernel_size, backend="sdpa")
    results = _run(triton, qkv, grad_out)
    references = _run(dense, [t.double() for t in qkv], grad_out.double())
    for name, result, reference in zip(("out", "dq", "dk", "dv"), results, references):
        scale = reference.abs().max()
        torch.testing.assert_close(result.double() / scale, reference / scale, rtol=tolerance, atol=tolerance, msg=name)


@pytest.mark.parametrize(
    ("query_name", "key_name", "kernel_size"), [("O8", "O16", (5, 5)), ("O16", "O8", (3, 5)), ("H8", "H4", (3, 5))]
)
def test_triton_bfloat16_gradients_with_offset_keys(query_name, key_name, kernel_size):
    """Keys sharing an offset 100 times their spread still give gradients as accurate as bfloat16 allows.

    The exact gradients do not depend on such an offset; rounding the score gradient to bfloat16 before
    the products with the keys and queries would leak it into dQ (https://arxiv.org/abs/2609.34272).
    """
    query_grid, key_grid, head_dim = _grid(query_name), _grid(key_name), 64
    q, k, v = _tensors(query_grid, key_grid, head_dim, torch.float32)
    offset = torch.randn(head_dim, generator=torch.Generator().manual_seed(1)).cuda()
    k = k + 100 * head_dim**0.5 * offset / offset.norm()
    qkv = [t.bfloat16() for t in (q, k, v)]
    grad_out = torch.randn_like(qkv[0])

    results = _run(cross_attention(query_grid, key_grid, kernel_size, backend="triton"), qkv, grad_out)
    dense = cross_attention(query_grid, key_grid, kernel_size, backend="sdpa")
    references = _run(dense, [t.double() for t in qkv], grad_out.double())
    for name, result, reference in zip(("dq", "dk", "dv"), results[1:], references[1:]):
        assert ((result.double() - reference).norm() / reference.norm()).item() < 1e-2, name


@pytest.mark.parametrize("grid_name", ["O16", "H8"])
def test_cross_on_one_grid_matches_self_attention_kernel(grid_name):
    grid = _grid(grid_name)
    qkv = _tensors(grid, grid, 32, torch.float32)
    grad_out = torch.randn_like(qkv[0])
    cross = _run(cross_attention(grid, grid, (5, 7), backend="triton"), qkv, grad_out)
    own = _run(self_attention(grid, (5, 7), backend="triton"), qkv, grad_out)
    for name, a, b in zip(("out", "dq", "dk", "dv"), cross, own):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize(("query_name", "key_name"), [("O8", "O16"), ("O16", "O8"), ("H8", "O8"), ("O8", "H8")])
def test_output_depends_only_on_keys_in_the_window(query_name, key_name):
    """Changing one key and value changes exactly the outputs of the queries whose window holds it."""
    query_grid, key_grid, kernel_size = _grid(query_name), _grid(key_name), (5, 7)
    q, k, v = _tensors(query_grid, key_grid, 32, torch.float32)
    attention = cross_attention(query_grid, key_grid, kernel_size, backend="triton")
    mask = cross_attention(query_grid, key_grid, kernel_size, backend="sdpa")._mask_on(torch.device("cuda"))
    out = attention(q, k, v, q.shape[0])
    for key in (0, key_grid.num_points // 2 + 5, key_grid.num_points - 1):
        k2, v2 = k.clone(), v.clone()
        k2[..., key, :] += 10.0
        v2[..., key, :] -= 10.0
        changed = (attention(q, k2, v2, q.shape[0]) - out).abs().amax(dim=(0, 1, 3)) > 0
        assert torch.equal(changed, mask[:, key]), f"key {key}"
