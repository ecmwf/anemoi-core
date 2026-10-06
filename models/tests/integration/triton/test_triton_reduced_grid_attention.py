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
from anemoi.models.layers.reduced_grid import matching_position

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

CASES = [
    # grid, kernel size, head dim
    ("O8", (5, 7), 16),
    ("O16", (7, 13), 32),
    ("O24", (3, 19), 64),  # window nearly as wide as the polar rows
    ("O16", (9, 3), 32),
    ("O12", (1, 5), 32),
    ("O8", (5, 27), 32),  # window wider than the rows next to the poles
    ("O12", (3, 45), 32),  # window wider than twice the shortest row
    ("H4", (3, 5), 32),  # HEALPix, rings shifted by half a spacing
    ("H8", (5, 13), 64),
    ("H8", (5, 45), 32),  # window wider than the polar rings of 4, 8, ... points
    ("H16", (7, 13), 32),
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


def _qkv(num_points, head_dim, dtype, batch=2, heads=3):
    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (batch, heads, num_points, head_dim)
    return [torch.randn(shape, generator=generator, device="cuda", dtype=dtype) for _ in range(3)]


def _run(backend, grid, kernel_size, qkv, grad_out):
    attention = self_attention(grid, kernel_size, backend=backend)
    qkv = [t.detach().clone().requires_grad_(True) for t in qkv]
    out = attention(*qkv, qkv[0].shape[0])
    out.backward(grad_out)
    return [out.detach()] + [t.grad for t in qkv]


@pytest.mark.parametrize(("grid_name", "kernel_size", "head_dim"), CASES)
@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
def test_triton_matches_dense_mask(grid_name, kernel_size, head_dim, dtype, tolerance):
    """Forward and gradients agree with the dense-mask reference computed in float64."""
    grid = _grid(grid_name)
    qkv = _qkv(grid.num_points, head_dim, dtype)
    grad_out = torch.randn_like(qkv[0])

    results = _run("triton", grid, kernel_size, qkv, grad_out)
    references = _run("sdpa", grid, kernel_size, [t.double() for t in qkv], grad_out.double())
    for name, result, reference in zip(("out", "dq", "dk", "dv"), results, references):
        scale = reference.abs().max()
        torch.testing.assert_close(result.double() / scale, reference / scale, rtol=tolerance, atol=tolerance, msg=name)


@pytest.mark.parametrize("grid_name", ["O16", "H8"])
def test_triton_float32_follows_torch_matmul_precision(grid_name):
    """Float32 inputs multiply in full float32 at "highest" and in TF32 at "high", as PyTorch's matmuls do."""
    grid, kernel_size = _grid(grid_name), (5, 7)
    qkv = _qkv(grid.num_points, 32, torch.float32)
    grad_out = torch.randn_like(qkv[0])
    references = _run("sdpa", grid, kernel_size, [t.double() for t in qkv], grad_out.double())

    full = _run("triton", grid, kernel_size, qkv, grad_out)
    torch.set_float32_matmul_precision("high")
    try:
        tf32 = _run("triton", grid, kernel_size, qkv, grad_out)
    finally:
        torch.set_float32_matmul_precision("highest")
    for name, a, b, reference in zip(("out", "dq", "dk", "dv"), full, tf32, references):
        scale = reference.abs().max()
        torch.testing.assert_close(a.double() / scale, reference / scale, rtol=1e-4, atol=1e-4, msg=name)
        torch.testing.assert_close(b.double() / scale, reference / scale, rtol=1e-2, atol=1e-2, msg=name)
        assert ((a - b).abs().max() / scale).item() > 1e-5, f"{name}: TF32 gave the full float32 result"


def _relative_error(result, reference):
    return ((result.double() - reference).norm() / reference.norm()).item()


@pytest.mark.parametrize("grid_name", ["O16", "H8"])
def test_triton_bfloat16_gradients_with_offset_keys(grid_name):
    """Keys sharing an offset 100 times their spread still give gradients as accurate as bfloat16 allows.

    The exact gradients do not depend on such an offset; rounding the score gradient to bfloat16 before
    the products with the keys and queries would leak it into dQ (https://arxiv.org/abs/2609.34272).
    """
    grid, kernel_size, head_dim = _grid(grid_name), (5, 7), 64
    q, k, v = _qkv(grid.num_points, head_dim, torch.float32)
    offset = torch.randn(head_dim, generator=torch.Generator().manual_seed(1)).cuda()
    k = k + 100 * head_dim**0.5 * offset / offset.norm()
    qkv = [t.bfloat16() for t in (q, k, v)]
    grad_out = torch.randn_like(qkv[0])

    results = _run("triton", grid, kernel_size, qkv, grad_out)
    references = _run("sdpa", grid, kernel_size, [t.double() for t in qkv], grad_out.double())
    for name, result, reference in zip(("dq", "dk", "dv"), results[1:], references[1:]):
        assert _relative_error(result, reference) < 1e-2, name


@pytest.mark.parametrize("grid_name", ["O16", "H8"])
def test_triton_bfloat16_dk_with_nearly_one_hot_attention(grid_name):
    """The winning key retains its gradient when the saved output rounds to its value."""
    grid, kernel_size = _grid(grid_name), (5, 7)
    dense = self_attention(grid, kernel_size, backend="sdpa")
    query = grid.num_points // 2
    keys = dense._mask_on(torch.device("cuda"))[query].nonzero().flatten()[:2]
    q = torch.zeros(1, 1, grid.num_points, 64, device="cuda", dtype=torch.bfloat16)
    k, v, grad_out = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
    q[..., 0] = 8.0
    k[..., 0] = -80.0
    k[..., keys[0], 0], k[..., keys[1], 0] = 0.0, -8.0
    v[..., keys[0], 0] = 1.0
    grad_out[..., query, 0] = 1.0

    # Scaling by 1/sqrt(64) gives logits 0, -8 and -80. The output rounds to 1 in BF16,
    # making uncorrected ds zero at the winning key, whose exact dK is about 3.35e-4.
    qkv = [q, k, v]
    _, _, dk, _ = _run("triton", grid, kernel_size, qkv, grad_out)
    _, _, reference, _ = _run("sdpa", grid, kernel_size, [t.double() for t in qkv], grad_out.double())
    torch.testing.assert_close(dk.double(), reference, rtol=1e-2, atol=1e-7)


@pytest.mark.parametrize("grid_name", ["O16", "H8"])
def test_triton_bfloat16_dq_ignores_a_key_coordinate_the_queries_do_not_see(grid_name):
    """With every query 0 in one coordinate and every key sharing a large value there, dQ is 0 in it."""
    grid, kernel_size = _grid(grid_name), (5, 7)
    q, k, v = _qkv(grid.num_points, 32, torch.float32)
    q[..., 0] = 0.0
    k[..., 0] = 4096.0
    qkv = [t.bfloat16() for t in (q, k, v)]

    _, dq, _, _ = _run("triton", grid, kernel_size, qkv, torch.randn_like(qkv[0]))
    dq = dq.double()
    assert dq[..., 0].abs().max() < 1e-2 * dq.pow(2).mean().sqrt()


@pytest.mark.parametrize("grid_name", ["O16", "H8"])
def test_triton_output_depends_only_on_keys_in_the_window(grid_name):
    """Changing one key and value changes exactly the outputs of the queries whose window holds it."""
    grid, kernel_size = _grid(grid_name), (5, 7)
    q, k, v = _qkv(grid.num_points, 32, torch.float32)
    attention = self_attention(grid, kernel_size, backend="triton")
    out = attention(q, k, v, q.shape[0])

    mask = self_attention(grid, kernel_size, backend="sdpa")._mask_on(torch.device("cuda"))
    for key in (0, int(grid.row_starts[10]) + 3, grid.num_points - 1):
        k2, v2 = k.clone(), v.clone()
        k2[..., key, :] += 10.0
        v2[..., key, :] -= 10.0
        changed = (attention(q, k2, v2, q.shape[0]) - out).abs().amax(dim=(0, 1, 3)) > 0
        assert torch.equal(changed, mask[:, key]), f"key {key}"


def test_triton_on_o96_matches_brute_force():
    """At full O96 size, sampled outputs agree with attention worked out directly from the windows."""
    grid, kernel_size = ReducedGrid.octahedral(96), (7, 13)
    q, k, v = _qkv(grid.num_points, 64, torch.bfloat16, batch=1, heads=4)
    out = self_attention(grid, kernel_size, backend="triton")(q, k, v, q.shape[0])

    rows, positions = (t.cuda() for t in grid.rows_and_positions)
    lengths = torch.tensor(grid.row_lengths, device="cuda")
    starts = grid.row_starts.cuda()
    points = torch.cat(
        [
            torch.randint(0, grid.num_points, (500,), device="cuda"),
            torch.tensor([0, grid.num_points - 1], device="cuda"),
        ]
    )

    first_row = (rows[points] - kernel_size[0] // 2).clamp(0, grid.num_rows - kernel_size[0])
    key_rows = first_row[:, None] + torch.arange(kernel_size[0], device="cuda")  # (points, rows)
    centre = matching_position(positions[points, None], lengths[rows[points], None], lengths[key_rows])
    offsets = torch.arange(-(kernel_size[1] // 2), kernel_size[1] // 2 + 1, device="cuda")
    key_pos = torch.remainder(centre[..., None] + offsets, lengths[key_rows][..., None])
    keys = (starts[key_rows][..., None] + key_pos).reshape(len(points), -1)  # (points, window)

    qp = q[0][:, points].float()
    kp, vp = k[0][:, keys].float(), v[0][:, keys].float()
    scores = torch.einsum("hpd,hpwd->hpw", qp, kp) * 64**-0.5
    expected = torch.einsum("hpw,hpwd->hpd", scores.softmax(-1), vp)
    got = out[0][:, points].float()
    torch.testing.assert_close(got / expected.abs().max(), expected / expected.abs().max(), rtol=2e-2, atol=2e-2)
