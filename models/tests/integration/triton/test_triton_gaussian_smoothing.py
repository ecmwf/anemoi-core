# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton and compiled torch Gaussian smoothing match the eager torch backend."""

import pytest
import torch

from anemoi.models.layers import gaussian_smoothing
from anemoi.models.layers.gaussian_smoothing import CompiledGaussianSmoothing
from anemoi.models.layers.gaussian_smoothing import gaussian_smooth
from anemoi.models.layers.graph_provider import NeighbourListProvider
from anemoi.models.triton.utils import is_triton_available
from tests.grids import octahedral_graph


@pytest.fixture(autouse=True)
def needs_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_triton_available():
        pytest.skip("Triton not available")


def neighbour_tables(n: int, num_neighbours: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Neighbour table and unit vectors of the octahedral grid O``n`` on the GPU."""
    return NeighbourListProvider(octahedral_graph(n), "data", num_neighbours).get_neighbours(
        device=torch.device("cuda")
    )


def forward_backward(
    x: torch.Tensor, neighbours, unit_vectors, sigma: float, backend: str, grad: torch.Tensor, compiled=None
):
    x = x.detach().clone().requires_grad_(True)
    out = gaussian_smooth(x, neighbours, unit_vectors, sigma, backend=backend, compiled=compiled)
    out.backward(grad)
    return out.detach(), x.grad


@pytest.mark.parametrize(("n", "num_neighbours", "channels"), [(8, 9, 5), (24, 32, 70), (48, 128, 37)])
@pytest.mark.parametrize("sigma", [0.01, 0.05])
def test_triton_matches_torch_in_float32(n: int, num_neighbours: int, channels: int, sigma: float) -> None:
    neighbours, unit_vectors = neighbour_tables(n, num_neighbours)
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(3, neighbours.shape[0], channels, device="cuda", generator=generator)
    grad = torch.randn(x.shape, device="cuda", generator=generator)

    out_t, grad_t = forward_backward(x, neighbours, unit_vectors, sigma, "triton", grad)
    out_r, grad_r = forward_backward(x, neighbours, unit_vectors, sigma, "torch", grad)

    torch.testing.assert_close(out_t, out_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_t, grad_r, rtol=1e-5, atol=1e-5)


def test_triton_matches_torch_in_float64() -> None:
    neighbours, unit_vectors = neighbour_tables(16, 24)
    generator = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(2, 1, neighbours.shape[0], 6, device="cuda", dtype=torch.float64, generator=generator)
    grad = torch.randn(x.shape, device="cuda", dtype=torch.float64, generator=generator)

    out_t, grad_t = forward_backward(x, neighbours, unit_vectors, 0.03, "triton", grad)
    out_r, grad_r = forward_backward(x, neighbours, unit_vectors, 0.03, "torch", grad)

    torch.testing.assert_close(out_t, out_r, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_t, grad_r, rtol=1e-12, atol=1e-12)


def test_triton_passes_gradcheck() -> None:
    neighbours, unit_vectors = neighbour_tables(2, 5)
    x = torch.randn(2, neighbours.shape[0], 3, device="cuda", dtype=torch.float64, requires_grad=True)

    # Atomic adds make the backward non-deterministic in the last bits.
    assert torch.autograd.gradcheck(
        lambda field: gaussian_smooth(field, neighbours, unit_vectors, 0.1, backend="triton"),
        (x,),
        nondet_tol=1e-12,
    )


def test_triton_in_bfloat16_matches_float32_to_bfloat16_precision() -> None:
    neighbours, unit_vectors = neighbour_tables(24, 32)
    generator = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(2, neighbours.shape[0], 40, device="cuda", generator=generator)
    grad = torch.randn(x.shape, device="cuda", generator=generator)

    out_b, grad_b = forward_backward(x.bfloat16(), neighbours, unit_vectors, 0.03, "triton", grad.bfloat16())
    out_f, grad_f = forward_backward(x, neighbours, unit_vectors, 0.03, "triton", grad)

    assert out_b.dtype == grad_b.dtype == torch.bfloat16
    torch.testing.assert_close(out_b.float(), out_f, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(grad_b.float(), grad_f, rtol=2e-2, atol=2e-2)


def test_triton_smoothing_of_a_field_with_zero_channels() -> None:
    neighbours, unit_vectors = neighbour_tables(4, 9)
    x = torch.randn(2, neighbours.shape[0], 0, device="cuda", requires_grad=True)

    out = gaussian_smooth(x, neighbours, unit_vectors, 0.1, backend="triton")
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("band_points", [2**19, 1000])
def test_compiled_torch_backend_matches_the_eager_one(band_points: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1000 points per band: O24 (3,168 points) gives three full bands and a shorter one.
    monkeypatch.setattr(gaussian_smoothing, "_COMPILED_BAND_POINTS", band_points)
    neighbours, unit_vectors = neighbour_tables(24, 32)
    generator = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(2, 3, neighbours.shape[0], 40, device="cuda", generator=generator)
    grad = torch.randn(x.shape, device="cuda", generator=generator)
    compiled = CompiledGaussianSmoothing()

    for sigma in (0.02, 0.05):
        out_c, grad_c = forward_backward(x, neighbours, unit_vectors, sigma, "torch", grad, compiled=compiled)
        out_e, grad_e = forward_backward(x, neighbours, unit_vectors, sigma, "torch", grad)

        torch.testing.assert_close(out_c, out_e, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grad_c, grad_e, rtol=1e-5, atol=1e-5)
