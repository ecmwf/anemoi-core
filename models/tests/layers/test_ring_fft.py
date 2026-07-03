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
from torch.utils.cpp_extension import is_ninja_available

from anemoi.models.layers.ring_fft import ring_irfft
from anemoi.models.layers.ring_fft import ring_rfft
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform

O96_NLAT = 192
O96_GRID_POINTS = 40320
EXTENSION_BACKENDS = ("direct", "cufft")
REPRESENTATIVE_NLON_TEST_VALUES = [
    20,
    24,
    28,
    240,
    360,
    480,
    720,
    864,
    900,
    960,
    1000,
    1024,
    1080,
    1152,
    1200,
    1280,
    1292,
    1296,
]


def _o96_lons_per_lat() -> list[int]:
    lons = [20 + 4 * i for i in range(O96_NLAT // 2)]
    return lons + list(reversed(lons))


def _o320_lons_per_lat() -> list[int]:
    lons = [20 + 4 * i for i in range(320)]
    return lons + list(reversed(lons))


def _n320_lons_per_lat() -> list[int]:
    lookup = pytest.importorskip("anemoi.transform.grids.named").lookup
    lats = lookup("n320")["latitudes"]
    unique_lats = sorted(set(lats))
    return [int((lats == unique_lat).sum()) for unique_lat in unique_lats]


def _reference_ring_rfft(x: torch.Tensor, lons_per_lat: list[int], truncation: int) -> torch.Tensor:
    grid_points = sum(lons_per_lat)
    x_flat = x.reshape(-1, grid_points)
    max_m = max(lons_per_lat) // 2 + 1
    out = torch.zeros(
        x_flat.shape[0],
        len(lons_per_lat),
        max_m,
        device=x.device,
        dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128,
    )

    groups: dict[int, tuple[list[int], list[int]]] = {}
    offset = 0
    for lat, nlon in enumerate(lons_per_lat):
        ring_indices, ring_offsets = groups.setdefault(nlon, ([], []))
        ring_indices.append(lat)
        ring_offsets.append(offset)
        offset += nlon

    for nlon, (ring_indices, ring_offsets) in groups.items():
        batch = torch.stack([x_flat[:, offset : offset + nlon] for offset in ring_offsets], dim=-2)
        ring_fft = torch.fft.rfft(batch, norm="forward")
        nmodes = ring_fft.shape[-1]
        out[:, ring_indices, :nmodes] = ring_fft

    return out.reshape(*x.shape[:-1], len(lons_per_lat), max_m)


def _reference_ring_irfft(x: torch.Tensor, lons_per_lat: list[int]) -> torch.Tensor:
    grid_points = sum(lons_per_lat)
    nlat = len(lons_per_lat)
    nmodes = x.shape[-1]
    x_flat = x.reshape(-1, nlat, nmodes)
    rings = [torch.fft.irfft(x_flat[:, lat, :], nlon, norm="forward") for lat, nlon in enumerate(lons_per_lat)]
    out = torch.cat(rings, dim=-1)
    return out.reshape(*x.shape[:-2], grid_points)


def _require_cuda_extension_environment() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_ninja_available():
        pytest.skip("Ninja is required to build the ring FFT CUDA extension")


def _o96_rfft_tolerances() -> tuple[float, float]:
    return 1e-6, 1e-7


def _o96_irfft_tolerances() -> tuple[tuple[float, float], tuple[float, float]]:
    return (1e-5, 1e-5), (1e-5, 3e-5)


def _large_grid_rfft_tolerances() -> tuple[float, float]:
    return 1e-6, 1e-7


def _large_grid_irfft_tolerances() -> tuple[float, float]:
    return 1e-5, 5e-5


@pytest.mark.gpu
@pytest.mark.parametrize("truncation", [95, 191])
@pytest.mark.parametrize("lead", [1, 3])
def test_o96_rfft_forward_matches_torch(truncation: int, lead: int) -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(0)
    x = torch.randn(lead, O96_GRID_POINTS, device="cuda", dtype=torch.float32)

    actual = ring_rfft(x, _o96_lons_per_lat(), truncation)
    expected = _reference_ring_rfft(x, _o96_lons_per_lat(), truncation)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("truncation", [95, 191])
def test_o96_rfft_backward_matches_torch(truncation: int) -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(1)
    x_actual = torch.randn(2, O96_GRID_POINTS, device="cuda", dtype=torch.float32, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_rfft(x_actual, _o96_lons_per_lat(), truncation)
    expected = _reference_ring_rfft(x_expected, _o96_lons_per_lat(), truncation)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual.real * grad_output.real + actual.imag * grad_output.imag).sum()
    expected_loss = (expected.real * grad_output.real + expected.imag * grad_output.imag).sum()

    actual_loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=2e-5, atol=5e-6)


@pytest.mark.gpu
def test_small_float64_known_values_high_precision() -> None:
    _require_cuda_extension_environment()

    lons_per_lat = [4, 8]
    x = torch.zeros(1, sum(lons_per_lat), device="cuda", dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[0, 0] = 1
        x[0, lons_per_lat[0]] = 1

    actual = ring_rfft(x, lons_per_lat, 4)
    expected = torch.zeros_like(actual)
    expected[0, 0, :3] = 0.25
    expected[0, 1, :] = 0.125
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    grad_output = torch.zeros_like(actual)
    grad_output[..., 0] = 1
    actual.backward(grad_output)

    expected_grad = torch.empty_like(x)
    expected_grad[0, : lons_per_lat[0]] = 0.25
    expected_grad[0, lons_per_lat[0] :] = 0.125
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.gpu
def test_small_irfft_float64_known_values_high_precision() -> None:
    _require_cuda_extension_environment()

    lons_per_lat = [4, 8]
    x = torch.zeros(1, len(lons_per_lat), 5, device="cuda", dtype=torch.complex128, requires_grad=True)
    with torch.no_grad():
        x[0, :, 0] = 1

    actual = ring_irfft(x, lons_per_lat)
    expected = torch.ones_like(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    grad_output = torch.ones_like(actual)
    actual.backward(grad_output)

    expected_grad = torch.zeros_like(x)
    expected_grad[0, 0, 0] = lons_per_lat[0]
    expected_grad[0, 1, 0] = lons_per_lat[1]
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=1e-14)


@pytest.mark.gpu
def test_irfft_ignores_self_conjugate_imaginary_parts_like_torch() -> None:
    _require_cuda_extension_environment()

    lons_per_lat = [4, 5, 8]
    torch.manual_seed(17)
    x = torch.randn(2, len(lons_per_lat), 5, device="cuda", dtype=torch.complex64, requires_grad=True)
    x_perturbed = x.detach().clone()
    with torch.no_grad():
        x_perturbed[:, :, 0] = x_perturbed[:, :, 0].real + 12345j
        for lat_idx, nlon in enumerate(lons_per_lat):
            if nlon % 2 == 0:
                x_perturbed[:, lat_idx, nlon // 2] = x_perturbed[:, lat_idx, nlon // 2].real - 23456j
    x_perturbed.requires_grad_(True)

    actual = ring_irfft(x, lons_per_lat)
    actual_perturbed = ring_irfft(x_perturbed, lons_per_lat)
    expected = _reference_ring_irfft(x.detach(), lons_per_lat)
    expected_perturbed = _reference_ring_irfft(x_perturbed.detach(), lons_per_lat)
    grad_output = torch.randn_like(expected)

    actual.backward(grad_output)
    actual_perturbed.backward(grad_output)

    torch.testing.assert_close(expected_perturbed, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_perturbed, actual, rtol=0, atol=1e-6)
    torch.testing.assert_close(x_perturbed.grad, x.grad, rtol=0, atol=1e-6)
    torch.testing.assert_close(x.grad[:, :, 0].imag, torch.zeros_like(x.grad[:, :, 0].imag), rtol=0, atol=1e-12)
    for lat_idx, nlon in enumerate(lons_per_lat):
        if nlon % 2 == 0:
            torch.testing.assert_close(
                x.grad[:, lat_idx, nlon // 2].imag,
                torch.zeros_like(x.grad[:, lat_idx, nlon // 2].imag),
                rtol=0,
                atol=1e-12,
            )


@pytest.mark.gpu
@pytest.mark.parametrize("nlon", REPRESENTATIVE_NLON_TEST_VALUES)
def test_generic_backend_representative_lengths_known_values_high_precision(
    nlon: int,
) -> None:
    _require_cuda_extension_environment()

    lons_per_lat = [nlon]
    x = torch.zeros(1, nlon, device="cuda", dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[0, 0] = 1

    actual = ring_rfft(x, lons_per_lat, nlon // 2)
    expected = torch.full_like(actual, 1 / nlon)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)

    grad_output = torch.zeros_like(actual)
    grad_output[..., 0] = 1
    actual.backward(grad_output)

    expected_grad = torch.full_like(x, 1 / nlon)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=1e-15)


@pytest.mark.gpu
@pytest.mark.parametrize("nlon", REPRESENTATIVE_NLON_TEST_VALUES)
def test_generic_backend_irfft_representative_lengths_known_values_high_precision(
    nlon: int,
) -> None:
    _require_cuda_extension_environment()

    lons_per_lat = [nlon]
    x = torch.zeros(1, 1, nlon // 2 + 1, device="cuda", dtype=torch.complex128, requires_grad=True)
    with torch.no_grad():
        x[0, 0, 0] = 1

    actual = ring_irfft(x, lons_per_lat)
    expected = torch.ones_like(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

    grad_output = torch.ones_like(actual)
    actual.backward(grad_output)

    expected_grad = torch.zeros_like(x)
    expected_grad[0, 0, 0] = nlon
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=3e-11)


@pytest.mark.gpu
@pytest.mark.parametrize("truncation", [95, 191])
def test_o96_sht_fast_path_matches_grouped_torch_path(truncation: int) -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(2)
    lons_per_lat = _o96_lons_per_lat()
    fast = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).cuda()
    grouped = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).cuda()
    grouped._use_cuda_ring_rfft = False

    x = torch.randn(2, O96_GRID_POINTS, device="cuda", dtype=torch.float32)

    torch.testing.assert_close(fast(x), grouped(x), rtol=2e-5, atol=5e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("truncation", [95, 191])
def test_o96_inverse_sht_fast_path_matches_torch_path(truncation: int) -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(12)
    lons_per_lat = _o96_lons_per_lat()
    fast = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).cuda()
    grouped = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).cuda()
    grouped._use_cuda_ring_irfft = False

    x = torch.randn(2, truncation + 1, truncation + 1, device="cuda", dtype=torch.complex64)

    torch.testing.assert_close(fast(x), grouped(x), rtol=3e-4, atol=3e-4)


@pytest.mark.gpu
def test_o96_inverse_sht_simple_float64_fast_path_matches_torch_path() -> None:
    _require_cuda_extension_environment()

    truncation = 95
    lons_per_lat = _o96_lons_per_lat()
    fast = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).cuda()
    grouped = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).cuda()
    grouped._use_cuda_ring_irfft = False

    x = torch.zeros(1, truncation + 1, truncation + 1, device="cuda", dtype=torch.complex128)
    x[0, 0, 0] = 1.0

    torch.testing.assert_close(fast(x), grouped(x), rtol=0, atol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape_prefix", [(2,), (2, 2, 3)])
def test_o96_rfft_realistic_shapes_and_dtypes(dtype: torch.dtype, shape_prefix: tuple[int, ...]) -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(3)
    shape = (*shape_prefix, O96_GRID_POINTS)
    x_actual = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_rfft(x_actual, _o96_lons_per_lat(), 95)
    expected = _reference_ring_rfft(x_expected, _o96_lons_per_lat(), 95)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual.real * grad_output.real + actual.imag * grad_output.imag).sum()
    expected_loss = (expected.real * grad_output.real + expected.imag * grad_output.imag).sum()

    actual_loss.backward()
    expected_loss.backward()

    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
        torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=2e-5, atol=5e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=1e-12, atol=1e-13)


@pytest.mark.gpu
def test_o96_rfft_backend_forward_backward_matches_torch() -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(23)
    x_actual = torch.randn(2, O96_GRID_POINTS, device="cuda", dtype=torch.float32, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_rfft(x_actual, _o96_lons_per_lat(), 95)
    expected = _reference_ring_rfft(x_expected, _o96_lons_per_lat(), 95)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual.real * grad_output.real + actual.imag * grad_output.imag).sum()
    expected_loss = (expected.real * grad_output.real + expected.imag * grad_output.imag).sum()

    actual_loss.backward()
    expected_loss.backward()

    rtol, atol = _o96_rfft_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape_prefix", [(2,), (2, 2, 3)])
def test_o96_irfft_realistic_shapes_and_dtypes(dtype: torch.dtype, shape_prefix: tuple[int, ...]) -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(13)
    lons_per_lat = _o96_lons_per_lat()
    shape = (*shape_prefix, len(lons_per_lat), 96)
    x_actual = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_irfft(x_actual, lons_per_lat)
    expected = _reference_ring_irfft(x_expected, lons_per_lat)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual * grad_output).sum()
    expected_loss = (expected * grad_output).sum()

    actual_loss.backward()
    expected_loss.backward()

    if dtype == torch.complex64:
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
        torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=5e-4, atol=8e-4)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-11)
        torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=1e-12, atol=1e-10)


@pytest.mark.gpu
def test_o96_irfft_backend_forward_backward_matches_torch() -> None:
    _require_cuda_extension_environment()

    torch.manual_seed(24)
    lons_per_lat = _o96_lons_per_lat()
    x_actual = torch.randn(2, len(lons_per_lat), 96, device="cuda", dtype=torch.complex64, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_irfft(x_actual, lons_per_lat)
    expected = _reference_ring_irfft(x_expected, lons_per_lat)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual * grad_output).sum()
    expected_loss = (expected * grad_output).sum()

    actual_loss.backward()
    expected_loss.backward()

    output_tolerances, grad_tolerances = _o96_irfft_tolerances()
    torch.testing.assert_close(actual, expected, rtol=output_tolerances[0], atol=output_tolerances[1])
    torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=grad_tolerances[0], atol=grad_tolerances[1])


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("grid_name", "lons_per_lat_fn", "truncation"),
    [
        ("o320", _o320_lons_per_lat, 319),
        ("o320", _o320_lons_per_lat, 319),
        ("o320", _o320_lons_per_lat, 639),
        ("o320", _o320_lons_per_lat, 639),
        ("n320", _n320_lons_per_lat, 319),
        ("n320", _n320_lons_per_lat, 319),
        ("n320", _n320_lons_per_lat, 639),
        ("n320", _n320_lons_per_lat, 639),
    ],
)
def test_large_grid_ring_rfft_forward_backward_matches_torch(
    grid_name: str,
    lons_per_lat_fn,
    truncation: int,
) -> None:
    _require_cuda_extension_environment()

    del grid_name

    torch.manual_seed(4)
    lons_per_lat = lons_per_lat_fn()
    grid_points = sum(lons_per_lat)
    x_actual = torch.randn(1, grid_points, device="cuda", dtype=torch.float32, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_rfft(x_actual, lons_per_lat, truncation)
    expected = _reference_ring_rfft(x_expected, lons_per_lat, truncation)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual.real * grad_output.real + actual.imag * grad_output.imag).sum()
    expected_loss = (expected.real * grad_output.real + expected.imag * grad_output.imag).sum()

    actual_loss.backward()
    expected_loss.backward()

    rtol, atol = _large_grid_rfft_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("grid_name", "lons_per_lat_fn", "truncation"),
    [
        ("o320", _o320_lons_per_lat, 319),
        ("o320", _o320_lons_per_lat, 319),
        ("o320", _o320_lons_per_lat, 639),
        ("o320", _o320_lons_per_lat, 639),
        ("n320", _n320_lons_per_lat, 319),
        ("n320", _n320_lons_per_lat, 319),
        ("n320", _n320_lons_per_lat, 639),
        ("n320", _n320_lons_per_lat, 639),
    ],
)
def test_large_grid_ring_irfft_forward_backward_matches_torch(
    grid_name: str,
    lons_per_lat_fn,
    truncation: int,
) -> None:
    _require_cuda_extension_environment()

    del grid_name

    torch.manual_seed(14)
    lons_per_lat = lons_per_lat_fn()
    x_actual = torch.randn(
        1,
        len(lons_per_lat),
        truncation + 1,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=True,
    )
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ring_irfft(x_actual, lons_per_lat)
    expected = _reference_ring_irfft(x_expected, lons_per_lat)
    grad_output = torch.randn_like(expected)

    actual_loss = (actual * grad_output).sum()
    expected_loss = (expected * grad_output).sum()

    actual_loss.backward()
    expected_loss.backward()

    rtol, atol = _large_grid_irfft_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_actual.grad, x_expected.grad, rtol=rtol, atol=atol)
