# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io

import numpy as np
import pytest
import torch

from anemoi.models.layers import spectral_helpers
from anemoi.models.layers.spectral_helpers import LEGENDRE_TABLES
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import first_orders_of_blocks
from anemoi.models.layers.spectral_helpers import legendre_by_degree
from anemoi.models.layers.spectral_helpers import legendre_gauss_weights
from anemoi.models.layers.spectral_transforms import InverseOctahedralSHT
from anemoi.models.layers.spectral_transforms import InverseReducedSHT
from anemoi.models.layers.spectral_transforms import OctahedralSHT
from anemoi.models.layers.spectral_transforms import ReducedSHT

"""
Random array of complex spectral coefficients.

By definition arranged on an upper triangular matrix of width and height (truncation + 1), but with
values below the diagonal just set to zero. The m = 0 coefficients are also purely real, to ensure
that inverse transformed fields are also real.
"""


def random_spectral_array(truncation: int, dtype: torch.dtype) -> torch.Tensor:
    # Shape: [batch index, ensemble member, l, m]
    shape = (1, 1, truncation + 1, truncation + 1)
    spectral_array = torch.complex(torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype))
    spectral_array[0, 0, :, 0].imag = 0.0  # m = 0 modes must be real

    # Zero the lower triangle, which has no meaning
    for i in range(truncation + 1):
        spectral_array[0, 0, :i, i] = 0.0 + 0.0j

    return spectral_array


def _lons_per_lat(nlat: int, grid_kind: str) -> list[int]:
    if grid_kind == "regular":
        return [2 * nlat] * nlat
    if grid_kind == "reduced":
        if nlat != 640:
            raise ValueError("Only the N320 reduced Gaussian grid SHT (nlat = 640) is supported.")
        # Fetch regular grid data
        from anemoi.transform.grids.named import lookup

        lats = lookup(f"n{nlat // 2}")["latitudes"]

        # Get latitudes of this grid
        unique_lats = sorted(set(lats))

        # Calculate longitudes per latitude
        lons = [int((lats == unique_lat).sum()) for unique_lat in unique_lats]

        return lons
    if grid_kind == "octahedral":
        lons = [20 + 4 * i for i in range(nlat // 2)]
        return lons + list(reversed(lons))

    raise ValueError(f"Unknown grid_kind={grid_kind!r}")


def _transforms(lons_per_lat: list[int], truncation: int):
    """Create forward and inverse transforms for one grid and truncation."""
    return SphericalHarmonicTransform(lons_per_lat, truncation), InverseSphericalHarmonicTransform(
        lons_per_lat, truncation
    )


def _assert_matches_legendre_sums(
    direct: SphericalHarmonicTransform, inverse: InverseSphericalHarmonicTransform, truncation: int
) -> None:
    """Compare both transforms with Legendre sums over all latitudes."""
    # Rings run from north to south, the Gauss-Legendre nodes from south to north.
    cos_colat, weight = legendre_gauss_weights(direct.nlat)
    table = torch.stack(list(legendre_by_degree(truncation, np.flip(cos_colat))), dim=1).to(
        torch.complex128
    )  # [m, l, lat]
    weight = torch.from_numpy(np.flip(weight).copy()).to(torch.complex128)

    x = torch.randn(3, direct.n_grid_points, dtype=torch.float64)
    fourier = 2 * torch.pi * direct.rfft_rings(x)[..., : truncation + 1]
    expected = 4 * torch.pi * torch.einsum("...km, mlk, k -> ...lm", fourier, table, weight)
    torch.testing.assert_close(direct(x), expected)

    coefficients = random_spectral_array(truncation, torch.float64)
    expected = inverse.irfft_rings(torch.einsum("...lm, mlk -> ...km", coefficients, table))
    torch.testing.assert_close(inverse(coefficients), expected)


def _dense_hemisphere_table(nlat: int, truncation: int) -> torch.Tensor:
    """Build a dense reference table with axes (m, r, j, northern latitude)."""
    cos_colat, _ = legendre_gauss_weights(nlat)
    table = torch.zeros(truncation + 1, 2, (truncation + 2) // 2, nlat // 2, dtype=torch.float64)
    for degree, values in enumerate(legendre_by_degree(truncation, np.flip(cos_colat)[: nlat // 2])):
        table[: degree + 1, degree % 2, degree // 2] = values[: degree + 1]
    return table


def _dense_reference(
    direct: SphericalHarmonicTransform,
    inverse: InverseSphericalHarmonicTransform,
    x: torch.Tensor,
    coefficients: torch.Tensor,
):
    """Compute forward and inverse transforms with a dense float64 table."""
    truncation, half = direct.truncation, direct.nlat // 2
    table = _dense_hemisphere_table(direct.nlat, truncation)
    _, weight = legendre_gauss_weights(direct.nlat)
    weight = torch.from_numpy(8 * np.pi**2 * np.flip(weight)[:half].copy())
    sign = (-1.0) ** torch.arange(truncation + 1, dtype=torch.float64)

    fourier = torch.view_as_real(direct.rfft_rings(x)[..., : truncation + 1])
    north, south = fourier[..., :half, :, :], fourier[..., half:, :, :].flip(-3) * sign[:, None]
    folded = torch.stack((north + south, north - south), dim=-2) * weight[:, None, None, None]
    forward = torch.einsum("...kmrc, mrjk -> ...mrjc", folded, table)
    forward = forward.permute(*range(forward.dim() - 4), -2, -3, -4, -1).flatten(-4, -3)[..., : truncation + 1, :, :]

    n_j = table.shape[2]
    spectral = torch.view_as_real(coefficients)
    spectral = torch.nn.functional.pad(spectral, (0, 0, 0, 0, 0, 2 * n_j - (truncation + 1))).unflatten(-3, (n_j, 2))
    rings = torch.einsum("...jrmc, mrjk -> ...kmrc", spectral, table)
    north = rings[..., 0, :] + rings[..., 1, :]
    south = (rings[..., 0, :] - rings[..., 1, :]) * sign[:, None]
    grid = inverse.irfft_rings(torch.view_as_complex(torch.cat((north, south.flip(-3)), dim=-3)))
    return torch.view_as_complex(forward.contiguous()), grid


@pytest.fixture(params=[1, 3, 16], ids=lambda width: f"orders_per_block={width}")
def order_blocks(request, monkeypatch):
    """Set the minimum number of orders per Legendre block for each test."""
    monkeypatch.setattr(spectral_helpers, "MIN_ORDERS_PER_BLOCK", request.param)
    LEGENDRE_TABLES.clear()  # tables built with other blocks must not be reused
    yield request.param
    LEGENDRE_TABLES.clear()


@pytest.mark.parametrize("truncation", [5, 6, 14, 15, 31])
@pytest.mark.parametrize("grid_kind", ["regular", "octahedral"])
def test_matches_legendre_sums_over_all_latitudes(grid_kind, truncation, order_blocks):
    torch.manual_seed(0)
    direct, inverse = _transforms(_lons_per_lat(32, grid_kind), truncation)
    _assert_matches_legendre_sums(direct, inverse, truncation)


@pytest.mark.parametrize("truncation", [5, 6, 14, 15, 31])
@pytest.mark.parametrize("grid_kind", ["regular", "octahedral"])
def test_matches_dense_reference(grid_kind, truncation, order_blocks):
    """Compare transforms using Legendre blocks with the dense reference."""
    torch.manual_seed(0)
    direct, inverse = _transforms(_lons_per_lat(32, grid_kind), truncation)
    x = torch.randn(3, direct.n_grid_points, dtype=torch.float64)
    coefficients = random_spectral_array(truncation, torch.float64)
    expected_forward, expected_inverse = _dense_reference(direct, inverse, x, coefficients)
    torch.testing.assert_close(direct(x), expected_forward, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(inverse(coefficients), expected_inverse, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_gradients_through_blocks(direction, order_blocks):
    """Check gradients through Legendre blocks and zero padding in both directions."""
    torch.manual_seed(0)
    direct, inverse = _transforms(_lons_per_lat(16, "octahedral"), 12)
    if direction == "forward":
        inputs = torch.randn(2, direct.n_grid_points, dtype=torch.float64, requires_grad=True)
        function = direct
    else:
        inputs = random_spectral_array(12, torch.float64).requires_grad_()
        function = inverse
    assert torch.autograd.gradcheck(function, (inputs,), fast_mode=True)


@pytest.mark.parametrize("truncation", [5, 6, 9, 10, 13])
def test_smaller_truncation_uses_part_of_a_larger_table(truncation, order_blocks):
    """Check that slicing a cached table for a lower truncation preserves the transform results."""
    torch.manual_seed(0)
    larger = SphericalHarmonicTransform(_lons_per_lat(32, "octahedral"), 31)
    larger(torch.randn(1, larger.n_grid_points, dtype=torch.float64))
    direct, inverse = _transforms(_lons_per_lat(32, "regular"), truncation)
    _assert_matches_legendre_sums(direct, inverse, truncation)
    x = torch.randn(3, direct.n_grid_points, dtype=torch.float64)
    coefficients = random_spectral_array(truncation, torch.float64)
    expected_forward, expected_inverse = _dense_reference(direct, inverse, x, coefficients)
    torch.testing.assert_close(direct(x), expected_forward, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(inverse(coefficients), expected_inverse, rtol=1e-13, atol=1e-14)
    assert LEGENDRE_TABLES.get(32, truncation, torch.float64, torch.device("cpu")).truncation == 31


def _block_bytes(nlat: int, truncation: int, element_size: int = 4) -> int:
    """Calculate table storage in bytes for one grid and truncation."""
    first_orders = first_orders_of_blocks(truncation)
    n_j = (truncation + 2) // 2
    legendre = sum(
        (m1 - m0) * 2 * (n_j - m0 // 2) * (nlat // 2)
        for m0, m1 in zip(first_orders, first_orders[1:] + [truncation + 1])
    )
    return element_size * (legendre + nlat // 2 + truncation + 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_transforms_on_one_grid_share_one_set_of_tables(device, order_blocks):
    """Check table sharing, replacement at higher truncation and cache clearing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    nlat = 18
    smaller = SphericalHarmonicTransform(_lons_per_lat(nlat, "regular"), 4).to(device)
    larger = SphericalHarmonicTransform(_lons_per_lat(nlat, "octahedral"), 8).to(device)
    smaller(torch.randn(2, smaller.n_grid_points, device=device))
    assert LEGENDRE_TABLES.stored_bytes() == _block_bytes(nlat, 4)
    for sht in (larger, smaller):
        sht(torch.randn(2, sht.n_grid_points, device=device))
    assert LEGENDRE_TABLES.stored_bytes() == _block_bytes(nlat, 8)

    # An inverse transform on the same latitudes uses the same tables.
    inverse = InverseSphericalHarmonicTransform(_lons_per_lat(nlat, "regular"), 8).to(device)
    inverse(torch.zeros(2, 9, 9, dtype=torch.complex64, device=device))
    assert LEGENDRE_TABLES.stored_bytes() == _block_bytes(nlat, 8)

    LEGENDRE_TABLES.clear()
    assert LEGENDRE_TABLES.stored_bytes() == 0


def test_blocks_leave_out_most_zeros():
    """Check the storage reduction from omitting zero entries at T319 on 640 latitudes."""
    dense = 320 * 2 * 160 * 320 * 4
    assert _block_bytes(640, 319) < 0.6 * dense


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("first_direction", ["forward", "inverse"])
def test_tables_first_built_in_inference_mode_serve_training(device, first_direction):
    """Check gradient computation when a shared table was first built in inference mode."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    LEGENDRE_TABLES.clear()
    lons_per_lat = _lons_per_lat(16, "octahedral")
    metric = [transform.to(device) for transform in _transforms(lons_per_lat, 7)]
    loss = [transform.to(device) for transform in _transforms(lons_per_lat, 7)]
    with torch.inference_mode():
        if first_direction == "forward":
            metric[0](torch.randn(2, metric[0].n_grid_points, device=device))
        else:
            metric[1](torch.zeros(2, 8, 8, dtype=torch.complex64, device=device))

    for direct, inverse in (metric, loss):
        x = torch.randn(2, direct.n_grid_points, device=device, requires_grad=True)
        inverse(direct(x)).square().sum().backward()
        assert torch.isfinite(x.grad).all()


def test_pickled_transform_leaves_the_tables_out():
    """Check that pickling excludes cached tables and loading preserves transform results."""
    sht = SphericalHarmonicTransform(_lons_per_lat(128, "octahedral"), 127)
    x = torch.randn(2, sht.n_grid_points)
    expected = sht(x)
    table_bytes = sum(
        table.nbytes for _, table in LEGENDRE_TABLES.get(128, 127, torch.float32, torch.device("cpu")).legendre_blocks
    )

    saved = io.BytesIO()
    torch.save(sht, saved)
    assert saved.tell() < table_bytes
    LEGENDRE_TABLES.clear()

    saved.seek(0)
    loaded = torch.load(saved, weights_only=False)
    torch.testing.assert_close(loaded(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize(
    "direct_cls,inverse_cls,grid_kwargs",
    [
        pytest.param(ReducedSHT, InverseReducedSHT, {"grid": "n320"}, id="reduced"),
        pytest.param(OctahedralSHT, InverseOctahedralSHT, {"nlat": 8}, id="octahedral"),
    ],
)
def test_sht_wrappers_preserve_axes_and_gradients(
    device, dtype, direction, direct_cls, inverse_cls, grid_kwargs, monkeypatch
):
    """Check values, axis ordering and gradients for reduced Gaussian SHT wrappers."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(42)
    nlat, truncation = 8, 2
    if direct_cls is ReducedSHT:
        # Only replace the grid lookup: exercise both real reduced wrappers on a
        # small grid. The existing core tests cover the full N320 grid.
        lengths = [8, 12, 16, 20, 20, 16, 12, 8]
        latitudes = np.repeat(np.arange(nlat), lengths)
        monkeypatch.setattr("anemoi.transform.grids.named.lookup", lambda grid: {"latitudes": latitudes})
    else:
        lengths = _lons_per_lat(nlat, "octahedral")
    direct = direct_cls(**grid_kwargs, truncation=truncation).to(device)
    inverse = inverse_cls(**grid_kwargs, truncation=truncation).to(device)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    # [batch, time, ensemble, variable, l, m], with physically meaningful modes.
    coefficients = torch.randn(2, 3, 2, 4, truncation + 1, truncation + 1, device=device, dtype=cdtype).tril()
    coefficients[..., 0].imag = 0
    tolerance = 3e-6 if dtype == torch.float32 else 2e-13
    if direction == "forward":
        reference = SphericalHarmonicTransform(lengths, truncation).to(device)
        x = inverse(coefficients).movedim(-2, -1).detach().requires_grad_()
        actual = direct(x)
        torch.testing.assert_close(actual, coefficients.movedim(-3, -1), rtol=tolerance, atol=tolerance)
        # Transform each variable independently to check the wrapper's axis handling.
        expected = torch.stack([reference(x[..., variable]) for variable in range(x.shape[-1])], dim=-1)
    else:
        reference = InverseSphericalHarmonicTransform(lengths, truncation).to(device)
        x = coefficients.requires_grad_()
        actual = inverse(x)
        expected = torch.stack([reference(x[..., variable, :, :]) for variable in range(x.shape[-3])], dim=-2)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, grad)[0]
    expected_grad = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)


@pytest.fixture
def sht_setup(request):
    # Choose GPUs if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        # We only support the N320 reduced Gaussian grid
        if request.param == "reduced":
            truncation = 319  # T319 corresponding to N320 grid
            tolerance = 1e-8  # Higher resolution grids need higher tolerance -> larger accumulated errors
        # Other grids, we can do what we like
        else:
            truncation = 39  # T39 corresponding to O40 grid
            tolerance = 1e-11

        dtype = torch.float64  # float64 for numerical correctness checking
        torch.manual_seed(0)  # fix RNG seed for reproducibility

        nlat = 2 * (truncation + 1)
        lons_per_lat = _lons_per_lat(nlat=nlat, grid_kind=request.param)

        direct = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).to(device)
        inverse = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation).to(device)

        yield {
            "grid_kind": request.param,
            "truncation": truncation,
            "dtype": dtype,
            "tolerance": tolerance,
            "direct": direct,
            "inverse": inverse,
        }


@pytest.mark.parametrize("sht_setup", ["regular", "reduced", "octahedral"], indirect=True)
def test_idempotency_direct_inverse(sht_setup):
    """Direct followed by inverse returns the original (band-limited) field."""
    truncation = sht_setup["truncation"]
    dtype = sht_setup["dtype"]
    tolerance = sht_setup["tolerance"]
    direct = sht_setup["direct"]
    inverse = sht_setup["inverse"]

    before_spectral = random_spectral_array(truncation, dtype)

    # Ensure the direct input is band-limited by constructing it via inverse.
    before = inverse(before_spectral)

    after = inverse(direct(before))
    assert torch.allclose(before, after, rtol=tolerance)


@pytest.mark.parametrize("sht_setup", ["regular", "reduced", "octahedral"], indirect=True)
def test_idempotency_inverse_direct(sht_setup):
    """Inverse followed by direct returns the original spectral coefficients."""
    truncation = sht_setup["truncation"]
    dtype = sht_setup["dtype"]
    tolerance = sht_setup["tolerance"]
    direct = sht_setup["direct"]
    inverse = sht_setup["inverse"]

    before = random_spectral_array(truncation, dtype)
    after = direct(inverse(before))

    # Compute max relative diff over the meaningful upper triangle (including diagonal)
    maxdiff = 0.0
    for m in range(truncation + 1):
        ref = before[0, 0, m:, m]
        got = after[0, 0, m:, m]
        maxdiff = max(maxdiff, torch.abs((ref - got) / ref).max().item())

    assert maxdiff < tolerance


@pytest.mark.skip(reason="CUDA graphs are experimental so this test is disabled by default")
@pytest.mark.parametrize("sht_setup", ["reduced", "octahedral"], indirect=True)
def test_multiple_direct_calls(sht_setup):
    """Test direct transform can be called multiple times, to verify the CUDA graph functionality works correctly.
    Reduced grids only.
    """
    dtype = sht_setup["dtype"]
    direct = sht_setup["direct"]

    before = torch.randn((1, 1, direct.n_grid_points), dtype=dtype)

    once = direct(before)

    twice = direct(before)

    assert torch.all(once == twice)


@pytest.mark.skip(reason="CUDA graphs are experimental so this test is disabled by default")
@pytest.mark.parametrize("sht_setup", ["reduced", "octahedral"], indirect=True)
def test_direct_with_graphed_reduced_fft(sht_setup):
    """Check gradients still work for reduced-grid FFT with CUDA graphs on."""
    dtype = sht_setup["dtype"]
    direct = sht_setup["direct"]

    x = torch.randn((2, 1, direct.n_grid_points), dtype=dtype, requires_grad=True)
    y = direct(x)
    loss = torch.square(torch.abs(y)).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
