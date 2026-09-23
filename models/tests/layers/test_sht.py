# (C) Copyright 2025-2026 Anemoi contributors.
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

from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
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
