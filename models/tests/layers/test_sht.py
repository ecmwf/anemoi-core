# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform

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


@pytest.fixture(params=["regular", "reduced", "octahedral"])
def sht_setup(request):
    # Choose GPUs if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

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

    direct = SphericalHarmonicTransform(nlat, lons_per_lat=lons_per_lat, lmax=truncation + 1, mmax=truncation + 1).to(
        device
    )
    inverse = InverseSphericalHarmonicTransform(
        nlat, lons_per_lat=lons_per_lat, lmax=truncation + 1, mmax=truncation + 1
    ).to(device)

    return {
        "grid_kind": request.param,
        "truncation": truncation,
        "dtype": dtype,
        "tolerance": tolerance,
        "direct": direct,
        "inverse": inverse,
    }


def test_idempotency_direct_inverse(sht_setup):
    """direct followed by inverse returns the original (band-limited) field."""
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


def test_idempotency_inverse_direct(sht_setup):
    """inverse followed by direct returns the original spectral coefficients."""
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
