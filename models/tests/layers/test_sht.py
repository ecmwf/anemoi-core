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


def random_spectral_array(truncation, dtype):
    # Shape: [batch index, ensemble member, l, m]
    shape = (1, 1, truncation + 1, truncation + 1)
    spectral_array = torch.complex(torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype))
    spectral_array[0, 0, :, 0].imag = 0.0  # m = 0 modes must be real
    # Zero the lower triangle, which has no meaning
    for i in range(truncation + 1):
        spectral_array[0, 0, :i, i] = 0.0 + 0.0j

    return spectral_array


class TestRegularSphericalHarmonicTransform:
    """Test suite for OctahedralSphericalHarmonicTransform."""

    @pytest.fixture
    def init(self):
        # Choose GPUs if available
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Spectral truncation
        torch.set_default_device(device)

        truncation = 39  # T39 corresponding to O40 grid
        dtype = torch.float64  # float 64 for numerical correctness checking
        torch.manual_seed(0)  # set the random seed for reproducibility
        tolerance = 1e-11  # define relative tolerance for numerical comparisons

        nlat = 2 * (truncation + 1)
        lons_per_lat = [2 * nlat] * nlat

        # Create SHT objects
        direct = SphericalHarmonicTransform(
            nlat, lons_per_lat=lons_per_lat, lmax=truncation + 1, mmax=truncation + 1
        ).to(device)
        inverse = InverseSphericalHarmonicTransform(
            nlat, lons_per_lat=lons_per_lat, lmax=truncation + 1, mmax=truncation + 1
        ).to(device)

        return {"truncation": truncation, "dtype": dtype, "tolerance": tolerance, "direct": direct, "inverse": inverse}

    def test_idempotency_direct_inverse(self, init):
        """Test that direct followed by inverse transform returns the original data."""

        truncation, dtype, tolerance = init["truncation"], init["dtype"], init["tolerance"]
        direct, inverse = init["direct"], init["inverse"]

        # Input: random numbers on the spectral grid
        before_spectral = random_spectral_array(truncation, dtype)

        # The input to the direct transform MUST be band limited in the latitudinal direction up to
        # the truncation
        # We achieve this by first performing the inverse transform on a randomised field
        before = inverse(before_spectral)

        # Idempotency test
        after = inverse(direct(before))
        assert torch.allclose(before, after, rtol=tolerance)

    def test_idempotency_inverse_direct(self, init):
        """Test that inverse followed by direct transform returns the original data."""

        truncation, dtype, tolerance = init["truncation"], init["dtype"], init["tolerance"]
        direct, inverse = init["direct"], init["inverse"]

        # Input: random numbers on the spectral grid
        before = random_spectral_array(truncation, dtype)

        # Idempotency test
        after = direct(inverse(before))

        # Compute max relative diff
        maxdiff = 0.0
        for i in range(truncation + 1):
            maxdiff = max(maxdiff, torch.abs((before[0, 0, i:, i] - after[0, 0, i:, i]) / before[0, 0, i:, i]).max())

        assert maxdiff < tolerance


class TestOctahedralSphericalHarmonicTransform(TestRegularSphericalHarmonicTransform):
    """Test suite for OctahedralSphericalHarmonicTransform."""

    @pytest.fixture
    def init(self):
        # Choose GPUs if available
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Spectral truncation
        torch.set_default_device(device)

        truncation = 39  # T39 corresponding to O40 grid
        dtype = torch.float64  # float 64 for numerical correctness checking
        torch.manual_seed(0)  # set the random seed for reproducibility
        tolerance = 1e-11  # define relative tolerance for numerical comparisons

        nlat = 2 * (truncation + 1)
        lons_per_lat = [20 + 4 * i for i in range(nlat // 2)]
        lons_per_lat += list(reversed(lons_per_lat))

        # Create SHT objects
        direct = SphericalHarmonicTransform(
            nlat, lons_per_lat=lons_per_lat, lmax=truncation + 1, mmax=truncation + 1
        ).to(device)
        inverse = InverseSphericalHarmonicTransform(
            nlat, lons_per_lat=lons_per_lat, lmax=truncation + 1, mmax=truncation + 1
        ).to(device)

        return {"truncation": truncation, "dtype": dtype, "tolerance": tolerance, "direct": direct, "inverse": inverse}
