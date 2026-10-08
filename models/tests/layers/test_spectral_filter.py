# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import pytest
import torch

from anemoi.models.layers.spectral_helpers import legendre_gauss_weights
from anemoi.models.layers.spectral_helpers import quadrature_weights
from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter

NLAT = 16
REGULAR = [2 * NLAT] * NLAT
OCTAHEDRAL = [20 + 4 * i for i in range(NLAT // 2)] + [20 + 4 * i for i in reversed(range(NLAT // 2))]


def band_limited_coeffs(lmax: int, batch: int = 3, seed: int = 0) -> torch.Tensor:
    """Random coefficients of a real field: m <= l, and a real m = 0 column."""
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(batch, lmax, lmax, generator=generator)
    imag = torch.randn(batch, lmax, lmax, generator=generator)
    imag[..., 0] = 0.0
    return torch.tril(torch.complex(real, imag))


def energy_above(coeffs: torch.Tensor, degree: int) -> float:
    return (coeffs[..., degree + 1 :, :].abs() ** 2).sum().item()


@pytest.mark.parametrize("lons_per_lat", [REGULAR, OCTAHEDRAL], ids=["regular", "octahedral"])
def test_quadrature_weights_are_an_exact_area_mean(lons_per_lat) -> None:
    weights = quadrature_weights(lons_per_lat)
    nodes, _ = legendre_gauss_weights(len(lons_per_lat))
    sin_lat = np.repeat(nodes, lons_per_lat)

    assert weights.shape == (sum(lons_per_lat),)
    assert weights.sum() == pytest.approx(1.0)
    # sin^2(lat) averages to exactly 1/3 over the sphere
    assert (weights * sin_lat**2).sum() == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_reduced_grid_polar_points_weigh_less_than_equatorial_ones() -> None:
    """A per-point mean would over-represent the densely packed polar rows."""
    weights = quadrature_weights(OCTAHEDRAL)
    assert weights[0] < weights[sum(OCTAHEDRAL[: NLAT // 2])]


def test_filter_round_trips_a_band_limited_field() -> None:
    spectral_filter = SphericalSpectralFilter(REGULAR, truncation=NLAT - 1)
    field = spectral_filter.synthesise(band_limited_coeffs(NLAT))

    filtered = spectral_filter(field, torch.ones(NLAT))

    torch.testing.assert_close(filtered, field, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("lons_per_lat", [REGULAR, OCTAHEDRAL], ids=["regular", "octahedral"])
def test_mean_square_from_the_coefficients_is_the_area_mean_of_the_field(lons_per_lat) -> None:
    """Parseval: the area-mean square of the synthesised field, without synthesising it."""
    spectral_filter = SphericalSpectralFilter(lons_per_lat, truncation=NLAT - 1)
    coeffs = band_limited_coeffs(NLAT)
    weights = torch.as_tensor(quadrature_weights(lons_per_lat), dtype=torch.float32)

    grid_mean_square = (spectral_filter.synthesise(coeffs) ** 2 * weights).sum(dim=-1)

    torch.testing.assert_close(spectral_filter.mean_square(coeffs), grid_mean_square, rtol=1e-5, atol=0.0)


def test_filter_removes_the_energy_a_product_puts_above_the_cutoff() -> None:
    spectral_filter = SphericalSpectralFilter(REGULAR, truncation=NLAT - 1)
    low = band_limited_coeffs(NLAT, seed=1)
    low[..., 5:, :] = 0.0
    product = spectral_filter.synthesise(low) * spectral_filter.synthesise(band_limited_coeffs(NLAT, seed=2))
    cutoff = 4
    response = (torch.arange(NLAT) <= cutoff).float()

    before = spectral_filter.analyse(product)
    after = spectral_filter.analyse(spectral_filter(product, response))

    assert energy_above(before, cutoff) > 0.5 * (before.abs() ** 2).sum().item()
    assert energy_above(after, cutoff) < 1e-8 * (after.abs() ** 2).sum().item()
    # and inside the band the field is untouched
    torch.testing.assert_close(after[..., : cutoff + 1, :], before[..., : cutoff + 1, :], atol=1e-4, rtol=1e-4)


def test_per_channel_responses_broadcast() -> None:
    spectral_filter = SphericalSpectralFilter(REGULAR, truncation=NLAT - 1)
    field = spectral_filter.synthesise(band_limited_coeffs(NLAT, batch=2)).unsqueeze(0)  # (1, channels, points)
    responses = torch.stack([torch.ones(NLAT), torch.zeros(NLAT)])

    filtered = spectral_filter(field, responses)

    torch.testing.assert_close(filtered[:, 0], field[:, 0], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(filtered[:, 1].abs() > 1e-6) == 0


def test_coefficients_carry_over_to_a_larger_truncation_by_zero_padding() -> None:
    """What lets a low-truncation filter write straight into a full-resolution spectral state."""
    small = SphericalSpectralFilter(REGULAR, truncation=7)
    large = SphericalSpectralFilter(REGULAR, truncation=NLAT - 1)
    coeffs = band_limited_coeffs(8)
    padded = torch.zeros(coeffs.shape[0], NLAT, NLAT, dtype=coeffs.dtype)
    padded[..., :8, :8] = coeffs

    torch.testing.assert_close(small.synthesise(coeffs), large.synthesise(padded), atol=1e-5, rtol=1e-5)


def test_legendre_tables_are_float32_and_not_checkpointed() -> None:
    spectral_filter = SphericalSpectralFilter(REGULAR, truncation=NLAT - 1)
    assert spectral_filter._sht.weight.dtype == torch.float32
    assert spectral_filter._isht.pct.dtype == torch.float32
    assert spectral_filter.state_dict() == {}


def test_exactness_is_measured_not_assumed(caplog) -> None:
    """How high a grid analyses exactly depends on its rings; the filter measures it and warns."""
    octahedral_32 = [20 + 4 * i for i in range(16)] + [20 + 4 * i for i in reversed(range(16))]
    with caplog.at_level(logging.WARNING):
        inexact = SphericalSpectralFilter(octahedral_32, truncation=31)
    assert inexact.round_trip_error > 1e-4
    assert "does not analyse degrees up to" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        exact = [
            SphericalSpectralFilter(octahedral_32, truncation=25),
            SphericalSpectralFilter(OCTAHEDRAL, truncation=NLAT - 1),
            SphericalSpectralFilter(REGULAR, truncation=NLAT - 1),
        ]
    assert all(spectral_filter.round_trip_error < 1e-5 for spectral_filter in exact)
    assert "does not analyse degrees up to" not in caplog.text
