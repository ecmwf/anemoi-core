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
from scipy.integrate import quad

from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import latitude_quadrature
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.models.layers.spherical_wavelets import RieszStructureTensor
from anemoi.models.layers.spherical_wavelets import SphericalGradient
from anemoi.models.layers.spherical_wavelets import s2let_step
from anemoi.models.layers.spherical_wavelets import scale_degree_ranges
from anemoi.models.layers.spherical_wavelets import wavelet_kernels

NLAT = 24
TRUNCATION = 11  # five scales: l = 1, 2-3, 3-7, 5-11 and 9-11
GRIDS = {
    "octahedral": octahedral_lons_per_lat(NLAT),
    "regular": [2 * NLAT] * NLAT,
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def point_coordinates(lons_per_lat: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitude, longitude and area (adding up to 1) of every point, ring by ring from north to south."""
    colatitude, weights = latitude_quadrature(len(lons_per_lat), "legendre-gauss")
    latitude = np.repeat(0.5 * np.pi - colatitude, lons_per_lat)
    longitude = np.concatenate([2.0 * np.pi * np.arange(n) / n for n in lons_per_lat])
    area = np.repeat(weights / np.asarray(lons_per_lat), lons_per_lat)
    return latitude, longitude, area / area.sum()


def band_limited(lons_per_lat: list[int], batch: int, truncation: int, seed: int = 0) -> torch.Tensor:
    """Random real fields ``[batch, grid]`` in float64 that the grid represents exactly up to the truncation."""
    generator = torch.Generator().manual_seed(seed)
    coeffs = torch.randn(batch, truncation + 1, truncation + 1, dtype=torch.complex128, generator=generator).tril()
    coeffs[..., 0] = coeffs[..., 0].real
    return InverseSphericalHarmonicTransform(lons_per_lat, truncation)(coeffs)


def test_the_step_is_the_s2let_step() -> None:
    """k(t) = integral of b(x) / x from t to 1, divided by the same from 1/2; b = exp(-2 / (1 - y^2)), y = 4 x - 3."""

    def integrand(x: float) -> float:
        y = 4.0 * (x - 0.5) - 1.0
        return np.exp(-2.0 / (1.0 - y * y)) / x if abs(y) < 1.0 else 0.0

    full = quad(integrand, 0.5, 1.0, epsabs=1e-14)[0]
    t = np.array([0.3, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0, 1.2])
    expected = np.array([quad(integrand, max(x, 0.5), 1.0, epsabs=1e-14)[0] / full if x < 1.0 else 0.0 for x in t])
    np.testing.assert_allclose(s2let_step(t), expected, atol=1e-6)


def test_scale_degree_ranges() -> None:
    assert scale_degree_ranges(TRUNCATION) == [(1, 1), (2, 3), (3, 7), (5, 11), (9, 11)]


def test_kernels_and_the_scaling_function_add_up_to_one_and_cover_their_octaves() -> None:
    kernels = wavelet_kernels(1279)
    assert kernels.shape == (12, 1280)
    ell = np.arange(1280)
    scaling = s2let_step(ell.astype(np.float64))
    np.testing.assert_allclose(scaling + (kernels**2).sum(0), 1.0, atol=1e-12)
    for j, (row, (lowest, highest)) in enumerate(zip(kernels, scale_degree_ranges(1279))):
        # Every degree of the scale has a weight, also those next to its ends.
        np.testing.assert_array_equal(np.flatnonzero(row > 0), np.arange(lowest, highest + 1))
        if 2**j <= 1279:
            assert row[2**j] == pytest.approx(1.0)


@pytest.mark.parametrize("truncation", [1, 2, 3, 4, 5, 8, 9, 65, 66, 129, 1024, 1025, 1279, 2048, 2049])
def test_one_scale_for_each_octave_that_starts_at_or_below_the_truncation(truncation: int) -> None:
    ranges = scale_degree_ranges(truncation)
    n_scales = len(ranges)
    assert ranges[-1][0] <= truncation
    assert 2**n_scales // 2 + 1 > truncation  # the next scale would start above the truncation
    assert wavelet_kernels(truncation).shape[0] == n_scales


@pytest.mark.parametrize("grid", GRIDS)
def test_gradient_of_a_known_field(grid: str) -> None:
    """A field of degrees 1 to 6 and orders 0, 1 and 5, against its gradient written out by hand."""
    lons_per_lat = GRIDS[grid]
    lat, lon, _ = point_coordinates(lons_per_lat)
    sin, cos = np.sin(lat), np.cos(lat)
    field = cos * np.cos(lon) + sin + sin**3 + sin * cos**5 * np.cos(5 * lon)
    east = -np.sin(lon) - 5 * sin * cos**4 * np.sin(5 * lon)
    north = cos - sin * np.cos(lon) + 3 * sin**2 * cos + (cos**6 - 5 * sin**2 * cos**4) * np.cos(5 * lon)
    coeffs = SphericalHarmonicTransform(lons_per_lat, 7)(torch.from_numpy(field))
    gradient = SphericalGradient(lons_per_lat, 7)(coeffs)
    np.testing.assert_allclose(gradient[0].numpy(), east, atol=1e-10)
    np.testing.assert_allclose(gradient[1].numpy(), north, atol=1e-10)


def test_gradient_needs_room_for_one_more_degree() -> None:
    with pytest.raises(ValueError, match="one degree beyond the truncation"):
        SphericalGradient(GRIDS["octahedral"], NLAT - 1)


@pytest.mark.parametrize("grid", GRIDS)
def test_the_riesz_pair_has_the_energy_of_its_band(grid: str) -> None:
    lons_per_lat = GRIDS[grid]
    _, _, area = point_coordinates(lons_per_lat)
    area = torch.from_numpy(area)
    structure = RieszStructureTensor(lons_per_lat, TRUNCATION, scales=[0, 1, 2, 3, 4])
    kernels = torch.from_numpy(wavelet_kernels(TRUNCATION))
    field = band_limited(lons_per_lat, 2, TRUNCATION)
    coeffs = structure.coefficients(field)
    inverse = InverseSphericalHarmonicTransform(lons_per_lat, TRUNCATION)
    for k, j in enumerate(structure.scales):
        band = inverse(coeffs * kernels[j][:, None])
        pair = structure.riesz_pair(coeffs, k)
        torch.testing.assert_close((pair.square().sum(-2) * area).sum(-1), (band.square() * area).sum(-1))


@pytest.mark.parametrize("grid", ["octahedral", "regular"])
@pytest.mark.parametrize(("scale", "octaves"), [(5, 2), (6, 2)])
def test_smoothing_loses_about_one_over_ls_squared_to_the_radial_direction(grid: str, scale: int, octaves: int) -> None:
    """The heat kernel is 1 at l = 0, so the smoothing keeps the global mean. The read-off in the tangent plane
    leaves out about 1 / Ls^2.
    """
    nlat, truncation = 128, 79
    lons_per_lat = octahedral_lons_per_lat(nlat) if grid == "octahedral" else [2 * nlat] * nlat
    _, _, area = point_coordinates(lons_per_lat)
    area = torch.from_numpy(area)
    structure = RieszStructureTensor(lons_per_lat, truncation, scales=[scale], smoothing_octaves=octaves)
    coeffs = structure.coefficients(band_limited(lons_per_lat, 2, truncation))
    riesz_energy = (structure.riesz_pair(coeffs, 0).square().sum(-2) * area).sum(-1)
    j_ee, j_nn, _ = structure.tensor(coeffs, 0).unbind(-2)
    lost = 1.0 - ((j_ee + j_nn) * area).sum(-1) / riesz_energy
    expected = 4.0 ** (octaves - scale)
    torch.testing.assert_close(lost, torch.full_like(lost, expected), rtol=0.05, atol=0.0)


def test_the_tensor_is_positive_semi_definite() -> None:
    """Up to about 3e-7 of the largest energy: the cut heat kernel dips that far below 0 for Ls >= 4."""
    nlat, truncation = 128, 79
    lons_per_lat = octahedral_lons_per_lat(nlat)
    structure = RieszStructureTensor(lons_per_lat, truncation, scales=[2, 5, 6], smoothing_octaves=2)  # Ls 1, 8, 16
    j_ee, j_nn, j_en = structure(band_limited(lons_per_lat, 3, truncation)).unbind(-2)
    scale = (j_ee + j_nn).amax(-1, keepdim=True)
    assert (j_ee >= -1e-6 * scale).all()
    assert (j_nn >= -1e-6 * scale).all()
    assert (j_ee * j_nn - j_en**2 >= -1e-6 * scale**2).all()


def test_crests_running_north_to_south_point_east() -> None:
    """A sectoral harmonic cos(m lon) cos(lat)^m has crests along the meridians, so J_ee dominates."""
    lons_per_lat = GRIDS["regular"]
    latitude, longitude, _ = point_coordinates(lons_per_lat)
    m = 8
    field = torch.from_numpy(np.cos(m * longitude) * np.cos(latitude) ** m)
    structure = RieszStructureTensor(lons_per_lat, TRUNCATION, scales=[3], smoothing_octaves=2)
    j_ee, j_nn, j_en = structure(field)[0]
    tropics = torch.from_numpy(np.abs(latitude) < np.radians(10.0))
    coherence = ((j_ee - j_nn) ** 2 + 4 * j_en**2).sqrt() / (j_ee + j_nn)
    assert (coherence[tropics] > 0.8).all()
    assert (j_ee[tropics] > 5 * j_nn[tropics]).all()


def test_reading_the_tensor_in_three_dimensions() -> None:
    generator = torch.Generator().manual_seed(1)
    entries = torch.randn(4, 6, 5, dtype=torch.float64, generator=generator)
    u = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    w = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    full = torch.zeros(4, 3, 3, 5, dtype=torch.float64)
    for index, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        full[:, a, b] = entries[:, index]
        full[:, b, a] = entries[:, index]
    expected = torch.einsum("ag, nabg, bg -> ng", u, full, w)
    torch.testing.assert_close(RieszStructureTensor._read(entries, u, w), expected)


def test_gradients_flow_to_the_field() -> None:
    lons_per_lat = octahedral_lons_per_lat(12)
    structure = RieszStructureTensor(lons_per_lat, 5, scales=[2, 3])
    field = band_limited(lons_per_lat, 1, 5).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: structure(x).square().sum(), (field,), fast_mode=True)


def test_rejects_scales_beyond_the_truncation() -> None:
    with pytest.raises(ValueError, match="scales must be"):
        RieszStructureTensor(GRIDS["octahedral"], TRUNCATION, scales=[5])


def test_rejects_a_scale_too_small_for_float64() -> None:
    with pytest.raises(ValueError, match="float64 can hold"):
        RieszStructureTensor(GRIDS["octahedral"], 2049, scales=[12])


def test_rejects_a_heat_kernel_cut_short() -> None:
    # Scale 3 with one octave needs degrees up to 5 * 4 = 20; the transforms reach 12.
    with pytest.raises(ValueError, match="needs smoothing degrees up to 20"):
        RieszStructureTensor(GRIDS["octahedral"], TRUNCATION, scales=[3], smoothing_octaves=1)


def test_rejects_too_few_latitudes_for_the_products() -> None:
    # The products of scale 4 at T22 reach degree 46, and the smoothing up to 20 needs 34 latitudes.
    with pytest.raises(ValueError, match="need at least 34 latitudes"):
        RieszStructureTensor(GRIDS["octahedral"], 22, scales=[4], smoothing_octaves=2)


def test_needs_more_longitudes_than_the_products_reach() -> None:
    # The products of scale 3 at T11 reach order 24, and the smoothing goes up to 10.
    with pytest.raises(ValueError, match="need more than 34 longitudes"):
        RieszStructureTensor([34] * NLAT, TRUNCATION, scales=[3], smoothing_octaves=2)
    RieszStructureTensor([35] * NLAT, TRUNCATION, scales=[3], smoothing_octaves=2)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_matches_float64(device: str) -> None:
    lons_per_lat = GRIDS["octahedral"]
    structure = RieszStructureTensor(lons_per_lat, TRUNCATION, scales=[2, 3], smoothing_octaves=2).to(device)
    field = band_limited(lons_per_lat, 2, TRUNCATION).to(device)
    expected = structure(field)
    result = structure(field.float())
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.double(), expected, rtol=1e-4, atol=1e-5 * expected.abs().max().item())
