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

from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import OctahedralSHT


def _make(transform: str, x_dim: int, y_dim: int):
    if transform == "fft2d":
        return FFT2D(x_dim=x_dim, y_dim=y_dim)
    pytest.importorskip("torch_dct")
    return DCT2D(x_dim=x_dim, y_dim=y_dim)


# Transforms exposing the power/cross spectral-density contract, spanning both families:
# the 2D Cartesian transforms (radial-wavenumber binning) and the spherical harmonics
# (per-degree reduction). Each entry yields (transform, number of spatial points it expects).
_DENSITY_TRANSFORMS = ["fft2d", "dct2d", "sht"]


def _make_density_transform(kind: str):
    if kind in ("fft2d", "dct2d"):
        return _make(kind, x_dim=8, y_dim=6), 8 * 6
    if kind == "sht":
        t = OctahedralSHT(nlat=8)
        return t, t._sht.n_grid_points
    raise ValueError(f"unknown transform {kind!r}")


def test_radial_band_index_values() -> None:
    """The (ky, kx) -> band map is the hand-computed radial wavenumber, including a
    rectangular grid (distinct y/x axes) so axis alignment is checked, not just totals.
    """
    # FFT2D 4x3: |fftfreq*N| -> ky=[0,1,2,1] (y_dim=4), kx=[0,1,1] (x_dim=3)
    # band[i,j] = round(sqrt(ky[i]**2 + kx[j]**2)), flattened row-major (y outer, x inner).
    t = FFT2D(x_dim=3, y_dim=4)
    expected = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1])
    assert torch.equal(t.radial_band_index, expected)
    assert t.n_radial_bands == 3


def test_radial_bands_are_built_lazily() -> None:
    """Bands cost nothing until a power/cross spectral density is actually requested."""
    t = FFT2D(x_dim=8, y_dim=8)
    assert "_radial_bands" not in t.__dict__  # not computed at construction
    _ = t.n_radial_bands  # first access triggers the cached_property
    assert "_radial_bands" in t.__dict__  # now cached on the instance


@pytest.mark.parametrize("transform", _DENSITY_TRANSFORMS)
def test_density_partitions_total_power(transform: str) -> None:
    """Sum over bands/degrees of the PSD == total power over the spectrum (Parseval).

    A contract of every spectral transform: the 2D radial binning and the SHT's per-degree
    sum are both partitions of the squared spectrum, so this also covers the SHT density.
    """
    t, n_points = _make_density_transform(transform)
    spec = t.forward(torch.randn(2, 1, 2, n_points, 3, dtype=torch.float64))

    psd = t.power_spectral_density(spec)
    total = torch.real(spec * torch.conj(spec)).flatten(-3, -2).sum(dim=-2)
    torch.testing.assert_close(psd.sum(dim=-2), total)


@pytest.mark.parametrize("transform", _DENSITY_TRANSFORMS)
def test_cross_self_consistency_and_cauchy_schwarz(transform: str) -> None:
    """cross(x,x) == psd(x), and |cross| <= sqrt(psd_a * psd_b) per band/degree."""
    t, n_points = _make_density_transform(transform)
    a = t.forward(torch.randn(2, 1, 1, n_points, 3, dtype=torch.float64))
    b = t.forward(torch.randn(2, 1, 1, n_points, 3, dtype=torch.float64))

    psd_a = t.power_spectral_density(a)
    torch.testing.assert_close(t.cross_spectral_density(a, a), psd_a)

    psd_b = t.power_spectral_density(b)
    cross = t.cross_spectral_density(a, b)
    assert torch.all(cross**2 <= psd_a * psd_b * (1 + 1e-9) + 1e-9)


@pytest.mark.parametrize("transform", ["fft2d", "dct2d"])
def test_pure_wave_localizes_to_expected_band(transform: str) -> None:
    """A single wave at (ky0, kx0) deposits its power in band round(sqrt(ky0^2+kx0^2))."""
    x_dim, y_dim = 16, 16
    t = _make(transform, x_dim, y_dim)
    ky0, kx0 = 2, 3
    yy = torch.arange(y_dim).view(y_dim, 1).double()
    xx = torch.arange(x_dim).view(1, x_dim).double()
    if transform == "fft2d":
        field = torch.cos(2 * torch.pi * (ky0 * yy / y_dim + kx0 * xx / x_dim))
    else:  # DCT-II cosine mode
        field = torch.cos(torch.pi * (yy + 0.5) * ky0 / y_dim) * torch.cos(torch.pi * (xx + 0.5) * kx0 / x_dim)

    spec = t.forward(field.reshape(1, 1, 1, y_dim * x_dim, 1))
    psd = t.power_spectral_density(spec)[0, 0, 0, :, 0]
    expected = round((ky0**2 + kx0**2) ** 0.5)
    assert (psd[expected] / psd.sum()).item() > 0.99


def test_dct_real_coeffs_do_not_use_imag() -> None:
    """DCT2D returns real coefficients; the density methods must not touch ``.imag``.

    (``tensor.imag`` raises for real dtypes, which is why the SHT methods cannot be reused.)
    """
    pytest.importorskip("torch_dct")
    t = DCT2D(x_dim=8, y_dim=6)
    spec = t.forward(torch.randn(1, 1, 1, 48, 2, dtype=torch.float64))
    assert not spec.is_complex()
    # would raise if the implementation referenced spec.imag
    psd = t.power_spectral_density(spec)
    assert torch.isfinite(psd).all()
