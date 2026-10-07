# (C) Copyright 2025-2026 Anemoi contributors.
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

_DENSITY_TRANSFORMS = ["fft2d", "dct2d", "sht"]


def _make_density_transform(kind: str):
    """Return ``(transform, n_spatial_points)`` for the spectral-density contract.

    The Cartesian transforms only build their radial-band index on demand; call the
    registrar here so the test mirrors what ``SpectralAMSELoss`` does at construction.
    """
    if kind == "fft2d":
        t = FFT2D(x_dim=8, y_dim=6)
        t._register_radial_bands()
        return t, 8 * 6
    if kind == "dct2d":
        pytest.importorskip("torch_dct")
        t = DCT2D(x_dim=8, y_dim=6)
        t._register_radial_bands()
        return t, 8 * 6
    if kind == "sht":
        t = OctahedralSHT(nlat=8)
        return t, t._sht.n_grid_points
    raise ValueError(f"unknown transform {kind!r}")


def test_radial_band_index() -> None:
    """The (ky, kx) -> band map on a rectangular FFT grid, so axis order is checked too."""
    # 3x4 grid: |fftfreq*N| gives ky=[0,1,2,1], kx=[0,1,1]; band = round(sqrt(ky^2 + kx^2)),
    # flattened row-major (y outer, x inner).
    t = FFT2D(x_dim=3, y_dim=4)
    t._register_radial_bands()
    assert torch.equal(t.radial_band_index, torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1]))
    assert t.n_radial_bands == 3


@pytest.mark.parametrize("transform", _DENSITY_TRANSFORMS)
def test_spectral_density_contract(transform: str) -> None:
    """Parseval partition, cross(x, x) == psd(x) and per-band Cauchy-Schwarz, for the 2D
    radial binning and the SHT per-degree sum alike.
    """
    t, n_points = _make_density_transform(transform)
    a = t.forward(torch.randn(2, 1, 2, n_points, 3, dtype=torch.float64))
    b = t.forward(torch.randn(2, 1, 2, n_points, 3, dtype=torch.float64))
    psd_a = t.power_spectral_density(a)
    psd_b = t.power_spectral_density(b)

    total = torch.real(a * torch.conj(a)).flatten(-3, -2).sum(dim=-2)
    torch.testing.assert_close(psd_a.sum(dim=-2), total)
    torch.testing.assert_close(t.cross_spectral_density(a, a), psd_a)

    cross = t.cross_spectral_density(a, b)
    assert torch.all(cross**2 <= psd_a * psd_b * (1 + 1e-9) + 1e-9)
