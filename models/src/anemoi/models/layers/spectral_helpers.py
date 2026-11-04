# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from anemoi.models.layers.ring_fft import shared_ring_fft

LOGGER = logging.getLogger(__name__)


LatitudeGrid = Literal["legendre-gauss", "equiangular-poles"]


def latitude_quadrature(nlat: int, latitude_grid: LatitudeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return north-to-south colatitudes and integration weights in cos(theta).

    ``equiangular-poles`` includes both poles: theta[j] = pi * j / (nlat - 1).
    Clenshaw-Curtis quadrature integrates polynomials through degree nlat - 1
    on these nodes. See https://dlmf.nist.gov/3.5.iv for the quadrature rule.
    """
    if nlat < 1:
        raise ValueError("nlat must be positive")
    if latitude_grid == "legendre-gauss":
        nodes, weights = legendre_gauss_weights(nlat)
        return np.arccos(nodes[::-1]), weights[::-1].copy()
    if latitude_grid == "equiangular-poles":
        if nlat < 2:
            raise ValueError("equiangular-poles requires at least two latitude rows")
        intervals = nlat - 1
        theta = np.linspace(0.0, np.pi, nlat, dtype=np.float64)
        k = np.arange(1, intervals // 2 + 1, dtype=np.float64)
        coefficients = 2.0 / (4.0 * k**2 - 1.0)
        if intervals % 2 == 0:
            coefficients[-1] *= 0.5
        weights = (2.0 / intervals) * (1.0 - np.cos(2.0 * theta[:, None] * k) @ coefficients)
        # The endpoint nodes have half the weight of the corresponding cosine sum.
        weights[[0, -1]] *= 0.5
        return theta, weights
    raise ValueError(f"Unknown latitude_grid={latitude_grid!r}; use 'legendre-gauss' or 'equiangular-poles'")


def _sht_quadrature(nlat: int, truncation: int, latitude_grid: LatitudeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Validate the latitude sampling and construct the shared SHT/ISHT quadrature."""
    max_truncation = (nlat - 1) // 2 if latitude_grid == "equiangular-poles" else nlat - 1
    if not 0 <= truncation <= max_truncation:
        raise ValueError(f"Truncation must be between 0 and {max_truncation} for {nlat} {latitude_grid} latitudes")
    return latitude_quadrature(nlat, latitude_grid)


def legendre_gauss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b].

    Parameters
    ----------
    n : int
        Number of latitudes at weight to compute weights and latitudes.
    a : float, optional
        Left endpoint of the interval. Default is -1.0.
    b : float, optional
        Right endpoint of the interval. Default is 1.0.

    Returns
    -------
    xlg : np.ndarray
        Legendre-Gauss nodes (latitudes) on the interval [a, b].
    wlg : np.ndarray
        Legendre-Gauss weights on the interval [a, b].
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def legpoly(
    mmax: int,
    lmax: int,
    x: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    r"""Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax + 1, lmax + 1, len(x)).

    Parameters
    ----------
    mmax : int
        Maximum zonal wavenumber. mmax + 1 is used to size the Legendre polynomials array.
    lmax : int
        Maximum total wavenumber. lmax + 1 is used to size the Legendre polynomials array.
    x : np.ndarray
        Points at which to evaluate the Legendre polynomials. Should be in the range [-1, 1].
    inverse : bool, optional
        Whether to invert the normalisation factor or not. Should be set to True for the inverse Legendre transform and
        False for the forward Legendre transform. Default is False.

    Returns
    -------
    np.ndarray
        Associated Legendre polynomials with shape (mmax + 1, lmax + 1, len(x)).

    Notes
    -----
    This is derived from the version in torch-harmonics.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3:
    Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic
    Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    # Compute the tensor P^m_n:
    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax + 1, nmax + 1, len(x)), dtype=np.float64)

    norm_factor = np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # Fill the diagonal and the lower diagonal
    for n in range(1, nmax + 1):
        vdm[n - 1, n, :] = np.sqrt(2 * n + 1) * x * vdm[n - 1, n - 1, :]
        vdm[n, n, :] = np.sqrt((2 * n + 1) * (1 + x) * (1 - x) / 2 / n) * vdm[n - 1, n - 1, :]

    # Fill the remaining values on the upper triangle and multiply b
    for n in range(2, nmax + 1):
        for m in range(0, n - 1):
            vdm[m, n, :] = (
                x * np.sqrt((2 * n - 1) / (n - m) * (2 * n + 1) / (n + m)) * vdm[m, n - 1, :]
                - np.sqrt((n + m - 1) / (n - m) * (2 * n + 1) / (2 * n - 3) * (n - m - 1) / (n + m)) * vdm[m, n - 2, :]
            )

    vdm = vdm[: mmax + 1, : lmax + 1]

    return vdm


class SphericalHarmonicTransform(Module):
    r"""Generic class for performing direct (AKA forward) transforms from a global gridded tensor to a space with a
    spherical harmonic basis.

    Attributes
    ----------
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    nlat : int
        Number of latitudes in the grid, from pole to pole.
    truncation : int
        Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array
    n_grid_points : int
        Total number of grid points in the global grid.

    Methods
    -------
    rfft_rings_reduced(x: Tensor) -> Tensor
        Performs direct real-to-complex FFT on each latitude ring of a reduced grid.
    rfft_rings_regular(x: Tensor) -> Tensor
        Performs direct real-to-complex FFT on each latitude ring of a regular grid.
    forward(x: Tensor) -> Tensor
        Performs direct SHT transform (Fourier transform followed by Legendre transform).

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        truncation: int,
        latitude_grid: LatitudeGrid = "legendre-gauss",
    ) -> None:
        r"""Initializes SphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array
        latitude_grid : {"legendre-gauss", "equiangular-poles"}, optional
            Gaussian latitude nodes or equally spaced rows including both poles, ordered north to south.
        """

        super().__init__()

        self.lons_per_lat = lons_per_lat
        self.nlat = len(self.lons_per_lat)
        self.truncation = truncation
        theta, weight = _sht_quadrature(self.nlat, truncation, latitude_grid)
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            self.rfft_rings = self.rfft_rings_reduced
        else:
            self.rfft_rings = self.rfft_rings_regular
        LOGGER.info(f"SphericalHarmonicTransform: Using {self.rfft_rings.__name__} for rfft_rings")

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta))
        pct = torch.from_numpy(pct)

        # Premultiple associated Legendre polynomials by quadrature weights
        weight = torch.from_numpy(weight)
        weight = torch.einsum("mlk, k -> mlk", pct, weight)

        self.register_buffer("weight", weight, persistent=False)

    def rfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Fourier coefficients ``[..., latitude, m]`` on each ring of a reduced grid, up to the truncation.

        All rings go through a few FFTs together, see ``RingFFT``.
        """
        modes = min(self.truncation + 1, max(self.lons_per_lat) // 2 + 1)
        return shared_ring_fft(tuple(self.lons_per_lat)).rfft(x, modes)

    def rfft_rings_regular(self, x: Tensor) -> Tensor:
        """Performs direct real-to-complex FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid].

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m].
        """

        return torch.fft.rfft(x.reshape(*x.shape[:-1], self.nlat, self.lons_per_lat[0]), norm="forward")

    def forward(self, x: Tensor) -> Tensor:
        """Performs direct SHT transform (Fourier transform followed by Legendre transform).

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid].

        Returns
        -------
        torch.Tensor
            spectral representation of field [..., total wavenumber l, zonal wavenumber m].
        """

        x = 2.0 * torch.pi * self.rfft_rings(x)
        x = torch.view_as_real(x)

        rl = torch.einsum("...km, mlk -> ...lm", x[..., : self.truncation + 1, 0], self.weight.to(x.dtype))
        im = torch.einsum("...km, mlk -> ...lm", x[..., : self.truncation + 1, 1], self.weight.to(x.dtype))

        x = torch.stack((rl, im), -1)
        x = torch.view_as_complex(x)

        return x


class InverseSphericalHarmonicTransform(Module):
    r"""Generic class for performing inverse (AKA backward) transforms from a spectral representation to a global gridded
    tensor.

    Attributes
    ----------
    truncation : int
        Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array
    nlat : int
        Number of latitudes in the grid, from pole to pole.
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    n_grid_points : int
        Total number of grid points in the global grid.

    Methods
    -------
    irfft_rings_reduced(x: Tensor) -> Tensor
        Performs inverse complex-to-real FFT on each latitude ring of a reduced grid.
    irfft_rings_regular(x: Tensor) -> Tensor
        Performs inverse complex-to-real FFT on each latitude ring of a regular grid.
    forward(x: Tensor) -> Tensor
        Performs inverse SHT transform (inverse Legendre transform followed by inverse Fourier transform).

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        truncation: int,
        latitude_grid: LatitudeGrid = "legendre-gauss",
    ) -> None:
        r"""Initializes InverseSphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array.
        latitude_grid : {"legendre-gauss", "equiangular-poles"}, optional
            Gaussian latitude nodes or equally spaced rows including both poles, ordered north to south.
        """

        super().__init__()

        nlat = len(lons_per_lat)
        theta, _ = _sht_quadrature(nlat, truncation, latitude_grid)

        self.truncation = truncation
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            self.irfft_rings = self.irfft_rings_reduced
        else:
            self.irfft_rings = self.irfft_rings_regular
        LOGGER.info(f"InverseSphericalHarmonicTransform: Using {self.irfft_rings.__name__} for irfft_rings")

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta), inverse=True)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def irfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Ring values ``[..., grid]`` of a reduced grid from their Fourier coefficients ``[..., latitude, m]``.

        All rings go through a few FFTs together, see ``RingFFT``.
        """
        return shared_ring_fft(tuple(self.lons_per_lat)).irfft(x)

    def irfft_rings_regular(self, x: Tensor) -> Tensor:
        """Performs inverse complex-to-real FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m].

        Returns
        -------
        torch.Tensor
            field [..., grid].
        """

        return torch.fft.irfft(x, self.lons_per_lat[0], norm="forward").reshape(*x.shape[:-2], self.n_grid_points)

    def forward(self, x: Tensor) -> Tensor:
        """Performs inverse SHT transform (inverse Legendre transform followed by inverse Fourier transform).

        Parameters
        ----------
        x : torch.Tensor
            spectral representation of field [..., total wavenumber l, zonal wavenumber m].

        Returns
        -------
        torch.Tensor
            field [..., grid].
        """

        x = torch.view_as_real(x)

        rl = torch.einsum("...lm, mlk -> ...km", x[..., 0], self.pct.to(x.dtype))
        im = torch.einsum("...lm, mlk -> ...km", x[..., 1], self.pct.to(x.dtype))

        x = torch.stack((rl, im), -1).to(x.dtype)
        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)

        return x
