# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import gc
import logging
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.cuda.graphs import make_graphed_callables
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import functional as F

from anemoi.models.layers.ring_fft import RingFFT

LOGGER = logging.getLogger(__name__)


def _ring_fft_bands(lons_per_lat: list[int], graphed: bool) -> tuple[ModuleList, list[slice]]:
    """Build shared ring FFTs, splitting graph captures into at most three bands."""
    bands = min(3, len(lons_per_lat)) if graphed else 1
    transforms, grid_slices = [], []
    offset = 0
    for band in range(bands):
        start = band * len(lons_per_lat) // bands
        end = (band + 1) * len(lons_per_lat) // bands
        fft = RingFFT(lons_per_lat[start:end])
        transforms.append(fft)
        grid_slices.append(slice(offset, offset + fft.points))
        offset += fft.points
    return ModuleList(transforms), grid_slices


def _capture_ring_ffts(
    functions: tuple[Callable[[Tensor], Tensor], ...], inputs: list[Tensor]
) -> tuple[Callable[[Tensor], Tensor], ...]:
    """Capture ring FFTs without collecting old CUDA graphs during a new capture."""
    samples = tuple((torch.zeros_like(band, requires_grad=band.requires_grad),) for band in inputs)
    # Destroying a previous graph can issue CUDA operations that invalidate an
    # active capture. Collect cycles beforehand and defer automatic collection.
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled(), cache_enabled=False):
            return make_graphed_callables(functions, samples)
    finally:
        if gc_enabled:
            gc.enable()


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

    def __init__(self, lons_per_lat: list[int], truncation: int, use_graphed_rfft: bool = False) -> None:
        r"""Initializes SphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array
        use_graphed_rfft : bool, optional
            Whether to use CUDA graphs for the reduced grid rFFT. Default is False.
        """

        super().__init__()

        self.lons_per_lat = lons_per_lat
        self.nlat = len(self.lons_per_lat)
        self.truncation = truncation
        assert (
            0 < self.truncation <= self.nlat
        ), f"Truncation {self.truncation} must be between 1 and number of latitudes {self.nlat}"
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            self._ring_ffts, self._ring_grid_slices = _ring_fft_bands(self.lons_per_lat, use_graphed_rfft)
            if use_graphed_rfft:
                self.rfft_rings = self.rfft_rings_reduced_graphed
            else:
                self.rfft_rings = self.rfft_rings_reduced
        else:
            self.rfft_rings = self.rfft_rings_regular
        LOGGER.info(f"SphericalHarmonicTransform: Using {self.rfft_rings.__name__} for rfft_rings")

        # Compute Gaussian latitudes and quadrature weights
        theta, weight = legendre_gauss_weights(self.nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta))
        pct = torch.from_numpy(pct)

        # Premultiple associated Legendre polynomials by quadrature weights
        weight = torch.from_numpy(weight)
        weight = torch.einsum("mlk, k -> mlk", pct, weight)

        self._graphed_rfft_cache = {}

        self.register_buffer("weight", weight, persistent=False)

    def rfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Transform equal-length rings together, with an explicit FFT adjoint for backward."""
        return self._ring_ffts[0].rfft(x)

    def rfft_rings_reduced_graphed(self, x: Tensor) -> Tensor:
        r"""Performs direct real-to-complex FFT on each latitude ring of a reduced grid.
        Uses graphs.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid].

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m].
        """

        if x.device.type != "cuda":
            raise RuntimeError('Graphed rfft requested but input device is not "cuda"')

        inputs = [x[..., grid_slice] for grid_slice in self._ring_grid_slices]
        key = (tuple(x.shape), x.dtype, x.device, x.requires_grad)
        if key not in self._graphed_rfft_cache:
            self._graphed_rfft_cache[key] = _capture_ring_ffts(tuple(fft.rfft for fft in self._ring_ffts), inputs)

        modes = max(self.lons_per_lat) // 2 + 1
        return torch.cat(
            [
                F.pad(fn(band), (0, modes - fft.modes))
                for fn, band, fft in zip(self._graphed_rfft_cache[key], inputs, self._ring_ffts)
            ],
            dim=-2,
        )

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

    def __init__(self, lons_per_lat: list[int], truncation: int, use_graphed_irfft: bool = False) -> None:
        r"""Initializes InverseSphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array.
        use_graphed_irfft : bool, optional
            Whether to use CUDA graphs for the reduced grid irFFT. Default is False.
        """

        super().__init__()

        nlat = len(lons_per_lat)

        self.truncation = truncation
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            self._ring_ffts, self._ring_grid_slices = _ring_fft_bands(self.lons_per_lat, use_graphed_irfft)
            if use_graphed_irfft:
                self.irfft_rings = self.irfft_rings_reduced_graphed
            else:
                self.irfft_rings = self.irfft_rings_reduced
        else:
            self.irfft_rings = self.irfft_rings_regular
        LOGGER.info(f"InverseSphericalHarmonicTransform: Using {self.irfft_rings.__name__} for irfft_rings")

        # Compute Gaussian latitudes (don't need quadrature weights for the inverse)
        theta, _ = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta), inverse=True)
        pct = torch.from_numpy(pct)

        self._graphed_irfft_cache = {}

        self.register_buffer("pct", pct, persistent=False)

    def irfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Transform equal-length rings together, with an explicit FFT adjoint for backward."""
        return self._ring_ffts[0].irfft(x)

    def irfft_rings_reduced_graphed(self, x: Tensor) -> Tensor:
        r"""Performs inverse complex-to-real FFT on each latitude ring of a reduced grid.
        Uses graphs.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m].

        Returns
        -------
        torch.Tensor
            field [..., grid].
        """

        if x.device.type != "cuda":
            raise RuntimeError('Graphed irfft requested but input device is not "cuda"')

        inputs = list(x.split([fft.nlat for fft in self._ring_ffts], dim=-2))
        key = (tuple(x.shape), x.dtype, x.device, x.requires_grad)
        if key not in self._graphed_irfft_cache:
            self._graphed_irfft_cache[key] = _capture_ring_ffts(tuple(fft.irfft for fft in self._ring_ffts), inputs)

        return torch.cat([fn(band) for fn, band in zip(self._graphed_irfft_cache[key], inputs)], dim=-1)

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
