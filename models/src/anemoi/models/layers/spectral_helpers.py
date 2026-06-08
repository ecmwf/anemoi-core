# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

LOGGER = logging.getLogger(__name__)


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
    slon : list[int]
        Starting index of each latitude ring in the flattened grid dimension.

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

    def __init__(self, lons_per_lat: list[int], truncation: int) -> None:
        r"""Initializes SphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array
        """

        super().__init__()

        self.lons_per_lat = lons_per_lat
        self.nlat = len(self.lons_per_lat)
        self.truncation = truncation
        assert (
            0 < self.truncation <= self.nlat
        ), f"Truncation {self.truncation} must be between 1 and number of latitudes {self.nlat}"
        self.n_grid_points = sum(self.lons_per_lat)

        # Set offsets to start of each latitude in flattened grid dimension
        self.slon = [0] + list(np.cumsum(self.lons_per_lat))[:-1]

        # Choose the appropriate rfft method based on the grid structure
        # Every hemisphere latitude is different (e.g. octahedral grid)
        # No optimised version available yet
        if len(set(self.lons_per_lat[: self.nlat])) == len(self.lons_per_lat[: self.nlat]):
            LOGGER.info(
                "SphericalHarmonicTransform: All latitudes have different number of longitude points, using"
                "rfft_rings_reduced"
            )
            self.rfft_rings = self.rfft_rings_reduced
        # At least two latitudes in a hemisphere are the same (e.g. classic reduced grid)
        # Use the optimised grouped version
        elif len(set(self.lons_per_lat[: self.nlat])) > 1:
            LOGGER.info("SphericalHarmonicTransform: Using rfft_rings_reduced_grouped for reduced grid")
            self.rfft_rings = self.rfft_rings_reduced_grouped
            self._nlon_groups = {}

            # For each latitude
            for lat_idx, (slon, nlon) in enumerate(zip(self.slon, self.lons_per_lat)):
                self._nlon_groups.setdefault(nlon, ([], []))
                self._nlon_groups[nlon][0].append(lat_idx)  # First list stores global latitude indices
                self._nlon_groups[nlon][1].append(slon)  # Second stores the offsets for each latitude
        # Else it must be a regular grid
        else:
            LOGGER.info("SphericalHarmonicTransform: Using rfft_rings_regular for regular grid")
            self.rfft_rings = self.rfft_rings_regular

        # Compute Gaussian latitudes and quadrature weights
        theta, weight = legendre_gauss_weights(self.nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta))
        pct = torch.from_numpy(pct)

        # Premultiple associated Legendre polynomials by quadrature weights
        weight = torch.from_numpy(weight)
        weight = torch.einsum("mlk, k -> mlk", pct, weight)

        self.register_buffer("weight", weight, persistent=False)

    def rfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Performs direct real-to-complex FFT on each latitude ring of a reduced grid.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]
        """

        # Prepare zero-padded output tensor for filling with rfft
        output_tensor = torch.zeros(
            *x.shape[:-1],
            self.nlat,
            max(self.lons_per_lat) // 2 + 1,
            device=x.device,
            dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128,
        )

        # Do a real-to-complex FFT on each latitude
        for i, (slon, nlon) in enumerate(zip(self.slon, self.lons_per_lat)):
            output_tensor[..., i, : nlon // 2 + 1] = torch.fft.rfft(x[..., slon : slon + nlon], norm="forward")

        return output_tensor

    def rfft_rings_reduced_grouped(self, x: Tensor) -> Tensor:
        """Performs direct real-to-complex FFT on each latitude ring of a reduced grid, grouping latitudes with the same
        number of longitudinal points.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]
        """

        ring_results = [None] * self.nlat
        max_m = max(self.lons_per_lat) // 2 + 1

        # For each latitude group
        for nlon, (ring_indices, slon_offsets) in self._nlon_groups.items():
            # Collect all latitude lines in this group into one tensor
            batch = torch.stack([x[..., slon : slon + nlon] for slon in slon_offsets], dim=-2)

            # Do the rFFT
            fft_result = torch.fft.rfft(batch, norm="forward")

            # Pad the rFFT output
            if fft_result.shape[-1] < max_m:
                fft_result = torch.nn.functional.pad(fft_result, (0, max_m - fft_result.shape[-1]))

            # Scatter the results back to the correct latitude lines in the output tensor
            for j, ring_idx in enumerate(ring_indices):
                ring_results[ring_idx] = fft_result[..., j : j + 1, :]

        # Cat into one output tensor
        return torch.cat(ring_results, dim=-2)

    def rfft_rings_regular(self, x: Tensor) -> Tensor:
        """Performs direct real-to-complex FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]
        """

        return torch.fft.rfft(x.reshape(*x.shape[:-1], self.nlat, self.lons_per_lat[0]), norm="forward")

    def forward(self, x: Tensor) -> Tensor:
        """Performs direct SHT transform (Fourier transform followed by Legendre transform).

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

        Returns
        -------
        torch.Tensor
            spectral representation of field [..., total wavenumber l, zonal wavenumber m]
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

    def __init__(self, lons_per_lat: list[int], truncation: int) -> None:
        r"""Initializes InverseSphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array.
        """

        super().__init__()

        nlat = len(lons_per_lat)

        self.truncation = truncation
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            LOGGER.info("InverseSphericalHarmonicTransform: Using irfft_rings_reduced for reduced grid")
            self.irfft_rings = self.irfft_rings_reduced
        else:
            LOGGER.info("InverseSphericalHarmonicTransform: Using irfft_rings_regular for regular grid")
            self.irfft_rings = self.irfft_rings_regular

        # Compute Gaussian latitudes (don't need quadrature weights for the inverse)
        theta, _ = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta), inverse=True)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def irfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Performs inverse complex-to-real FFT on each latitude ring of a reduced grid.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            field [..., grid]
        """

        irfft = [torch.fft.irfft(x[..., t, :], nlon, norm="forward") for t, nlon in enumerate(self.lons_per_lat)]

        return torch.cat(
            tensors=irfft,
            dim=-1,
        )

    def irfft_rings_regular(self, x: Tensor) -> Tensor:
        """Performs inverse complex-to-real FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            field [..., grid]
        """

        return torch.fft.irfft(x, self.lons_per_lat[0], norm="forward").reshape(*x.shape[:-2], self.n_grid_points)

    def forward(self, x: Tensor) -> Tensor:
        """Performs inverse SHT transform (inverse Legendre transform followed by inverse Fourier transform).

        Parameters
        ----------
        x : torch.Tensor
            spectral representation of field [..., total wavenumber l, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            field [..., grid]
        """

        x = torch.view_as_real(x)

        rl = torch.einsum("...lm, mlk -> ...km", x[..., 0], self.pct.to(x.dtype))
        im = torch.einsum("...lm, mlk -> ...km", x[..., 1], self.pct.to(x.dtype))

        x = torch.stack((rl, im), -1).to(x.dtype)
        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)

        return x
