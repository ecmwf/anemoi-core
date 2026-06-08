# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import torch
from torch import Tensor
from torch.cuda.graphs import make_graphed_callables
from torch.nn import Module

LOGGER = logging.getLogger(__name__)


def _rfft_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float16:
        return torch.complex32
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"Unsupported real FFT dtype: {dtype}")


def _irfft_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex32:
        return torch.float16
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Unsupported inverse real FFT dtype: {dtype}")


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _use_cuda_ring_fft(use_cuda_ring_fft: bool | None) -> bool:
    if use_cuda_ring_fft is not None:
        return use_cuda_ring_fft
    return _bool_env("ANEMOI_USE_CUDA_RING_FFT", True)


def _is_octahedral_grid(lons_per_lat: list[int], resolution: int) -> bool:
    if len(lons_per_lat) != 2 * resolution:
        return False
    half = [20 + 4 * i for i in range(resolution)]
    return lons_per_lat == half + list(reversed(half))


def _is_n320_grid(lons_per_lat: list[int]) -> bool:
    return (
        len(lons_per_lat) == 640
        and sum(lons_per_lat) == 542_080
        and min(lons_per_lat) == 18
        and max(lons_per_lat) == 1280
        and len(set(lons_per_lat)) == 69
        and lons_per_lat == list(reversed(lons_per_lat))
    )


def _supports_cuda_ring_fft(lons_per_lat: list[int]) -> bool:
    # Keep the CUDA path to grids covered by the ring FFT tests.
    return (
        _is_octahedral_grid(lons_per_lat, 96) or _is_octahedral_grid(lons_per_lat, 320) or _is_n320_grid(lons_per_lat)
    )


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
        Legendre polynomial values.

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
    rlon : list[int]
        Number of zeros to add to the end of each rFFT output, so that each zonal wavenumber Legendre transform has the
        same shape.

    Methods
    -------
    rfft_rings_reduced_naive(x: Tensor) -> Tensor
        Performs direct real-to-complex FFT on each latitude ring of a reduced grid using a ring loop.
    rfft_rings_reduced_grouped(x: Tensor) -> Tensor
        Performs direct real-to-complex FFT on reduced-grid latitude rings grouped by longitude count.
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
        use_cuda_ring_fft: bool | None = None,
        use_graphed_rfft: bool = False,
    ) -> None:
        r"""Initializes SphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array
        use_cuda_ring_fft : bool | None, optional
            Whether to use the CUDA ring FFT extension for supported reduced grids. If ``None``, read
            ``ANEMOI_USE_CUDA_RING_FFT`` and default to ``True``.
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

        # Set offsets to start of each latitude in flattened grid dimension
        self.slon = [0] + list(np.cumsum(self.lons_per_lat))[:-1]
        self._max_m = max(self.lons_per_lat) // 2 + 1

        # Set padding for each latitude so every rFFT output ring has the same length
        self.rlon = [max(self.lons_per_lat) // 2 - nlon // 2 for nlon in self.lons_per_lat]

        unique_nlons = set(self.lons_per_lat)
        self._use_cuda_ring_rfft = False
        if len(unique_nlons) > 1:
            # Reduced grids have different ring lengths; regular grids can use one batched FFT.
            self._init_reduced_rfft_groups()
            if use_graphed_rfft:
                self.rfft_rings = self.rfft_rings_reduced_graphed
            elif self._nlon_groups:
                self.rfft_rings = self.rfft_rings_reduced_auto
                self._use_cuda_ring_rfft = _use_cuda_ring_fft(use_cuda_ring_fft) and _supports_cuda_ring_fft(
                    self.lons_per_lat
                )
            else:
                self.rfft_rings = self.rfft_rings_reduced_naive
        else:
            self.rfft_rings = self.rfft_rings_regular
        LOGGER.info(f"SphericalHarmonicTransform: Using {self.rfft_rings.__name__} for rfft_rings")

        # To have further control over the memory consumption of the graphed implementation, we
        # group latitudes together into "bands" and create one graph for each band.
        # It seems that most devices today do not have enough memory to handle a graphed global
        # rFFT.
        # 3 bands works well on our H100s with 120 GB of memory.
        number_of_latitude_bands = 3
        self.latitude_bands = []
        for band_idx in range(number_of_latitude_bands):
            start_lat = band_idx * self.nlat // number_of_latitude_bands
            end_lat = (band_idx + 1) * self.nlat // number_of_latitude_bands
            self.latitude_bands.append((start_lat, end_lat))

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

    def _init_reduced_rfft_groups(self) -> None:
        nlon_groups: dict[int, tuple[list[int], list[int]]] = {}
        for lat_idx, (slon, nlon) in enumerate(zip(self.slon, self.lons_per_lat)):
            ring_indices, slon_offsets = nlon_groups.setdefault(nlon, ([], []))
            ring_indices.append(lat_idx)
            slon_offsets.append(slon)

        self._nlon_groups = []
        self._single_rings = []
        for nlon, (ring_indices, slon_offsets) in nlon_groups.items():
            if len(ring_indices) == 1:
                self._single_rings.append((ring_indices[0], slon_offsets[0], nlon))
            else:
                self._nlon_groups.append((nlon, ring_indices, slon_offsets))

    def rfft_rings_reduced_naive(self, x: Tensor) -> Tensor:
        r"""Performs direct real-to-complex FFT on each latitude ring of a reduced grid.
        Naive (eager) implementation using rfft_rings_reduced_banded with a single band.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]
        """

        return self.rfft_rings_reduced_banded(x, start_lat=0, end_lat=self.nlat)

    def rfft_rings_reduced_banded(self, x: Tensor, start_lat: int, end_lat: int) -> Tensor:
        r"""Performs direct real-to-complex FFT on each latitude ring of a reduced grid, from start_lat to end_lat.
        Naive (eager) implementation.

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
            end_lat - start_lat,
            self._max_m,
            device=x.device,
            dtype=_rfft_complex_dtype(x.dtype),
        )

        # Do a real-to-complex FFT on each latitude
        for i, (slon, nlon) in enumerate(zip(self.slon[start_lat:end_lat], self.lons_per_lat[start_lat:end_lat])):
            output_tensor[..., i, : nlon // 2 + 1] = torch.fft.rfft(x[..., slon : slon + nlon], norm="forward")

        return output_tensor

    def rfft_rings_reduced_auto(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            if self._use_cuda_ring_rfft and x.dtype in (torch.float32, torch.float64):
                from anemoi.models.layers.ring_fft import ring_rfft

                return ring_rfft(x, self.lons_per_lat, self.truncation)
            return self.rfft_rings_reduced_grouped(x)

        return self.rfft_rings_reduced_naive(x)

    def rfft_rings_reduced_grouped(self, x: Tensor) -> Tensor:
        output_tensor = torch.zeros(
            *x.shape[:-1],
            self.nlat,
            self._max_m,
            device=x.device,
            dtype=_rfft_complex_dtype(x.dtype),
        )

        for ring_idx, slon, nlon in self._single_rings:
            output_tensor[..., ring_idx, : nlon // 2 + 1] = torch.fft.rfft(
                x[..., slon : slon + nlon],
                norm="forward",
            )

        for nlon, ring_indices, slon_offsets in self._nlon_groups:
            batch = torch.stack([x[..., slon : slon + nlon] for slon in slon_offsets], dim=-2)
            fft_result = torch.fft.rfft(batch, norm="forward")
            output_tensor[..., ring_indices, : fft_result.shape[-1]] = fft_result

        return output_tensor

    def rfft_rings_reduced_graphed(self, x: Tensor) -> Tensor:
        r"""Performs direct real-to-complex FFT on each latitude ring of a reduced grid.
        Uses graphs.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]
        """

        from functools import partial

        if x.device.type != "cuda":
            raise RuntimeError('Graphed rFFT requested but input device is not "cuda"')

        key = (tuple(x.shape), x.dtype, x.device, x.requires_grad)
        if key not in self._graphed_rfft_cache:
            sample_x = torch.zeros_like(x, requires_grad=x.requires_grad)
            with torch.amp.autocast("cuda", cache_enabled=False):
                # Separate graphs for each latitude band, but all created with a single make_graphed_callables call
                self._graphed_rfft_cache[key] = make_graphed_callables(
                    tuple(
                        partial(self.rfft_rings_reduced_banded, start_lat=latitude_band[0], end_lat=latitude_band[1])
                        for latitude_band in self.latitude_bands
                    ),
                    tuple([(sample_x,)] * len(self.latitude_bands)),
                )

        return torch.cat([f(x) for f in self._graphed_rfft_cache[key]], dim=-2)

    def rfft_rings_regular(self, x: Tensor) -> Tensor:
        """Performs direct real-to-complex FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid]

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
            field [..., grid]

        Returns
        -------
        torch.Tensor
            Spectral representation of field [..., total wavenumber l, zonal wavenumber m].
        """

        x = 2.0 * torch.pi * self.rfft_rings(x)
        x = torch.view_as_real(x)
        weight = self.weight
        if weight.device != x.device or weight.dtype != x.dtype:
            weight = weight.to(device=x.device, dtype=x.dtype)
            self.weight = weight

        rl = torch.einsum("...km, mlk -> ...lm", x[..., : self.truncation + 1, 0], weight)
        im = torch.einsum("...km, mlk -> ...lm", x[..., : self.truncation + 1, 1], weight)

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
        use_cuda_ring_fft: bool | None = None,
        use_graphed_irfft: bool = False,
    ) -> None:
        r"""Initializes InverseSphericalHarmonicTransform.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Maximum wavenumber. truncation + 1 is used to size the Legendre polynomials array.
        use_cuda_ring_fft : bool | None, optional
            Whether to use the CUDA ring FFT extension for supported reduced grids. If ``None``, read
            ``ANEMOI_USE_CUDA_RING_FFT`` and default to ``True``.
        use_graphed_irfft : bool, optional
            Whether to use CUDA graphs for the reduced grid irFFT. Default is False.
        """

        super().__init__()

        nlat = len(lons_per_lat)

        self.truncation = truncation
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Set offsets to start of each latitude in flattened grid dimension
        self.slon = [0] + list(np.cumsum(self.lons_per_lat))[:-1]

        self._use_cuda_ring_irfft = False
        if len(set(self.lons_per_lat)) > 1:
            # Reduced grids need per-ring inverse FFTs.
            if use_graphed_irfft:
                self.irfft_rings = self.irfft_rings_reduced_graphed
            else:
                self.irfft_rings = self.irfft_rings_reduced_auto
                self._use_cuda_ring_irfft = _use_cuda_ring_fft(use_cuda_ring_fft) and _supports_cuda_ring_fft(
                    self.lons_per_lat
                )
        else:
            self.irfft_rings = self.irfft_rings_regular
        LOGGER.info(f"InverseSphericalHarmonicTransform: Using {self.irfft_rings.__name__} for irfft_rings")

        # To have further control over the memory consumption of the graphed implementation, we
        # group latitudes together into "bands" and create one graph for each band.
        # It seems that most devices today do not have enough memory to handle a graphed global
        # irFFT.
        # 3 bands works well on our H100s with 120 GB of memory.
        number_of_latitude_bands = 3
        self.latitude_bands = []
        for band_idx in range(number_of_latitude_bands):
            start_lat = band_idx * self.nlat // number_of_latitude_bands
            end_lat = (band_idx + 1) * self.nlat // number_of_latitude_bands
            self.latitude_bands.append((start_lat, end_lat))

        # Compute Gaussian latitudes (don't need quadrature weights for the inverse)
        theta, _ = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials
        pct = legpoly(self.truncation, self.truncation, np.cos(theta), inverse=True)
        pct = torch.from_numpy(pct)

        self._graphed_irfft_cache = {}

        self.register_buffer("pct", pct, persistent=False)

    def irfft_rings_reduced_naive(self, x: Tensor) -> Tensor:
        """Performs inverse complex-to-real FFT on each latitude ring of a reduced grid.
        Naive (eager) implementation using irfft_rings_reduced_banded with a single band.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            field [..., grid]
        """

        return self.irfft_rings_reduced_banded(x, start_lat=0, end_lat=self.nlat)

    def irfft_rings_reduced_banded(self, x: Tensor, start_lat: int, end_lat: int) -> Tensor:
        """Performs inverse complex-to-real FFT on each latitude ring of a reduced grid, from start_lat to end_lat.
        Naive (eager) implementation.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            field [..., grid]
        """

        # Prepare zero-padded output tensor for filling with irfft
        output_tensor = torch.zeros(
            *x.shape[:-2],
            sum(self.lons_per_lat[start_lat:end_lat]),
            device=x.device,
            dtype=_irfft_real_dtype(x.dtype),
        )

        # Do a complex-to-real IFFT on each latitude
        for i, (slon, nlon) in enumerate(zip(self.slon[start_lat:end_lat], self.lons_per_lat[start_lat:end_lat])):
            output_tensor[..., slon - self.slon[start_lat] : slon - self.slon[start_lat] + nlon] = torch.fft.irfft(
                x[..., start_lat + i, :], nlon, norm="forward"
            )

        return output_tensor

    def irfft_rings_reduced_auto(self, x: Tensor) -> Tensor:
        if x.is_cuda and self._use_cuda_ring_irfft and x.dtype in (torch.complex64, torch.complex128):
            from anemoi.models.layers.ring_fft import ring_irfft

            return ring_irfft(x, self.lons_per_lat)

        return self.irfft_rings_reduced_naive(x)

    def irfft_rings_reduced_graphed(self, x: Tensor) -> Tensor:
        r"""Performs inverse complex-to-real FFT on each latitude ring of a reduced grid.
        Uses graphs.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            field [..., grid]
        """

        from functools import partial

        if x.device.type != "cuda":
            raise RuntimeError('Graphed irFFT requested but input device is not "cuda"')

        key = (tuple(x.shape), x.dtype, x.device, x.requires_grad)
        if key not in self._graphed_irfft_cache:
            sample_x = torch.zeros_like(x, requires_grad=x.requires_grad)
            with torch.amp.autocast("cuda", cache_enabled=False):
                # Separate graphs for each latitude band, but all created with a single make_graphed_callables call
                self._graphed_irfft_cache[key] = make_graphed_callables(
                    tuple(
                        partial(self.irfft_rings_reduced_banded, start_lat=latitude_band[0], end_lat=latitude_band[1])
                        for latitude_band in self.latitude_bands
                    ),
                    tuple([(sample_x,)] * len(self.latitude_bands)),
                )

        return torch.cat([f(x) for f in self._graphed_irfft_cache[key]], dim=-1)

    def irfft_rings_regular(self, x: Tensor) -> Tensor:
        """Performs inverse complex-to-real FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, zonal wavenumber m]

        Returns
        -------
        torch.Tensor
            Field [..., grid].
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
            Field [..., grid].
        """

        x = torch.view_as_real(x)
        pct = self.pct
        if pct.device != x.device or pct.dtype != x.dtype:
            pct = pct.to(device=x.device, dtype=x.dtype)
            self.pct = pct

        rl = torch.einsum("...lm, mlk -> ...km", x[..., 0], pct)
        im = torch.einsum("...lm, mlk -> ...km", x[..., 1], pct)

        x = torch.stack((rl, im), -1)
        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)

        return x
