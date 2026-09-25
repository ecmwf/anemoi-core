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
from collections.abc import Iterator
from dataclasses import dataclass

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
    """Create ring FFT modules for one latitude band, or up to three bands for CUDA graphs."""
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
    """Capture ring FFTs with garbage collection suspended, so freeing old CUDA graphs cannot disturb the capture."""
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


def legendre_by_degree(truncation: int, x: np.ndarray, device: torch.device | str = "cpu") -> Iterator[Tensor]:
    r"""Yield (-1)^m c^l_m P^l_m(x) for each total wavenumber l.

    The recurrence uses float64 on ``device`` and keeps only the two preceding degrees, so its memory
    does not grow with the square of the truncation.

    Parameters
    ----------
    truncation : int
        Maximum total wavenumber.
    x : np.ndarray
        Points at which to evaluate the Legendre polynomials, shape (n,). Should be in the range [-1, 1].
    device : torch.device | str, optional
        Device on which the values are computed. Default is the CPU.

    Yields
    ------
    Tensor
        For l = 0, 1, ..., truncation, a float64 tensor of shape (truncation + 1, n) indexed by the zonal
        wavenumber m, with zeros for m > l. Values use the inverse transform's normalisation,
        with 1 / (4 pi) at l = 0.

    Notes
    -----
    Based on the recurrence in torch-harmonics.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3:
    Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic
    Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    # The factors of the recurrence are worked out once with numpy on the CPU. The loop below then only
    # multiplies and subtracts, which round the same way on every device, so the values are identical
    # wherever they are computed.
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = np.arange(truncation + 1, dtype=np.float64)[:, None]
    k = np.arange(truncation + 1, dtype=np.float64)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Weights of the degree before and the degree before that, for degree n and order k <= n - 2.
        from_previous = np.sqrt((2 * n - 1) / (n - k) * (2 * n + 1) / (n + k))
        from_before_previous = np.sqrt((n + k - 1) / (n - k) * (2 * n + 1) / (2 * n - 3) * (n - k - 1) / (n + k))
        # Weights that take the diagonal value m = l of the degree before to the two top orders of degree n.
        to_below_diagonal = np.sqrt(2 * n[:, 0] + 1)
        to_diagonal = np.sqrt((2 * n + 1) * (1 + x) * (1 - x) / 2 / n)

    def on_device(values: np.ndarray) -> Tensor:
        return torch.from_numpy(values).to(device)

    x, from_previous, from_before_previous = on_device(x), on_device(from_previous), on_device(from_before_previous)
    to_below_diagonal, to_diagonal = on_device(to_below_diagonal), on_device(to_diagonal)

    previous = torch.zeros(truncation + 1, x.shape[-1], dtype=torch.float64, device=x.device)
    current = torch.zeros_like(previous)
    current[0] = 1.0 / (4 * np.pi)
    yield current

    for degree in range(1, truncation + 1):
        following = torch.zeros_like(current)
        below = degree - 1  # orders 0 .. degree - 2 follow from the two degrees before
        following[:below] = (
            x * from_previous[degree, :below, None] * current[:below]
            - from_before_previous[degree, :below, None] * previous[:below]
        )
        following[degree - 1] = to_below_diagonal[degree] * x * current[degree - 1]
        following[degree] = to_diagonal[degree] * current[degree - 1]
        previous, current = current, following
        yield current


# The orders m are split into at most this many blocks, each at least this wide.
BLOCKS_OF_ORDERS = 8
MIN_ORDERS_PER_BLOCK = 16


def first_orders_of_blocks(truncation: int) -> list[int]:
    """Return the first order m of each block covering orders 0 through truncation."""
    width = max(MIN_ORDERS_PER_BLOCK, -(-(truncation + 1) // BLOCKS_OF_ORDERS))
    return list(range(0, truncation + 1, width))


def hemisphere_legendre_blocks(
    nlat: int, truncation: int, dtype: torch.dtype, device: torch.device
) -> list[tuple[int, Tensor]]:
    r"""Build Legendre tables on northern Gaussian latitudes, grouped by order and degree parity.

    At the corresponding southern latitude, P^l_m equals (-1)^(l + m) times its northern value.
    Only northern latitudes are stored. Writing l = 2 j + r separates even and odd degrees.

    Since P^l_m is zero for l < m, each block starting at order m0 stores degrees from
    l = 2 j0 onward, where j0 = m0 // 2. This omits the leading zero degrees in each block.

    Parameters
    ----------
    nlat : int
        Number of Gaussian latitudes from pole to pole. Must be even.
    truncation : int
        Maximum wavenumber.
    dtype : torch.dtype
        Precision of the returned tables. The polynomials are computed in float64, one degree at a time.
    device : torch.device
        Device on which the tables are computed and stored.

    Returns
    -------
    list[tuple[int, Tensor]]
        Pairs of (m0, table), where m0 is the block's first order. Each table has shape
        (number of orders, 2, n_j - j0, nlat // 2), indexed by
        (m - m0, r, j - j0, northern latitude), with n_j = (truncation + 2) // 2.
        Values use the inverse transform's normalisation. Entries with m > l or l > truncation are zero.
    """
    if nlat % 2:
        raise ValueError(f"A Gaussian grid has an even number of latitudes, got {nlat}.")

    # The Gauss-Legendre nodes run from south to north; keep the northern half, ordered from the pole.
    cos_colat, _ = legendre_gauss_weights(nlat)
    cos_colat = np.flip(cos_colat)[: nlat // 2]

    n_j = (truncation + 2) // 2
    first_orders = first_orders_of_blocks(truncation)
    blocks = [
        (m0, torch.zeros(m1 - m0, 2, n_j - m0 // 2, nlat // 2, dtype=dtype, device=device))
        for m0, m1 in zip(first_orders, first_orders[1:] + [truncation + 1])
    ]
    for degree, values in enumerate(legendre_by_degree(truncation, cos_colat, device)):
        for m0, table in blocks:
            if m0 > degree:
                break
            top = min(m0 + table.shape[0], degree + 1)
            table[: top - m0, degree % 2, degree // 2 - m0 // 2] = values[m0:top]
    return blocks


@dataclass
class LegendreTableSet:
    """Legendre blocks, quadrature weights and signs (-1)^m for one grid, truncation, dtype and device."""

    truncation: int
    legendre_blocks: list[tuple[int, Tensor]]
    quadrature_weight: Tensor
    zonal_sign: Tensor

    @classmethod
    def build(cls, nlat: int, truncation: int, dtype: torch.dtype, device: torch.device) -> "LegendreTableSet":
        LOGGER.info(f"Building Legendre tables for {nlat} latitudes up to T{truncation} ({dtype}, {device})")
        # The forward direction weights each northern ring (and its southern mirror) by its Gaussian
        # quadrature weight. The factor 8 pi^2 combines the 2 pi that turns the ring FFT's average over
        # longitude into an integral with the 4 pi between the forward and inverse normalisation.
        _, weight = legendre_gauss_weights(nlat)
        weight = 8 * np.pi**2 * np.flip(weight)[: nlat // 2]
        return cls(
            truncation=truncation,
            legendre_blocks=hemisphere_legendre_blocks(nlat, truncation, dtype, device),
            quadrature_weight=torch.from_numpy(weight.copy()).to(device=device, dtype=dtype),
            zonal_sign=(-1.0) ** torch.arange(truncation + 1, dtype=dtype, device=device),
        )


class LegendreTables:
    """Cache Legendre tables for spherical harmonic transforms in this process.

    Each (nlat, dtype, device) has one table set, built on first use. A higher truncation replaces
    the cached set; lower truncations use slices. Tables remain cached until ``clear`` is called.

    Building tables under ``FakeTensorMode`` stores tensors without data and breaks subsequent calls
    with real inputs. ``torch.compile`` with the ``eager`` and ``aot_eager`` backends builds real
    tables (checked with PyTorch 2.9.1).
    """

    def __init__(self) -> None:
        self._tables: dict[tuple[int, torch.dtype, torch.device], LegendreTableSet] = {}

    def get(self, nlat: int, truncation: int, dtype: torch.dtype, device: torch.device) -> LegendreTableSet:
        """Return cached tables covering at least ``truncation``, building or replacing them as needed."""
        key = (nlat, dtype, device)
        tables = self._tables.get(key)
        if tables is None or tables.truncation < truncation:
            # Tables are built as ordinary tensors even when first asked for in inference mode, since
            # training may use them later.
            with torch.inference_mode(False):
                tables = LegendreTableSet.build(nlat, truncation, dtype, device)
            self._tables[key] = tables
        return tables

    def clear(self) -> None:
        """Clear the cache. Subsequent calls rebuild the tables."""
        self._tables.clear()

    def stored_bytes(self) -> int:
        """Return the total size of cached tensors in bytes."""
        return sum(
            sum(table.nbytes for _, table in t.legendre_blocks) + t.quadrature_weight.nbytes + t.zonal_sign.nbytes
            for t in self._tables.values()
        )


LEGENDRE_TABLES = LegendreTables()


def _transform_tables(
    nlat: int, truncation: int, dtype: torch.dtype, device: torch.device
) -> tuple[list[tuple[int, Tensor]], Tensor, Tensor]:
    """Return cached Legendre blocks, quadrature weights and signs (-1)^m for this truncation.

    Select blocks starting at m <= truncation and slice their order and degree axes to the required
    size. See ``hemisphere_legendre_blocks`` for the layout.
    """
    tables = LEGENDRE_TABLES.get(nlat, truncation, dtype, device)
    n_j = (truncation + 2) // 2
    blocks = [
        (m0, table[: truncation + 1 - m0, :, : n_j - m0 // 2])
        for m0, table in tables.legendre_blocks
        if m0 <= truncation
    ]
    return blocks, tables.quadrature_weight, tables.zonal_sign[: truncation + 1]


class SphericalHarmonicTransform(Module):
    r"""Transform a field on a Gaussian grid to spherical harmonic coefficients.

    Legendre tables are built on first use and shared by transforms with the same latitude count,
    precision and device. See ``LegendreTables``.

    Attributes
    ----------
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    nlat : int
        Number of latitudes in the grid, from pole to pole.
    truncation : int
        Maximum total wavenumber; coefficients include degrees 0 through truncation.
    n_grid_points : int
        Total number of grid points in the global grid.

    Methods
    -------
    rfft_rings_reduced(x: Tensor) -> Tensor
        Compute real-to-complex FFTs on a reduced grid, grouping rings by length.
    rfft_rings_regular(x: Tensor) -> Tensor
        Compute real-to-complex FFTs on a regular grid.
    forward(x: Tensor) -> Tensor
        Apply the ring FFTs followed by the Legendre transform.

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(self, lons_per_lat: list[int], truncation: int, use_graphed_rfft: bool = False) -> None:
        r"""Initialize the forward SHT for the given grid and truncation.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole. The number of rings must
            be even.
        truncation : int
            Maximum total wavenumber; coefficients include degrees 0 through truncation.
        use_graphed_rfft : bool, optional
            Use CUDA graphs for the ring FFTs on a reduced grid. Default is False.
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

        if self.nlat % 2:
            raise ValueError(f"A Gaussian grid has an even number of latitudes, got {self.nlat}.")

        self._graphed_rfft_cache = {}

    def rfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Compute real-to-complex FFTs, grouping rings by length and using the adjoint for gradients."""
        return self._ring_ffts[0].rfft(x)

    def rfft_rings_reduced_graphed(self, x: Tensor) -> Tensor:
        r"""Compute real-to-complex FFTs on a reduced grid using CUDA graphs.

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
        """Compute real-to-complex FFTs on each latitude ring of a regular grid.

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
        """Apply the ring FFTs followed by the Legendre transform.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid].

        Returns
        -------
        torch.Tensor
            spectral representation of field [..., total wavenumber l, zonal wavenumber m].
        """

        x = torch.view_as_real(self.rfft_rings(x)[..., : self.truncation + 1])  # [..., latitude, m, real/imaginary]
        blocks, quadrature_weight, zonal_sign = _transform_tables(self.nlat, self.truncation, x.dtype, x.device)

        # Pair each northern ring with its mirror in the south. Degrees with l + m even take the sum of
        # the two rings and degrees with l + m odd the difference: with l = 2 j + r, that is
        # north + (-1)^(m + r) * south.
        half = self.nlat // 2
        north = x[..., :half, :, :]
        south = x[..., half:, :, :].flip(-3) * zonal_sign[:, None]
        x = torch.stack((north + south, north - south), dim=-2)  # [..., northern latitude, m, r, real/imaginary]
        x = x * quadrature_weight[:, None, None, None]

        # Each block of orders gives its degrees from its first j on; the degrees below are zero. Keeping
        # (m, r) in the table's order lets the contraction use the table in place, without a copy.
        x = torch.cat(
            [
                F.pad(
                    torch.einsum("...kmrc, mrjk -> ...mrjc", x[..., m0 : m0 + table.shape[0], :, :], table),
                    (0, 0, m0 // 2, 0),
                )
                for m0, table in blocks
            ],
            dim=-4,
        )  # [..., m, r, j, real/imaginary]
        x = x.permute(*range(x.dim() - 4), -2, -3, -4, -1).flatten(-4, -3)[..., : self.truncation + 1, :, :]

        return torch.view_as_complex(x.contiguous())


class InverseSphericalHarmonicTransform(Module):
    r"""Transform spherical harmonic coefficients to a field on a Gaussian grid.

    Legendre tables are built on first use and shared by transforms with the same latitude count,
    precision and device. See ``LegendreTables``.

    Attributes
    ----------
    truncation : int
        Maximum total wavenumber; coefficients include degrees 0 through truncation.
    nlat : int
        Number of latitudes in the grid, from pole to pole.
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    n_grid_points : int
        Total number of grid points in the global grid.

    Methods
    -------
    irfft_rings_reduced(x: Tensor) -> Tensor
        Compute complex-to-real FFTs on a reduced grid, grouping rings by length.
    irfft_rings_regular(x: Tensor) -> Tensor
        Compute complex-to-real FFTs on a regular grid.
    forward(x: Tensor) -> Tensor
        Apply the inverse Legendre transform followed by the inverse ring FFTs.

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(self, lons_per_lat: list[int], truncation: int, use_graphed_irfft: bool = False) -> None:
        r"""Initialize the inverse SHT for the given grid and truncation.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole. The number of rings must
            be even.
        truncation : int
            Maximum total wavenumber; coefficients include degrees 0 through truncation.
        use_graphed_irfft : bool, optional
            Use CUDA graphs for the inverse ring FFTs on a reduced grid. Default is False.
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

        if nlat % 2:
            raise ValueError(f"A Gaussian grid has an even number of latitudes, got {nlat}.")

        self._graphed_irfft_cache = {}

    def irfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Compute complex-to-real FFTs, grouping rings by length and using the adjoint for gradients."""
        return self._ring_ffts[0].irfft(x)

    def irfft_rings_reduced_graphed(self, x: Tensor) -> Tensor:
        r"""Compute complex-to-real FFTs on a reduced grid using CUDA graphs.

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
        """Compute complex-to-real FFTs on each latitude ring of a regular grid.

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
        """Apply the inverse Legendre transform followed by the inverse ring FFTs.

        Parameters
        ----------
        x : torch.Tensor
            spectral representation of field [..., total wavenumber l, zonal wavenumber m].

        Returns
        -------
        torch.Tensor
            field [..., grid].
        """

        x = torch.view_as_real(x)  # [..., l, m, real/imaginary]
        blocks, _, zonal_sign = _transform_tables(self.nlat, self.truncation, x.dtype, x.device)
        n_j = (self.truncation + 2) // 2
        x = F.pad(x, (0, 0, 0, 0, 0, 2 * n_j - (self.truncation + 1)))
        x = x.unflatten(-3, (n_j, 2))  # [..., j, r, m, real/imaginary], with l = 2 j + r

        # Sum the even and the odd degrees separately on the northern rings. The mirrored southern ring
        # takes the same sums, but with the sign (-1)^(l + m) = (-1)^(m + r) on each. Each block of orders
        # needs only its degrees from its first j on.
        x = torch.cat(
            [
                torch.einsum("...jrmc, mrjk -> ...kmrc", x[..., m0 // 2 :, :, m0 : m0 + table.shape[0], :], table)
                for m0, table in blocks
            ],
            dim=-3,
        )  # [..., northern latitude, m, r, real/imaginary]
        north = x[..., 0, :] + x[..., 1, :]
        south = (x[..., 0, :] - x[..., 1, :]) * zonal_sign[:, None]
        x = torch.cat((north, south.flip(-3)), dim=-3)  # [..., latitude, m, real/imaginary]

        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)

        return x
