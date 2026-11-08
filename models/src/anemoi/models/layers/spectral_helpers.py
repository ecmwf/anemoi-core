# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from anemoi.models.layers.ring_fft import FFTBackend
from anemoi.models.layers.ring_fft import shared_ring_fft
from anemoi.models.layers.ring_fft import validate_fft_options
from anemoi.models.layers.ring_fft_base import at_least_float32
from anemoi.models.layers.ring_fft_base import empty_transform

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


def _check_sht_truncation(nlat: int, truncation: int, latitude_grid: LatitudeGrid) -> None:
    """Check that the latitude rows resolve every degree up to the truncation."""
    if latitude_grid not in ("legendre-gauss", "equiangular-poles"):
        raise ValueError(f"Unknown latitude_grid={latitude_grid!r}; use 'legendre-gauss' or 'equiangular-poles'")
    max_truncation = (nlat - 1) // 2 if latitude_grid == "equiangular-poles" else nlat - 1
    if not 0 <= truncation <= max_truncation:
        raise ValueError(f"Truncation must be between 0 and {max_truncation} for {nlat} {latitude_grid} latitudes")


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
    r"""Yield (-1)^m c^l_m P^l_m(x) for each degree l.

    The recurrence uses float64 on ``device`` and keeps only the two preceding degrees, so its memory
    does not grow with the square of the truncation.

    Parameters
    ----------
    truncation : int
        Highest degree.
    x : np.ndarray
        Points at which the Legendre polynomials are evaluated, with shape (n,) and values in [-1, 1].
    device : torch.device | str, optional
        Device on which the values are computed. Default is the CPU.

    Yields
    ------
    Tensor
        For l = 0, 1, ..., truncation, a float64 tensor of shape (truncation + 1, n), indexed by the order m
        and zero for m > l. The values use the normalization of the inverse transform, with 1 / (4 pi) at
        l = 0.

    Notes
    -----
    The recurrence follows torch-harmonics.

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
        # Weights that take the diagonal value of the degree before, at order n - 1, to the two top orders of degree n.
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


# The orders m are split into blocks FIRST_BLOCK_WIDTH wide for the first BLOCKS_BEFORE_WIDENING blocks, and the
# width doubles each time m doubles from there. The blocks are narrow where most zeros can be left out and few at
# high orders. Their boundaries do not depend on the truncation, so a lower truncation uses the leading blocks of a
# higher one, and its tables look the same whether they were built for it or cut from a larger set.
FIRST_BLOCK_WIDTH = 16
BLOCKS_BEFORE_WIDENING = 8


def first_orders_of_blocks(truncation: int) -> list[int]:
    """Return the first order m of each block covering orders 0 through truncation."""
    starts, m, width = [], 0, FIRST_BLOCK_WIDTH
    widen_at = FIRST_BLOCK_WIDTH * BLOCKS_BEFORE_WIDENING
    while m <= truncation:
        if m >= widen_at:
            width, widen_at = 2 * width, 2 * widen_at
        starts.append(m)
        m += width
    return starts


def hemisphere_legendre_blocks(
    cos_colat: np.ndarray, truncation: int, dtype: torch.dtype, device: torch.device
) -> list[tuple[int, Tensor]]:
    r"""Build Legendre tables on northern latitudes, grouped by order and degree parity.

    At the corresponding southern latitude, P^l_m equals (-1)^(l + m) times its northern value, so only the
    northern latitudes are stored, together with the equator when the grid has a row there. Writing
    l = 2 j + r separates the even and odd degrees.

    Since P^l_m is zero for l < m, each block that starts at order m0 stores the degrees from l = 2 j0 onward,
    with j0 = m0 // 2, and leaves out the leading zero degrees.

    Parameters
    ----------
    cos_colat : np.ndarray
        Cosine of the colatitude of each northern row, ordered from the pole, shape (n_north,).
    truncation : int
        Highest degree.
    dtype : torch.dtype
        Precision of the returned tables. The polynomials are computed in float64, one degree at a time.
    device : torch.device
        Device on which the tables are computed and stored.

    Returns
    -------
    list[tuple[int, Tensor]]
        Pairs (m0, table), with m0 the first order of the block. Each table has shape
        (number of orders, 2, n_j - j0, n_north) and is indexed by (m - m0, r, j - j0, northern latitude),
        with n_j = (truncation + 2) // 2. The values use the normalization of the inverse transform, and
        entries with m > l or l > truncation are zero.
    """
    n_j = (truncation + 2) // 2
    first_orders = first_orders_of_blocks(truncation)
    blocks = [
        (m0, torch.zeros(m1 - m0, 2, n_j - m0 // 2, len(cos_colat), dtype=dtype, device=device))
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
    def build(
        cls, nlat: int, latitude_grid: LatitudeGrid, truncation: int, dtype: torch.dtype, device: torch.device
    ) -> "LegendreTableSet":
        LOGGER.info(
            f"Building Legendre tables for {nlat} {latitude_grid} latitudes up to T{truncation} ({dtype}, {device})"
        )
        # Keep the northern rows, and the equator row when nlat is odd.
        theta, weight = latitude_quadrature(nlat, latitude_grid)
        n_north = (nlat + 1) // 2
        # The forward direction weights each northern ring (and its southern mirror) by its quadrature
        # weight. The factor 8 pi^2 combines the 2 pi that turns the ring FFT's average over longitude
        # into an integral with the 4 pi between the forward and inverse normalization. The equator row
        # is paired with itself, so it is counted twice and its weight is halved.
        weight = 8 * np.pi**2 * weight[:n_north]
        if nlat % 2:
            weight[-1] *= 0.5
        return cls(
            truncation=truncation,
            legendre_blocks=hemisphere_legendre_blocks(np.cos(theta[:n_north]), truncation, dtype, device),
            quadrature_weight=torch.from_numpy(weight.copy()).to(device=device, dtype=dtype),
            zonal_sign=(-1.0) ** torch.arange(truncation + 1, dtype=dtype, device=device),
        )

    def up_to(self, truncation: int) -> tuple[list[tuple[int, Tensor]], Tensor, Tensor]:
        """Return the Legendre blocks, quadrature weights and signs (-1)^m for a truncation up to this set's own.

        The blocks that start at m <= truncation are selected, and their order and degree axes are cut to the
        required size. ``hemisphere_legendre_blocks`` describes the layout.
        """
        n_j = (truncation + 2) // 2
        blocks = [
            (m0, table[: truncation + 1 - m0, :, : n_j - m0 // 2])
            for m0, table in self.legendre_blocks
            if m0 <= truncation
        ]
        return blocks, self.quadrature_weight, self.zonal_sign[: truncation + 1]


class LegendreTables:
    """Process-wide cache of Legendre tables for spherical harmonic transforms.

    Each combination of nlat, latitude grid, dtype and device has one table set, built on first use. A higher
    truncation replaces the cached set, and lower truncations use slices of it. These slices have the same blocks
    as tables built for the lower truncation, as described in ``first_orders_of_blocks``, so a transform does the
    same work with either set. The tables stay cached until ``clear`` is called.

    Tables built under ``FakeTensorMode`` hold tensors without data and break later calls with real inputs.
    ``torch.compile`` with the ``eager`` and ``aot_eager`` backends builds real tables, as checked with
    PyTorch 2.9.1.
    """

    def __init__(self) -> None:
        self._tables: dict[tuple[int, LatitudeGrid, torch.dtype, torch.device], LegendreTableSet] = {}

    def get(
        self, nlat: int, latitude_grid: LatitudeGrid, truncation: int, dtype: torch.dtype, device: torch.device
    ) -> LegendreTableSet:
        """Return cached tables covering at least ``truncation``, building or replacing them as needed."""
        key = (nlat, latitude_grid, dtype, device)
        tables = self._tables.get(key)
        if tables is None or tables.truncation < truncation:
            # Tables are built as ordinary tensors even when first asked for in inference mode, since
            # training may use them later.
            with torch.inference_mode(False):
                tables = LegendreTableSet.build(nlat, latitude_grid, truncation, dtype, device)
            self._tables[key] = tables
        return tables

    def clear(self) -> None:
        """Clear the cache, so that later calls rebuild the tables."""
        self._tables.clear()

    def stored_bytes(self) -> int:
        """Return the total size of cached tensors in bytes."""
        return sum(
            sum(table.nbytes for _, table in t.legendre_blocks) + t.quadrature_weight.nbytes + t.zonal_sign.nbytes
            for t in self._tables.values()
        )


LEGENDRE_TABLES = LegendreTables()


class SphericalHarmonicTransform(Module):
    r"""Transform a field on latitude rings to spherical harmonic coefficients.

    The Legendre tables are built on first use and shared by all transforms with the same number of latitudes,
    latitude grid, precision and device, see ``LegendreTables``.

    The transform runs in float32 or float64 with autocast switched off. Half-precision fields are promoted to
    float32, and float32 and float64 fields keep their precision. Since the Legendre sums are matrix products,
    in float32 they follow ``torch.get_float32_matmul_precision``, which allows TF32 with "high".

    Attributes
    ----------
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    nlat : int
        Number of latitudes in the grid, from pole to pole.
    truncation : int
        Highest degree. The coefficients include the degrees 0 to truncation.
    n_grid_points : int
        Total number of grid points in the global grid.
    mode_weights : torch.Tensor
        Weight of each ``(l, m)`` entry of the output, shape ``(truncation + 1, truncation + 1)``:
        1 for m = 0, 2 for 0 < m <= l and 0 for m > l. The output only has m >= 0, and for a real
        field the coefficient at -m is the complex conjugate of the one at m, so m > 0 counts
        twice. Entries with m > l are always zero. The weights add up to ``(truncation + 1)**2``.

    Methods
    -------
    rfft_rings_reduced(x: Tensor) -> Tensor
        Compute the Fourier coefficients on all rings of a reduced grid together.
    rfft_rings_regular(x: Tensor) -> Tensor
        Compute real-to-complex FFTs on a regular grid.
    forward(x: Tensor) -> Tensor
        Apply the ring FFTs followed by the Legendre transform.

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        truncation: int,
        latitude_grid: LatitudeGrid = "legendre-gauss",
        *,
        fft_backend: FFTBackend = "ring",
        use_graphed_rfft: bool = False,
    ) -> None:
        r"""Initialize the forward SHT for the given grid and truncation.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Highest degree. The coefficients include the degrees 0 to truncation.
        latitude_grid : {"legendre-gauss", "equiangular-poles"}, optional
            Gaussian latitude nodes or equally spaced rows including both poles, ordered north to south.
        fft_backend : {"ring", "padded"}
            FFT algorithm on reduced grids. The default "ring" uses native FFTs grouped by ring length.
        use_graphed_rfft : bool
            Capture the native reduced-grid FFTs in CUDA graphs. Defaults to False and cannot be combined with
            "padded".
        """

        super().__init__()
        validate_fft_options(fft_backend, use_graphed_rfft)
        self.fft_backend = fft_backend
        self.use_graphed_rfft = use_graphed_rfft

        self.lons_per_lat = lons_per_lat
        self.nlat = len(self.lons_per_lat)
        self.truncation = truncation
        self.latitude_grid = latitude_grid
        _check_sht_truncation(self.nlat, truncation, latitude_grid)
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            self.rfft_rings = self.rfft_rings_reduced
        else:
            self.rfft_rings = self.rfft_rings_regular
        LOGGER.info(f"SphericalHarmonicTransform: Using {self.rfft_rings.__name__} for rfft_rings")

        zonal_wavenumber = torch.arange(self.truncation + 1)
        total_wavenumber = zonal_wavenumber.unsqueeze(-1)
        mode_weights = (zonal_wavenumber <= total_wavenumber) * torch.where(zonal_wavenumber > 0, 2.0, 1.0)
        self.register_buffer("mode_weights", mode_weights, persistent=False)

    def rfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Compute the Fourier coefficients ``[..., latitude, m]`` on each ring of a reduced grid, up to the truncation.

        The configured native or padded backend supplies the ring FFTs and their gradients.
        """
        modes = min(self.truncation + 1, max(self.lons_per_lat) // 2 + 1)
        return shared_ring_fft(tuple(self.lons_per_lat), self.fft_backend, self.use_graphed_rfft).rfft(x, modes)

    def rfft_rings_regular(self, x: Tensor) -> Tensor:
        """Compute real-to-complex FFTs on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            field [..., grid].

        Returns
        -------
        torch.Tensor
            Fourier space field [..., latitude, order m].
        """

        if x.numel() == 0:
            shape = (*x.shape[:-1], self.nlat, self.lons_per_lat[0] // 2 + 1)
            return empty_transform(x, shape, torch.promote_types(x.dtype, torch.complex64))
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
            spectral representation of field [..., degree l, order m], in complex64 or complex128.
        """
        x = at_least_float32(x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self._transform(x)

    def _transform(self, x: Tensor) -> Tensor:
        x = torch.view_as_real(self.rfft_rings(x)[..., : self.truncation + 1])  # [..., latitude, m, real/imaginary]
        tables = LEGENDRE_TABLES.get(self.nlat, self.latitude_grid, self.truncation, x.dtype, x.device)
        blocks, quadrature_weight, zonal_sign = tables.up_to(self.truncation)

        # Pair each northern ring with its mirror in the south. Degrees with l + m even take the sum of
        # the two rings and degrees with l + m odd the difference: with l = 2 j + r, that is
        # north + (-1)^(m + r) * south. An equator row is paired with itself.
        n_north = (self.nlat + 1) // 2
        north = x[..., :n_north, :, :]
        south = x[..., self.nlat - n_north :, :, :].flip(-3) * zonal_sign[:, None]
        x = torch.stack((north + south, north - south), dim=-2)  # [..., northern latitude, m, r, real/imaginary]
        x = x * quadrature_weight[:, None, None, None]

        # Each block of orders gives its degrees from its first j on, since the degrees below are zero.
        # Keeping (m, r) in the table's order lets the contraction use the table in place, without a copy.
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
    r"""Transform spherical harmonic coefficients to a field on latitude rings.

    The Legendre tables are built on first use and shared by all transforms with the same number of latitudes,
    latitude grid, precision and device, see ``LegendreTables``.

    The transform runs in complex64 or complex128 with autocast switched off. Half-precision coefficients are
    promoted to complex64, and complex64 and complex128 coefficients keep their precision. In single precision,
    the Legendre sums follow ``torch.get_float32_matmul_precision``.

    Attributes
    ----------
    truncation : int
        Highest degree. The coefficients include the degrees 0 to truncation.
    nlat : int
        Number of latitudes in the grid, from pole to pole.
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    n_grid_points : int
        Total number of grid points in the global grid.

    Methods
    -------
    irfft_rings_reduced(x: Tensor) -> Tensor
        Compute the values on all rings of a reduced grid together from their Fourier coefficients.
    irfft_rings_regular(x: Tensor) -> Tensor
        Compute complex-to-real FFTs on a regular grid.
    forward(x: Tensor) -> Tensor
        Apply the inverse Legendre transform followed by the inverse ring FFTs.

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        truncation: int,
        latitude_grid: LatitudeGrid = "legendre-gauss",
        *,
        fft_backend: FFTBackend = "ring",
        use_graphed_irfft: bool = False,
    ) -> None:
        r"""Initialize the inverse SHT for the given grid and truncation.

        Parameters
        ----------
        lons_per_lat : list[int]
            Number of longitudinal points on each latitude ring, from pole to pole.
        truncation : int
            Highest degree. The coefficients include the degrees 0 to truncation.
        latitude_grid : {"legendre-gauss", "equiangular-poles"}, optional
            Gaussian latitude nodes or equally spaced rows including both poles, ordered north to south.
        fft_backend : {"ring", "padded"}
            FFT algorithm on reduced grids. The default "ring" uses native FFTs grouped by ring length.
        use_graphed_irfft : bool
            Capture the native reduced-grid FFTs in CUDA graphs. Defaults to False and cannot be combined with
            "padded".
        """

        super().__init__()
        validate_fft_options(fft_backend, use_graphed_irfft)
        self.fft_backend = fft_backend
        self.use_graphed_irfft = use_graphed_irfft

        nlat = len(lons_per_lat)
        _check_sht_truncation(nlat, truncation, latitude_grid)

        self.truncation = truncation
        self.latitude_grid = latitude_grid
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Use more efficient batched rfft for regular grids
        if len(set(self.lons_per_lat)) > 1:
            self.irfft_rings = self.irfft_rings_reduced
        else:
            self.irfft_rings = self.irfft_rings_regular
        LOGGER.info(f"InverseSphericalHarmonicTransform: Using {self.irfft_rings.__name__} for irfft_rings")

    def irfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Compute the values ``[..., grid]`` on the rings of a reduced grid from coefficients ``[..., latitude, m]``.

        The configured native or padded backend supplies the ring FFTs and their gradients.
        """
        return shared_ring_fft(tuple(self.lons_per_lat), self.fft_backend, self.use_graphed_irfft).irfft(x)

    def irfft_rings_regular(self, x: Tensor) -> Tensor:
        """Compute complex-to-real FFTs on each latitude ring of a regular grid.

        Parameters
        ----------
        x : torch.Tensor
            Fourier space field [..., latitude, order m].

        Returns
        -------
        torch.Tensor
            field [..., grid].
        """

        if x.numel() == 0:
            return empty_transform(x, (*x.shape[:-2], self.n_grid_points), x.real.dtype)
        return torch.fft.irfft(x, self.lons_per_lat[0], norm="forward").reshape(*x.shape[:-2], self.n_grid_points)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the inverse Legendre transform followed by the inverse ring FFTs.

        Parameters
        ----------
        x : torch.Tensor
            spectral representation of field [..., degree l, order m]. Both sizes may be smaller than
            truncation + 1, and the missing higher degrees and orders are then taken as zero.

        Returns
        -------
        torch.Tensor
            field [..., grid], in float32 or float64.
        """

        n_l, n_m = x.shape[-2], x.shape[-1]
        size = self.truncation + 1
        if not (1 <= n_l <= size and 1 <= n_m <= size):
            raise ValueError(
                f"Spectral shape (l={n_l}, m={n_m}) must lie between 1 and truncation + 1 = {size} in both dimensions."
            )
        x = at_least_float32(x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self._transform(x)

    def _transform(self, x: Tensor) -> Tensor:
        n_l, n_m = x.shape[-2], x.shape[-1]
        # A smaller input uses only the part of the tables that it reaches, so it costs less. The tables
        # are still built for the full truncation, so that inputs of growing size do not rebuild them.
        truncation = max(n_l, n_m) - 1
        tables = LEGENDRE_TABLES.get(self.nlat, self.latitude_grid, self.truncation, x.real.dtype, x.device)
        blocks, _, zonal_sign = tables.up_to(truncation)
        n_j = (truncation + 2) // 2

        x = torch.view_as_real(x)  # [..., l, m, real/imaginary]
        x = F.pad(x, (0, 0, 0, truncation + 1 - n_m, 0, 2 * n_j - n_l))
        x = x.unflatten(-3, (n_j, 2))  # [..., j, r, m, real/imaginary], with l = 2 j + r

        # Sum the even and the odd degrees separately on the northern rings. The mirrored southern ring
        # takes the same sums, but with the sign (-1)^(l + m) = (-1)^(m + r) on each. Each block of orders
        # needs only its degrees from its first j on. An equator row is its own mirror and appears once.
        x = torch.cat(
            [
                torch.einsum("...jrmc, mrjk -> ...kmrc", x[..., m0 // 2 :, :, m0 : m0 + table.shape[0], :], table)
                for m0, table in blocks
            ],
            dim=-3,
        )  # [..., northern latitude, m, r, real/imaginary]
        north = x[..., 0, :] + x[..., 1, :]
        south = (x[..., 0, :] - x[..., 1, :]) * zonal_sign[:, None]
        south = south[..., : self.nlat // 2, :, :]
        x = torch.cat((north, south.flip(-3)), dim=-3)  # [..., latitude, m, real/imaginary]

        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)

        return x
