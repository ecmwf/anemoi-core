# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Local amount and direction of structure at wavelet scales on the sphere.

Scales
------
The total wavenumbers l are split into dyadic wavelet scales with the kernels of S2LET (Wiaux et al. 2008,
Leistedt et al. 2013). A smooth step k(t), 1 for t < 1/2 and 0 for t > 1, gives

    kappa_j(l) = sqrt(k(l / 2^(j + 1)) - k(l / 2^j))

- Scale j = 0, 1, ... covers 2^(j - 1) < l < 2^(j + 1) and peaks at l = 2^j.
- The scaling function sqrt(k(l)) holds only l = 0. With it, the squared kernels add up to 1 at every degree.
- The truncation cuts the last scale or two short.

Riesz pair
----------
The band of scale j is the field with each coefficient multiplied by kappa_j(l). Its Riesz pair (Felsberg &
Sommer 2001) is the east and north gradient of the band, with each degree divided by sqrt(l (l + 1)).

- It has the same energy as the band.
- At each point it points across the band's crests, whatever their phase.
- The gradient comes from the Legendre recurrence. It needs one degree more than the band, but no table of
  derivatives.

Structure tensor
----------------
The tensor r r^T of the Riesz pair r, averaged over a few wavelengths, says how much structure there is around
each point (its trace E), and how strongly and along which direction it lines up.

- The average is a smoothing with the heat kernel exp(-l (l + 1) / (2 Ls^2)), with Ls = 2^(j - smoothing_octaves).
  It is cut at 5 Ls, where it is below 4e-6.
- On the sphere the cut kernel dips to about -3e-7 of its peak, so the tensor is positive semi-definite up to
  that.
- Scales j below smoothing_octaves have Ls < 1: their tensor is close to the global mean.
- The tensor is smoothed in three dimensions, which have no special direction at the poles, and then read off in
  the local east and north directions: J_ee, J_nn and J_en.
- The neighbours' tangent planes are tilted against the point's own. About 1 / Ls^2 of the energy ends up in the
  radial direction, which the read-off leaves out (below 1e-3 for Ls >= 32).
"""

import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import functional as F

from anemoi.models.layers.spectral_helpers import LEGENDRE_TABLES
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import latitude_quadrature

# The heat kernel is cut at this multiple of the smoothing degree Ls, where it has fallen below 4e-6.
_HEAT_KERNEL_REACH = 5.0


def _s2let_step_and_rest(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The S2LET step k(t) and 1 - k(t), each worked out directly.

    The small values near the ends of a scale are then not lost as the difference of two numbers close to 1.

    k(t) is 1 for t <= 1/2 and 0 for t >= 1. In between it is the integral of b(x) / x from t to 1, divided by the
    same integral from 1/2. The bump b(x) = exp(-2 / (1 - y^2)), with y = 4 (x - 1/2) - 1, runs over [1/2, 1].
    """
    grid = np.linspace(0.5, 1.0, 4096)
    y = 4.0 * (grid - 0.5) - 1.0
    with np.errstate(divide="ignore"):  # 1 - y^2 is 0 at both ends, where the bump is 0
        integrand = np.exp(-2.0 / (1.0 - y * y)) / grid
    # Integrals from 1/2 to each grid point and from each grid point to 1, by the trapezoid rule.
    segments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid)
    total = segments.sum()
    from_start = np.concatenate([[0.0], np.cumsum(segments)]) / total
    to_end = np.concatenate([np.cumsum(segments[::-1])[::-1], [0.0]]) / total
    return np.interp(t, grid, to_end, left=1.0, right=0.0), np.interp(t, grid, from_start, left=0.0, right=1.0)


def s2let_step(t: np.ndarray) -> np.ndarray:
    """Smooth step k(t) of S2LET for dyadic scales: 1 for t <= 1/2, 0 for t >= 1."""
    return _s2let_step_and_rest(t)[0]


def scale_degree_ranges(truncation: int) -> list[tuple[int, int]]:
    """Lowest and highest degree of each dyadic scale up to the truncation, from scale 0 up.

    Scale j covers 2^(j - 1) < l < 2^(j + 1). There is one scale for each octave that starts at or below the
    truncation.
    """
    if truncation < 1:
        raise ValueError(f"truncation must be at least 1, got {truncation}")
    n_scales = math.ceil(math.log2(truncation)) + 1
    return [(2**j // 2 + 1, min(truncation, 2 ** (j + 1) - 1)) for j in range(n_scales)]


def wavelet_kernels(truncation: int) -> np.ndarray:
    """Dyadic S2LET kernels kappa_j(l), ``[n_scales, truncation + 1]``, one row per scale of ``scale_degree_ranges``.

    - A kernel is 0 outside its scale.
    - Inside, a value is 0 only where the true value is below what float64 holds (about 1e-300). A whole row is
      then 0 only when the finest scale holds just its lowest degree: at T2049, T4097 and so on.
    """
    ell = np.arange(truncation + 1, dtype=np.float64)
    rows = []
    for j in range(len(scale_degree_ranges(truncation))):
        lower, upper = ell / 2.0 ** (j + 1), ell / 2.0**j
        # kappa_j^2 = k(lower) - k(upper). At each degree one of the two steps is exactly 1 or 0, so take the
        # other one alone: a difference of two numbers close to 1 would round the small values to 0.
        k_lower, _ = _s2let_step_and_rest(lower)
        _, rest_upper = _s2let_step_and_rest(upper)
        rows.append(np.sqrt(np.where(lower <= 0.5, rest_upper, k_lower)))
    return np.stack(rows)


class SphericalGradient(Module):
    """East and north gradient, on the unit sphere, of a field given by its spectral coefficients.

    - East: each coefficient times i m, then divided by sin(colatitude) on the grid.
    - North: the Legendre recurrence writes sin(colatitude) times the latitude derivative as a mix of the degrees
      one up and one down. So the inverse transform needs one degree beyond the truncation, but no table of
      derivatives.

    ``inverse`` is that inverse transform. It accepts any smaller input, so other fields on the same grid can use it.

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitudes on each Gaussian latitude ring, from north to south. Gaussian latitudes avoid the
        poles, where east and north are not defined.
    truncation : int
        Highest total wavenumber of the input coefficients.
    """

    def __init__(self, lons_per_lat: list[int], truncation: int) -> None:
        super().__init__()
        nlat = len(lons_per_lat)
        if truncation > nlat - 2:
            raise ValueError(
                f"The gradient needs one degree beyond the truncation, so on {nlat} Gaussian latitudes the "
                f"truncation can be at most {nlat - 2}, got {truncation}."
            )
        self.truncation = truncation
        self.inverse = InverseSphericalHarmonicTransform(lons_per_lat, truncation + 1)

        colatitude, _ = latitude_quadrature(nlat, "legendre-gauss")
        self.register_buffer(
            "sin_colat", torch.from_numpy(np.repeat(np.sin(colatitude), lons_per_lat)), persistent=False
        )

        size = truncation + 2
        ell = np.arange(size, dtype=np.float64)[:, None]
        m = np.arange(size, dtype=np.float64)[None, :]
        eps = np.sqrt(np.clip(ell**2 - m**2, 0.0, None) / (4.0 * ell**2 - 1.0))
        # For a field with coefficients a, sin(colatitude) times its derivative towards the south has the
        # coefficient (l - 1) eps_l a_(l-1) - (l + 2) eps_(l+1) a_(l+1) at degree l. The weights do not depend on
        # the truncation, so a smaller input uses the top-left corner of the tables.
        eps_above = np.concatenate([eps[1:], np.zeros((1, size))], axis=0)
        self.register_buffer("from_below", torch.from_numpy((ell - 1) * eps), persistent=False)
        self.register_buffer("from_above", torch.from_numpy(-(ell + 2) * eps_above), persistent=False)
        self.register_buffer("zonal_wavenumber", torch.arange(truncation + 1, dtype=torch.float64), persistent=False)

    def forward(self, coeffs: Tensor) -> Tensor:
        """East and north gradient ``[..., 2, grid]`` of the field with coefficients ``[..., n, n]``.

        n can be at most truncation + 1.
        """
        n = coeffs.shape[-1]
        if coeffs.shape[-2] != n or n > self.truncation + 1:
            raise ValueError(
                f"Expected square coefficients of size at most {self.truncation + 1}, got {tuple(coeffs.shape[-2:])}."
            )
        real_dtype = coeffs.real.dtype
        # Coefficients of sin(colatitude) times the east and north derivatives. North reaches one degree further.
        east = coeffs * (1j * self.zonal_wavenumber[:n].to(real_dtype))
        padded = F.pad(coeffs, (0, 0, 1, 2))  # one zero row above, two below
        below = padded[..., 0 : n + 1, :]
        above = padded[..., 2 : n + 3, :]
        north = -(
            self.from_below[: n + 1, :n].to(real_dtype) * below + self.from_above[: n + 1, :n].to(real_dtype) * above
        )
        components = torch.stack([F.pad(east, (0, 0, 0, 1)), north], dim=-3)
        return self.inverse(components) / self.sin_colat.to(real_dtype)


def _check_room_for_products(lons_per_lat: list[int], tops: list[int], smoothing_tops: list[int]) -> None:
    """Check that the grid holds the products of each Riesz pair up to its smoothing degrees.

    - The Riesz pair of a scale with top degree L reaches degree L + 1, so its products reach 2 (L + 1).
    - Their transform up to the smoothing top S is exact on Gaussian latitudes while 2 (L + 1) + S <= 2 nlat - 1.
    - On a regular grid the rings must also have more than 2 (L + 1) + S points, or high orders of the products
      fold onto the orders that are kept.
    - Reduced rings near the poles are shorter and fold some orders onto degrees near S. The heat kernel is below
      4e-6 there, so the smoothed tensor is barely affected.
    """
    needed = max(2 * (top + 1) + smoothing_top for top, smoothing_top in zip(tops, smoothing_tops))
    nlat = len(lons_per_lat)
    if needed > 2 * nlat - 1:
        raise ValueError(
            f"The products of the Riesz pair at T{max(tops)} need at least {needed // 2 + 1} latitudes, got {nlat}. "
            "Use a lower truncation."
        )
    if len(set(lons_per_lat)) == 1 and lons_per_lat[0] <= needed:
        raise ValueError(
            f"The products of the Riesz pair at T{max(tops)} need more than {needed} longitudes, got {lons_per_lat[0]}."
        )


class RieszStructureTensor(Module):
    """Smoothed structure tensor of the Riesz pair of a field, at chosen wavelet scales.

    See the module description for the scales, the Riesz pair and the smoothing.

    - The result is J_ee, J_nn and J_en at every grid point, for each chosen scale.
    - A field goes through the forward transform once. Each scale then uses only the degrees it reaches.
    - All transforms share one set of Legendre tables.

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitudes on each Gaussian latitude ring, from north to south. The points on each ring are
        equally spaced, starting at longitude 0.
    truncation : int
        Highest total wavenumber. Let L be the top degree of the finest chosen scale, at most 2^(j + 1) - 1, and Ls
        its smoothing degree. The products of its Riesz pair reach degree 2 (L + 1) and its heat kernel 5 Ls. The
        grid must hold both: 2 (L + 1) + 5 Ls <= 2 nlat - 1, and on a regular grid more longitudes than that.
    scales : list[int]
        Wavelet scales j to return, each covering 2^(j - 1) < l < 2^(j + 1), in the order given.
    smoothing_octaves : int
        Octaves between a scale and its smoothing degree Ls = 2^(j - smoothing_octaves). The smoothing is then
        2^smoothing_octaves / (2 pi) wavelengths of the scale's peak wide, about 1.3 for the default 3. The heat
        kernel must fit below the highest degree of the transforms: with fewer octaves the finest scales are
        rejected.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        truncation: int,
        scales: list[int],
        smoothing_octaves: int = 3,
    ) -> None:
        super().__init__()
        ranges = scale_degree_ranges(truncation)
        if not scales or any(not 0 <= j < len(ranges) for j in scales):
            raise ValueError(
                f"scales must be a non-empty list of scales in [0, {len(ranges) - 1}] at T{truncation}, got {scales}."
            )
        if smoothing_octaves < 0:
            raise ValueError(f"smoothing_octaves must be at least 0, got {smoothing_octaves}.")
        kernels = wavelet_kernels(truncation)
        for j in scales:
            if not kernels[j].any():
                raise ValueError(f"Scale {j} has no degree up to T{truncation} that float64 can hold; leave it out.")
        self.scales = list(scales)
        self.tops = [ranges[j][1] for j in self.scales]

        # The forward transform only needs the degrees the finest chosen scale reaches.
        self.sht = SphericalHarmonicTransform(lons_per_lat, max(self.tops))
        self.gradient = SphericalGradient(lons_per_lat, max(self.tops))
        ell = np.arange(max(self.tops) + 1, dtype=np.float64)
        riesz_scale = np.zeros_like(ell)
        riesz_scale[1:] = 1.0 / np.sqrt(ell[1:] * (ell[1:] + 1.0))
        # Each scale's kernel with the division by sqrt(l (l + 1)), one row per chosen scale.
        windows = kernels[self.scales, : max(self.tops) + 1] * riesz_scale
        self.register_buffer("riesz_windows", torch.from_numpy(windows), persistent=False)

        # The smoothed tensor goes back through the gradient's inverse transform, so the heat kernel must have
        # fallen off before the highest degree of that transform.
        smoothing_degrees = [2.0 ** (j - smoothing_octaves) for j in self.scales]
        smoothing_tops = [math.ceil(_HEAT_KERNEL_REACH * degree) for degree in smoothing_degrees]
        reach = self.gradient.inverse.truncation
        for j, top in zip(self.scales, smoothing_tops):
            if top > reach:
                raise ValueError(
                    f"Scale {j} needs smoothing degrees up to {top}, but the transforms reach only {reach}. "
                    "Use more smoothing_octaves or leave the scale out."
                )
        _check_room_for_products(lons_per_lat, self.tops, smoothing_tops)
        self.smoothing_shts = ModuleList(SphericalHarmonicTransform(lons_per_lat, top) for top in smoothing_tops)
        degrees = np.arange(max(smoothing_tops) + 1, dtype=np.float64)
        heat_kernels = np.stack([np.exp(-degrees * (degrees + 1.0) / (2.0 * ls**2)) for ls in smoothing_degrees])
        self.register_buffer("heat_kernels", torch.from_numpy(heat_kernels), persistent=False)

        colatitude, quadrature = latitude_quadrature(len(lons_per_lat), "legendre-gauss")
        latitude = np.repeat(0.5 * np.pi - colatitude, lons_per_lat)
        longitude = np.concatenate([2.0 * np.pi * np.arange(n) / n for n in lons_per_lat])
        area = np.repeat(quadrature / np.asarray(lons_per_lat, dtype=np.float64), lons_per_lat)
        self.register_buffer("area", torch.from_numpy(area / area.sum()), persistent=False)
        east = np.stack([-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)])
        north = np.stack(
            [-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)]
        )
        self.register_buffer("east", torch.from_numpy(east), persistent=False)
        self.register_buffer("north", torch.from_numpy(north), persistent=False)

    @staticmethod
    def _read(entries: Tensor, u: Tensor, w: Tensor) -> Tensor:
        """u^T S w ``[..., grid]`` of symmetric tensors S given as ``[..., 6, grid]``: xx, yy, zz, xy, xz, yz."""
        diagonal = (entries[..., :3, :] * u * w).sum(-2)
        off_diagonal = (
            entries[..., 3, :] * (u[0] * w[1] + u[1] * w[0])
            + entries[..., 4, :] * (u[0] * w[2] + u[2] * w[0])
            + entries[..., 5, :] * (u[1] * w[2] + u[2] * w[1])
        )
        return diagonal + off_diagonal

    def coefficients(self, x: Tensor) -> Tensor:
        """Spectral coefficients ``[..., l, m]`` of fields ``[..., grid]`` up to the finest chosen scale's top degree."""
        # Build the Legendre tables once, at the largest size used here (the gradient's). The smaller transforms
        # then use parts of them.
        inverse = self.gradient.inverse
        LEGENDRE_TABLES.get(inverse.nlat, inverse.latitude_grid, inverse.truncation, x.dtype, x.device)
        return self.sht(x)

    def riesz_pair(self, coeffs: Tensor, k: int) -> Tensor:
        """East and north Riesz parts ``[..., 2, grid]`` of the k-th chosen scale, from coefficients ``[..., l, m]``."""
        top = self.tops[k]
        window = self.riesz_windows[k, : top + 1].to(coeffs.real.dtype)
        return self.gradient(coeffs[..., : top + 1, : top + 1] * window[:, None])

    @staticmethod
    def products(pair: Tensor, east: Tensor, north: Tensor) -> Tensor:
        """The six products ``[..., 6, grid]`` of a Riesz pair ``[..., 2, grid]``, formed in 3D: xx, yy, zz, xy, xz, yz."""
        x, y, z = (pair[..., :1, :] * east + pair[..., 1:, :] * north).unbind(-2)
        return torch.stack([x * x, y * y, z * z, x * y, x * z, y * z], dim=-2)

    def smooth(self, products: Tensor, k: int) -> Tensor:
        """Products ``[..., 6, grid]`` smoothed with the heat kernel of the k-th chosen scale."""
        smoothed = self.smoothing_shts[k](products)
        return self.gradient.inverse(smoothed * self.heat_kernels[k, : smoothed.shape[-2], None].to(products.dtype))

    @classmethod
    def read_tensor(cls, smoothed: Tensor, east: Tensor, north: Tensor) -> Tensor:
        """J_ee, J_nn and J_en ``[..., 3, grid]`` from smoothed products ``[..., 6, grid]``."""
        return torch.stack(
            [cls._read(smoothed, east, east), cls._read(smoothed, north, north), cls._read(smoothed, east, north)],
            dim=-2,
        )

    def tensor(self, coeffs: Tensor, k: int) -> Tensor:
        """J_ee, J_nn and J_en ``[..., 3, grid]`` of the k-th chosen scale, from coefficients ``[..., l, m]``."""
        dtype = coeffs.real.dtype
        east, north = self.east.to(dtype), self.north.to(dtype)
        products = self.products(self.riesz_pair(coeffs, k), east, north)
        return self.read_tensor(self.smooth(products, k), east, north)

    def forward(self, x: Tensor) -> Tensor:
        """J_ee, J_nn and J_en ``[..., scale, 3, grid]`` of fields ``[..., grid]``, for the chosen scales in order."""
        coeffs = self.coefficients(x)
        return torch.stack([self.tensor(coeffs, k) for k in range(len(self.scales))], dim=-3)
