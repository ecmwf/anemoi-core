# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.fft import irfft
from torch.fft import rfft
from torch.nn import Module


def legendre_gauss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b].
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def clenshaw_curtiss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""Computation of the Clenshaw-Curtis quadrature nodes and weights.

    This implementation follows
    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001-018.
    """

    assert n > 1

    tcc = np.cos(np.linspace(np.pi, 0, n))

    if n == 2:
        wcc = np.array([1.0, 1.0])
    else:

        n1 = n - 1
        N = np.arange(1, n1, 2)
        num_odd = len(N)
        m = n1 - num_odd

        v = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
        v = 0 - v[:-1] - v[-1:0:-1]

        g0 = (-1) * np.ones(n1)
        g0[num_odd] = g0[num_odd] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1**2 - 1 + (n1 % 2))
        wcc = np.fft.ifft(v + g).real
        wcc = np.concatenate((wcc, wcc[:1]))

    # Rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc


def legpoly(
    mmax: int,
    lmax: int,
    x: np.ndarray,
    norm: str = "ortho",
    inverse: bool = False,
    csphase: bool = True,
) -> np.ndarray:
    r"""Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    # Compute the tensor P^m_n:
    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)

    norm_factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # Fill the diagonal and the lower diagonal
    for n in range(1, nmax):
        vdm[n - 1, n, :] = np.sqrt(2 * n + 1) * x * vdm[n - 1, n - 1, :]
        vdm[n, n, :] = np.sqrt((2 * n + 1) * (1 + x) * (1 - x) / 2 / n) * vdm[n - 1, n - 1, :]

    # Fill the remaining values on the upper triangle and multiply b
    for n in range(2, nmax):
        for m in range(0, n - 1):
            vdm[m, n, :] = (
                x * np.sqrt((2 * n - 1) / (n - m) * (2 * n + 1) / (n + m)) * vdm[m, n - 1, :]
                - np.sqrt((n + m - 1) / (n - m) * (2 * n + 1) / (2 * n - 3) * (n - m - 1) / (n + m)) * vdm[m, n - 2, :]
            )

    if norm == "schmidt":
        for num in range(0, nmax):
            if inverse:
                vdm[:, num, :] = vdm[:, num, :] * np.sqrt(2 * num + 1)
            else:
                vdm[:, num, :] = vdm[:, num, :] / np.sqrt(2 * num + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= -1

    return vdm


def precompute_legpoly(
    mmax: int,
    lmax: int,
    t: np.ndarray,
    norm: str = "ortho",
    inverse: bool = False,
    csphase: bool = True,
) -> np.ndarray:
    r"""Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)


class CartesianRealSHT(Module):

    def __init__(self, nlat: int, nlon: int, grid: str) -> None:

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon

        self.lmax = self.nlat
        self.mmax = min(self.lmax, self.nlon // 2 + 1)

        if grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
        elif grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise NotImplementedError(f"Unknown grid {grid}.")

        tq = np.flip(np.arccos(cost))

        pct = precompute_legpoly(self.mmax, self.lmax, tq)
        pct = torch.from_numpy(pct)

        weights = torch.from_numpy(w)
        weights = torch.einsum("mlk, k -> mlk", pct, weights)

        self.register_buffer("weights", weights, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

        x = torch.view_as_real(x)

        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        xout[..., 0] = torch.einsum("...km, mlk -> ...lm", x[..., : self.mmax, 0], self.weights.to(x.dtype))
        xout[..., 1] = torch.einsum("...km, mlk -> ...lm", x[..., : self.mmax, 1], self.weights.to(x.dtype))
        x = torch.view_as_complex(xout)

        return x


class CartesianInverseRealSHT(Module):

    def __init__(self, nlat: int, nlon: int, lmax: int, grid: str) -> None:

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon

        self.lmax = self.mmax = lmax

        if grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
        elif grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise NotImplementedError(f"Unknown grid {grid}.")

        t = np.flip(np.arccos(cost))

        pct = precompute_legpoly(self.mmax, self.lmax, t, inverse=True)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.view_as_real(x)
        x = torch.einsum("...lmr, mlk -> ...kmr", x, self.pct.to(x.dtype)).to(x.dtype)
        x = torch.view_as_complex(x.contiguous())

        x[..., 0].imag = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < self.mmax):
            x[..., self.nlon // 2].imag = 0.0

        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x


class OctahedralRealSHT(Module):

    def __init__(
        self,
        nlat: int,
        lmax: int | None = None,
        mmax: int | None = None,
        folding: bool = False,
    ) -> None:

        super().__init__()

        self.lmax = lmax or nlat
        self.mmax = mmax or nlat

        self.nlat = nlat
        self.nlon = [4 * (i + 1) + 16 for i in range(nlat // 2)]
        self.nlon = self.nlon + self.nlon[::-1]

        self.slon = [0] + list(np.cumsum(self.nlon))[:-1]
        self.rlon = [nlat + 8 - nlon // 2 for nlon in self.nlon]

        self.folding = self.spectral_folding if folding else self.no_spectral_folding

        theta, weight = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        pct = precompute_legpoly(self.mmax, self.lmax, theta)
        pct = torch.from_numpy(pct)

        weight = torch.from_numpy(weight)
        weight = torch.einsum("mlk, k -> mlk", pct, weight)

        self.register_buffer("weight", weight, persistent=False)

    def spectral_folding(self, x: Tensor) -> Tensor:

        raise NotImplementedError

    def no_spectral_folding(self, x: Tensor) -> Tensor:

        return x

    def rfft(self, x: Tensor) -> Tensor:

        return torch.fft.rfft(
            input=self.folding(x),
            norm="forward",
        )

    def rfft_rings(self, x: Tensor) -> Tensor:

        rfft = [self.rfft(x[..., slon : slon + nlon]) for slon, nlon in zip(self.slon, self.nlon)]

        rfft = [
            torch.cat([x, torch.zeros((*x.shape[:-1], rlon), device=x.device)], dim=-1)
            for x, rlon in zip(rfft, self.rlon)
        ]

        return torch.stack(
            tensors=rfft,
            dim=-2,
        )

    def forward(self, x: Tensor) -> Tensor:

        x = 2.0 * torch.pi * self.rfft_rings(x)
        x = torch.view_as_real(x)

        rl = torch.einsum("...km, mlk -> ...lm", x[..., : self.mmax, 0], self.weight.to(x.dtype))
        im = torch.einsum("...km, mlk -> ...lm", x[..., : self.mmax, 1], self.weight.to(x.dtype))

        x = torch.stack((rl, im), -1)
        x = torch.view_as_complex(x)

        return x


class OctahedralInverseRealSHT(Module):

    def __init__(
        self,
        nlat: int,
        lmax: int | None = None,
        mmax: int | None = None,
        folding: bool = False,
    ) -> None:

        super().__init__()

        self.lmax = lmax or nlat
        self.mmax = mmax or nlat

        self.nlat = nlat
        self.nlon = [4 * (i + 1) + 16 for i in range(nlat // 2)]
        self.nlon = self.nlon + self.nlon[::-1]

        self.folding = self.spectral_folding if folding else self.no_spectral_folding

        theta, _ = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        pct = precompute_legpoly(self.mmax, self.lmax, theta, inverse=True)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def spectral_folding(self, x: Tensor, nlon: int) -> Tensor:

        raise NotImplementedError

    def no_spectral_folding(self, x: Tensor, nlon: int) -> Tensor:

        return x

    def irfft(self, x: Tensor, nlon: int) -> Tensor:

        return torch.fft.irfft(
            input=self.folding(x, nlon),
            n=nlon,
            norm="forward",
        )

    def irfft_rings(self, x: Tensor) -> Tensor:

        irfft = [self.irfft(x[..., t, :], nlon) for t, nlon in enumerate(self.nlon)]

        return torch.cat(
            tensors=irfft,
            dim=-1,
        )

    def forward(self, x: Tensor) -> Tensor:

        x = torch.view_as_real(x)

        rl = torch.einsum("...lm, mlk -> ...km", x[..., 0], self.pct.to(x.dtype))
        im = torch.einsum("...lm, mlk -> ...km", x[..., 1], self.pct.to(x.dtype))

        x = torch.stack((rl, im), -1).to(x.dtype)
        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)

        return x


def healpix_latitude_theta(nside: int) -> np.ndarray:

    t = np.arange(0, 4 * nside - 1).astype(np.float64)

    z = np.zeros_like(t)
    z[t < nside - 1] = 1 - (t[t < nside - 1] + 1) ** 2 / (3 * nside**2)

    z[(t >= nside - 1) & (t <= 3 * nside - 1)] = 4 / 3 - 2 * (t[(t >= nside - 1) & (t <= 3 * nside - 1)] + 1) / (
        3 * nside
    )

    z[(t > 3 * nside - 1) & (t <= 4 * nside - 2)] = (
        4 * nside - 1 - t[(t > 3 * nside - 1) & (t <= 4 * nside - 2)]
    ) ** 2 / (3 * nside**2) - 1

    return np.arccos(z)


def healpix_rings_offset(t: np.ndarray, nside: int) -> np.ndarray:

    shift = 1 / 2
    tt = np.zeros_like(t)

    tt = np.where(
        (t + 1 >= nside) & (t + 1 <= 3 * nside),
        shift * ((t - nside + 2) % 2) * np.pi / (2 * nside),
        tt,
    )

    tt = np.where(t + 1 > 3 * nside, shift * np.pi / (2 * (4 * nside - t - 1)), tt)
    tt = np.where(t + 1 < nside, shift * np.pi / (2 * (t + 1)), tt)

    return tt


def healpix_nlon_ring(t: int, nside: int) -> int:

    if (t >= 0) and (t < nside - 1):
        return 4 * (t + 1)

    elif (t >= nside - 1) and (t <= 3 * nside - 1):
        return 4 * nside

    elif (t > 3 * nside - 1) and (t <= 4 * nside - 2):
        return 4 * (4 * nside - t - 1)

    else:
        raise ValueError(f"Ring t={t} not contained by nside={nside}.")


def healpix_longitude_offset(mmax: int, nside: int) -> np.ndarray:

    t = np.arange(0, 4 * nside - 1)

    offsets = healpix_rings_offset(t, nside)
    offsets = np.einsum("t, m -> tm", offsets, np.arange(0, mmax))

    return offsets


class HEALPixInverseRealSHT(Module):

    def __init__(
        self,
        nside: int,
        lmax: int | None = None,
        mmax: int | None = None,
        folding: bool = False,
    ) -> None:

        super().__init__()

        self.lmax = lmax or 4 * nside - 1
        self.mmax = mmax or 4 * nside - 1
        self.nside = nside

        self.nlat = 4 * nside - 1
        self.nlon = [healpix_nlon_ring(t, nside) for t in range(self.nlat)]
        self.folding = self.spectral_folding if folding else self.no_spectral_folding

        theta = healpix_latitude_theta(nside)
        offset = healpix_longitude_offset(self.mmax, nside)

        pct = precompute_legpoly(self.mmax, self.lmax, theta, inverse=True)

        pct = torch.from_numpy(pct)
        offset = torch.from_numpy(offset)

        self.register_buffer("pct", pct, persistent=False)
        self.register_buffer("offset", offset, persistent=False)

    def spectral_folding(self, x: Tensor, nlon: int) -> Tensor:

        raise NotImplementedError

    def no_spectral_folding(self, x: Tensor, nlon: int) -> Tensor:

        return x

    def irfft(self, x: Tensor, nlon: int) -> Tensor:

        return torch.fft.irfft(
            input=self.folding(x, nlon),
            n=nlon,
            norm="forward",
        )

    def irfft_rings(self, x: Tensor) -> Tensor:

        irfft = [self.irfft(x[..., t, :], nlon) for t, nlon in enumerate(self.nlon)]

        return torch.cat(
            tensors=irfft,
            dim=-1,
        )

    def forward(self, x: Tensor) -> Tensor:

        x = torch.view_as_real(x)

        rl = torch.einsum("...lm, mlk -> ...km", x[..., 0], self.pct.to(x.dtype))
        im = torch.einsum("...lm, mlk -> ...km", x[..., 1], self.pct.to(x.dtype))
        xs = torch.stack((rl, im), -1)

        x = torch.view_as_complex(xs)
        f = torch.exp(1j * self.offset.to(x.dtype))

        return self.irfft_rings(x * f)


polytype = np.float64

LOGGER = logging.getLogger(__name__)

_COMPLEX_DTYPE_MAP = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


class EcTransOctahedralSHTModule(Module):
    """Octahedral SHT based on ecTrans assets (via ectrans4py or precomputed npz).

    Notes
    -----
    * Expects input fields with flattened octahedral reduced grid:
      ``[batch, ensemble, grid_points, variables]``.
    * Returns spectrum:
      ``[batch, ensemble, l, m, variables]``, where `l` and `m` are total and zonal wavenumber,
      respectively. The lower triangular parts of the spectrum is stored as zeroes.
    """

    def __init__(self, truncation: int, dtype: torch.dtype = torch.float32, filepath: str | Path | None = None) -> None:
        super().__init__()
        self.truncation = truncation
        self.n_lat_nh = truncation + 1
        self.dtype = dtype
        self.n_lon_for_each_lat_nh = np.array([20 + 4 * i for i in range(self.n_lat_nh)])
        self.highest_zonal_wavenumber_per_lat_nh = None
        lons_per_lat = [20 + 4 * i for i in range(truncation + 1)]
        self.lons_per_lat = lons_per_lat + lons_per_lat[::-1]
        self.n_grid_points = 2 * int(sum(lons_per_lat))
        # Needed to access the corresponding grid points from the 1D input fields
        self.cumsum_indices = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(self.lons_per_lat), dim=0)])

        symmetric, antisymmetric, gaussian_weights = self._get_polynomials_and_weights(filepath)

        self._register_polynomials(symmetric, antisymmetric, gaussian_weights)

        # Padding required to pad up the maximum wavenumber of the rfft output

        padding = [self.highest_zonal_wavenumber_per_lat_nh[-1] - m for m in self.highest_zonal_wavenumber_per_lat_nh]
        self.padding = padding + padding[::-1]

        self.highest_zonal_wavenumber_per_lat_nh = torch.from_numpy(self.highest_zonal_wavenumber_per_lat_nh)

    def _register_polynomials(
        self, symmetric: torch.Tensor, antisymmetric: torch.Tensor, gaussian_weights: torch.Tensor
    ) -> None:

        symmetric *= gaussian_weights.view(1, 1, -1)
        antisymmetric *= gaussian_weights.view(1, 1, -1)

        self.register_buffer("symmetric", symmetric, persistent=False)
        self.register_buffer("antisymmetric", antisymmetric, persistent=False)

    @cached_property
    def n_lats_per_wavenumber(self) -> list[int]:
        # Calculate latitudes involved in Legendre transform for each zonal wavenumber m
        assert self.highest_zonal_wavenumber_per_lat_nh is not None

        n_lats_per_wavenumber = np.zeros(self.truncation + 1, dtype=np.int32)
        for i in range(self.truncation + 1):
            n_lats_per_wavenumber[i] = self.highest_zonal_wavenumber_per_lat_nh[
                self.highest_zonal_wavenumber_per_lat_nh >= i
            ].shape[0]
        return n_lats_per_wavenumber

    def generate(self):
        # Fetch relevant arrays from ecTrans
        # Note that all of these arrays (including the input points-per-latitude array) are
        # specified across the full globe, pole to pole
        try:
            import ectrans4py  # type: ignore
        except Exception as exc:  # pragma: no cover
            msg = (
                "ectrans4py is required to generate octahedral SHT assets. "
                "Either install ectrans4py or provide a precomputed npz via `filepath=`."
            )
            raise ModuleNotFoundError(msg) from exc

        poly_size = sum(self.truncation + 2 - im for im in range(self.truncation + 1))

        (highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials) = ectrans4py.get_legendre_assets(
            2 * self.n_lat_nh,
            self.truncation,
            2 * self.n_lat_nh,
            poly_size,
            np.array(self.lons_per_lat),
            1,
        )
        return highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials

    def generate_and_save(self, filepath: Path):
        highest_zonal_wavenumber_per_lat, gaussian_weights, legendre_polynomials = self.generate()
        if filepath:
            np.savez(
                filepath,
                legendre_polynomials=legendre_polynomials,
                gaussian_weights=gaussian_weights,
                highest_zonal_wavenumber_per_lat=highest_zonal_wavenumber_per_lat,
            )
        return highest_zonal_wavenumber_per_lat, gaussian_weights, legendre_polynomials

    def load_from_disk(self, filepath: Path):
        loaded_assets = np.load(filepath)
        return (
            loaded_assets["highest_zonal_wavenumber_per_lat"],
            loaded_assets["gaussian_weights"],
            loaded_assets["legendre_polynomials"],
        )

    def _get_polynomials_and_weights(self, filepath: Path | str = None) -> list[torch.Tensor]:
        """Provides associated Legendre polynomials.

        Either loads precomputed polynomials and normalisation from disk or generates them via ectrans4py. Note
        that the latter requires ectrans to be installed in your environment.

        Parameters
        ----------
        filepath : Path, optional
            Path to polynomials, by default None

        Returns
        -------
        list[torch.Tensor]
            Returns symmetric and antisymmetric polynomials and normalisation
        """

        if filepath and Path(filepath).exists():
            self.highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = self.load_from_disk(
                filepath
            )
        else:
            self.highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = self.generate_and_save(
                filepath
            )

        # Flatten Legendre polynomial array to make it easier to unpack
        all_legendre_polynomials = all_legendre_polynomials.flatten()

        # Extract just the Northern hemisphere for these arrays
        self.highest_zonal_wavenumber_per_lat_nh = self.highest_zonal_wavenumber_per_lat[: self.n_lat_nh]

        gaussian_weights = gaussian_weights[: self.n_lat_nh]

        # Read Legendre polynomials, looping over each zonal wavenumber m
        # We separate symmetric from antisymmetric polynomials to reduce computational cost

        symmetric, antisymmetric = [], []
        m_off_symm, m_off_anti = 0, 0
        off_symm, off_anti = 0, 0
        for m in range(self.truncation + 1):
            i_s = (m + 1) % 2
            i_a = m % 2

            n_lats = int(self.n_lats_per_wavenumber[m])
            n_total_values = self.truncation - m + 2  # Number of total wavenumbers valid at this m

            # Determine the maximum total wavenumber for this m, symm and antisymmetric
            max_total_wavenumber_symm = (self.truncation - m + 3) // 2
            max_total_wavenumber_anti = (self.truncation - m + 2) // 2

            symm_matrix = np.zeros(((self.truncation + 3) // 2, self.n_lat_nh), dtype=polytype)
            offset = (self.truncation + 3) // 2 - max_total_wavenumber_symm

            # Parse symmetric Legendre polynomials
            for i in range(max_total_wavenumber_symm):
                # Construct slicing indices for unpacked and packed arrays

                # In the packed array data for each (zonal, total) wavenumber are zero padded up to
                # Hence we have to add an extra offset to step through the zeroed phony latitudes
                all_1 = m_off_symm + 2 * i * self.n_lat_nh + (self.n_lat_nh - n_lats)
                all_2 = m_off_symm + (2 * i + 1) * self.n_lat_nh

                symm_matrix[i + offset, self.n_lat_nh - n_lats :] = all_legendre_polynomials[all_1:all_2]

            if i_s == 1:
                symm_matrix[offset, :] = 0

            symmetric.append(torch.from_numpy(symm_matrix))

            off_symm += max_total_wavenumber_symm * n_lats

            # Parse antisymmetric Legendre polynomials
            anti_matrix = np.zeros(((self.truncation + 2) // 2, self.n_lat_nh), dtype=polytype)

            offset = (self.truncation + 2) // 2 - max_total_wavenumber_anti
            for i in range(max_total_wavenumber_anti):

                # Ditto comment above for zero padding in the packed array
                all_1 = m_off_anti + (2 * i + 1) * self.n_lat_nh + (self.n_lat_nh - n_lats)
                all_2 = m_off_anti + (2 * i + 2) * self.n_lat_nh

                # Copy latitudes for this (zonal, total) wavenumber from packed to unpacked array
                anti_matrix[i + offset, self.n_lat_nh - n_lats :] = all_legendre_polynomials[all_1:all_2]

            if i_a == 1:
                anti_matrix[offset, :] = 0

            antisymmetric.append(torch.from_numpy(anti_matrix))

            off_anti += max_total_wavenumber_anti * n_lats

            # Offset into flattened Legendre polynomial array
            if m % 2 == 0:
                m_off_symm += (n_total_values + 1) * self.n_lat_nh
                m_off_anti += (n_total_values - 1) * self.n_lat_nh
            else:
                m_off_symm += (n_total_values - 1) * self.n_lat_nh
                m_off_anti += (n_total_values + 1) * self.n_lat_nh

        symmetric = torch.stack(symmetric)[:, 1:, :].to(_COMPLEX_DTYPE_MAP[self.dtype])
        antisymmetric = torch.stack(antisymmetric).to(_COMPLEX_DTYPE_MAP[self.dtype])
        gaussian_weights = torch.from_numpy(gaussian_weights).to(self.dtype)

        return symmetric, antisymmetric, gaussian_weights

    def longitudinal_rfft(self, x: torch.Tensor, highest_zonal_wavenumber_per_lat=None, padding=None) -> torch.Tensor:
        """Performs rfft along the longitude.

        Cuts off the result at the highest zonal wavenumber to avoid aliasing effects. Padd up
        to the highest possible wavenumber to put in matrix form.

        Parameters
        ----------
        x : torch.Tensor
            field [bs, ens, grid, vars]

        highest_zonal_wavenumber_per_lat : list[int]
            list of length {number of latitudes pole-to-pole} which contains the highest
            zonal wavenumber that must be kept for each latitude. By default takes the
            cubic truncation stored in self.

        padding : list[int]
            list of length {number of latitudes pole-to-pole} which contains the number of zeros required to
            pad up to the maximum zonal wavenumber, so all latitudes are equal length on exiting the Fourier
            transform

        Returns
        -------
        torch.Tensor
            intermediate state after Fourier transform [bs, ens, lat, m, vars]
        """

        if highest_zonal_wavenumber_per_lat is None:
            highest_zonal_wavenumber_per_lat = self.highest_zonal_wavenumber_per_lat

        if padding is None:
            padding = self.padding

        four_out = []
        for i in range(2 * self.truncation + 2):
            out = rfft(x[:, :, self.cumsum_indices[i] : self.cumsum_indices[i + 1], :], axis=2, norm="forward")[
                :, :, : highest_zonal_wavenumber_per_lat[i] + 1, :
            ]
            four_out.append(
                torch.cat([out, torch.zeros((*out.shape[:2], padding[i], out.shape[-1]), device=out.device)], dim=2)
            )

        return torch.stack(four_out, dim=2)

    def legendre(self, x: torch.Tensor) -> torch.Tensor:
        """Performs legendre transform.

        Uses the symmetry of the legendre polynomials by adding and substructing the southern hemisphere
        from the northern hemisphere for symmetric and antisymmetric part, respectively.

        Parameters
        ----------
        x : torch.Tensor
            fourier-transformed field [bs, ens, lat, m, vars]

        Returns
        -------
        torch.Tensor
            spectrum [bs, ens, l, m, vars]
        """

        # Add/substract southern hemisphere from northern hemisphere
        fourier_sh_flipped = torch.flip(x[:, :, self.n_lat_nh :, :, :], dims=[2])
        fourier_sym = x[:, :, : self.n_lat_nh, :, :] + fourier_sh_flipped
        fourier_anti = x[:, :, : self.n_lat_nh, :, :] - fourier_sh_flipped

        [bs, ens, _, mmax, nvars] = fourier_sym.shape

        # Compute symmetric and antisymmetric component
        spectrum = torch.empty(bs, ens, self.truncation + 1, mmax, nvars, dtype=x.dtype, device=x.device)
        spectrum[:, :, 1::2, :, :] = torch.einsum("mnijk,jli->mnljk", fourier_sym, self.symmetric)  # noqa: F841
        spectrum[:, :, 0::2, :, :] = torch.einsum("mnijk,jli->mnljk", fourier_anti, self.antisymmetric)  # noqa: F841

        return spectrum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fourier = self.longitudinal_rfft(x)
        spectrum = self.legendre(x_fourier)
        return spectrum


class InverseEcTransOctahedralSHTModule(EcTransOctahedralSHTModule):
    """Octahedral SHT based on ecTrans assets (via ectrans4py or precomputed npz).
    Inverse version.

    Notes
    -----
    * Expects input spectrum:
      ``[batch, ensemble, l, m, variables]``, where `l` and `m` are total and zonal wavenumber,
      respectively. The lower triangular parts of the spectrum is stored as zeroes.
    * Returns fields with flattened octahedral reduced grid:
      ``[batch, ensemble, grid_points, variables]``.
    """

    def _register_polynomials(
        self, symmetric: torch.Tensor, antisymmetric: torch.Tensor, gaussian_weights: torch.Tensor
    ) -> None:
        self.register_buffer("symmetric", symmetric, persistent=False)
        self.register_buffer("antisymmetric", antisymmetric, persistent=False)

    def inverse_longitudinal_rfft(self, x: torch.Tensor, highest_zonal_wavenumber_per_lat=None) -> torch.Tensor:
        """Performs irfft along the longitude.

        Parameters
        ----------
        x : torch.Tensor
            intermediate state before inverse Fourier transform

        highest_zonal_wavenumber_per_lat : list[int]
            list of length {number of latitudes pole-to-pole} which contains the highest
            zonal wavenumber that must be kept for each latitude. By default takes the
            cubic truncation stored in self.

        Returns
        -------
        torch.Tensor
            field
        """

        if highest_zonal_wavenumber_per_lat is None:
            highest_zonal_wavenumber_per_lat = self.highest_zonal_wavenumber_per_lat

        four_out = []
        for i in range(2 * self.truncation + 2):

            out = irfft(
                x[:, :, i, : highest_zonal_wavenumber_per_lat[i] + 1, :],
                axis=2,
                norm="forward",
                n=self.lons_per_lat[i],
            )

            four_out.append(out)

        return torch.cat(four_out, dim=2)

    def inverse_legendre(self, spectrum: torch.Tensor) -> torch.Tensor:
        spec_symm = spectrum[:, :, 1::2, :, :]
        spec_anti = spectrum[:, :, 0::2, :, :]

        fourier_symm = torch.einsum("jli, mnljk -> mnijk", self.symmetric, spec_symm)
        fourier_anti = torch.einsum("jli, mnljk -> nmijk", self.antisymmetric, spec_anti)

        fourier_nh = fourier_symm + fourier_anti
        fourier_sh = torch.flip(fourier_symm - fourier_anti, dims=[2])

        return torch.cat([fourier_nh, fourier_sh], 2)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x_fourier = self.inverse_legendre(spectrum)
        x_field = self.inverse_longitudinal_rfft(x_fourier)
        return x_field
