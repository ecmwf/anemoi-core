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

polytype = np.float64

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
      ``[batch, ensemble, l, m, variables]`` where ``l = truncation+1``.
    """

    def __init__(self, truncation: int, dtype: torch.dtype = torch.float32, filepath: str | Path | None = None) -> None:
        super().__init__()
        self.truncation = int(truncation)
        self.n_lat_nh = self.truncation + 1  # northern hemisphere latitudes (incl equator)

        # full globe points for octahedral grid: sum(lons_per_lat over all lats)
        self.n_grid_points = 2 * int(np.sum(self.n_lon_for_each_lat_nh))

        # Full-globe lons-per-lat for indexing into flattened grid
        lons_per_lat = [20 + 4 * i for i in range(self.truncation + 1)]
        lons_per_lat = lons_per_lat + lons_per_lat[::-1]
        self.cumsum_indices = [0] + np.cumsum(lons_per_lat).tolist()

        self.dtype = dtype

        symmetric, antisymmetric, gaussian_weights = self._get_polynomials_and_weights(filepath)

        # Normalise polynomials by gaussian weights (broadcast over (m, l, lat))
        symmetric = symmetric * gaussian_weights.view(1, 1, -1)
        antisymmetric = antisymmetric * gaussian_weights.view(1, 1, -1)

        self.register_buffer("symmetric", symmetric, persistent=False)
        self.register_buffer("antisymmetric", antisymmetric, persistent=False)

        # Padding required to pad up to the maximum zonal wavenumber of the rfft output
        assert self.highest_zonal_wavenumber_per_lat_nh is not None
        padding = [
            int(self.highest_zonal_wavenumber_per_lat_nh[-1] - m) for m in self.highest_zonal_wavenumber_per_lat_nh
        ]
        self.padding = padding + padding[::-1]

        self.highest_zonal_wavenumber_per_lat_nh = torch.from_numpy(self.highest_zonal_wavenumber_per_lat_nh)

    @cached_property
    def n_lon_for_each_lat_nh(self) -> np.ndarray:
        """Number of longitudes per latitude ring for NH (reduced octahedral grid)."""
        return np.asarray([20 + 4 * i for i in range(self.truncation + 1)], dtype=np.int32)

    @cached_property
    def n_lats_per_wavenumber(self) -> np.ndarray:
        """Number of latitudes involved in Legendre transform for each zonal wavenumber m."""
        assert self.highest_zonal_wavenumber_per_lat_nh is not None
        out = np.zeros(self.truncation + 1, dtype=np.int32)
        for m in range(self.truncation + 1):
            out[m] = int((self.highest_zonal_wavenumber_per_lat_nh >= m).sum())
        return out

    def _generate_assets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Legendre assets via ectrans4py (requires it to be installed)."""
        try:
            import ectrans4py  # type: ignore
        except Exception as exc:  # pragma: no cover
            msg = (
                "ectrans4py is required to generate octahedral SHT assets. "
                "Either install ectrans4py or provide a precomputed npz via `filepath=`."
            )
            raise ModuleNotFoundError(msg) from exc

        poly_size = int(sum(self.truncation + 2 - m for m in range(self.truncation + 1)))

        full_nlon = np.concatenate((self.n_lon_for_each_lat_nh, self.n_lon_for_each_lat_nh[::-1]))

        highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = ectrans4py.get_legendre_assets(
            2 * self.n_lat_nh,
            self.truncation,
            2 * self.n_lat_nh,
            poly_size,
            full_nlon,
            1,
        )
        return highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials

    def _load_assets(self, filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        loaded = np.load(filepath)
        return (
            loaded["highest_zonal_wavenumber_per_lat"],
            loaded["gaussian_weights"],
            loaded["legendre_polynomials"],
        )

    def _get_polynomials_and_weights(
        self, filepath: str | Path | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return symmetric/antisymmetric polynomials and gaussian weights."""
        fp = Path(filepath) if filepath is not None else None

        if fp is not None and fp.exists():
            highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = self._load_assets(fp)
        elif fp is None:
            msg = (
                "EcTransOctahedralSHTModule requires `filepath` to a precomputed assets npz, "
                "or a writable `filepath` plus ectrans4py installed to generate assets."
            )
            raise FileNotFoundError(msg)
        else:
            highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = self._generate_assets()
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                fp,
                legendre_polynomials=all_legendre_polynomials,
                gaussian_weights=gaussian_weights,
                highest_zonal_wavenumber_per_lat=highest_zonal_wavenumber_per_lat,
            )

        # Store metadata
        self.highest_zonal_wavenumber_per_lat = np.asarray(highest_zonal_wavenumber_per_lat)
        self.highest_zonal_wavenumber_per_lat_nh = self.highest_zonal_wavenumber_per_lat[: self.n_lat_nh]

        gaussian_weights = np.asarray(gaussian_weights)[: self.n_lat_nh]

        # Flatten polynomial array to make it easier to unpack
        all_legendre_polynomials = np.asarray(all_legendre_polynomials).flatten()

        # Read Legendre polynomials, looping over each zonal wavenumber m
        symmetric_list: list[torch.Tensor] = []
        antisymmetric_list: list[torch.Tensor] = []

        m_off_symm = 0
        m_off_anti = 0

        for m in range(self.truncation + 1):
            is_symmetric = (m + 1) % 2
            is_antisymmetric = m % 2

            n_lats = int(self.n_lats_per_wavenumber[m])
            n_total_values = self.truncation - m + 2  # number of total wavenumbers valid at this m

            # Maximum total wavenumbers for this m
            max_total_wavenumber_symm = (self.truncation - m + 3) // 2
            max_total_wavenumber_anti = (self.truncation - m + 2) // 2

            # Symmetric matrix
            symm_matrix = np.zeros(((self.truncation + 3) // 2, self.n_lat_nh), dtype=polytype)
            offset_symm = (self.truncation + 3) // 2 - max_total_wavenumber_symm

            for ii in range(max_total_wavenumber_symm):
                all_1 = m_off_symm + 2 * ii * self.n_lat_nh + (self.n_lat_nh - n_lats)
                all_2 = m_off_symm + (2 * ii + 1) * self.n_lat_nh
                symm_matrix[ii + offset_symm, self.n_lat_nh - n_lats :] = all_legendre_polynomials[all_1:all_2]

            if is_symmetric == 1:
                symm_matrix[offset_symm, :] = 0.0

            symmetric_list.append(torch.from_numpy(symm_matrix))

            # Antisymmetric matrix
            anti_matrix = np.zeros(((self.truncation + 2) // 2, self.n_lat_nh), dtype=polytype)
            offset_anti = (self.truncation + 2) // 2 - max_total_wavenumber_anti

            for ii in range(max_total_wavenumber_anti):
                all_1 = m_off_anti + (2 * ii + 1) * self.n_lat_nh + (self.n_lat_nh - n_lats)
                all_2 = m_off_anti + (2 * ii + 2) * self.n_lat_nh
                anti_matrix[ii + offset_anti, self.n_lat_nh - n_lats :] = all_legendre_polynomials[all_1:all_2]

            if is_antisymmetric == 1:
                anti_matrix[offset_anti, :] = 0.0

            antisymmetric_list.append(torch.from_numpy(anti_matrix))

            # Offset into flattened polynomial array
            if m % 2 == 0:
                m_off_symm += (n_total_values + 1) * self.n_lat_nh
                m_off_anti += (n_total_values - 1) * self.n_lat_nh
            else:
                m_off_symm += (n_total_values - 1) * self.n_lat_nh
                m_off_anti += (n_total_values + 1) * self.n_lat_nh

        complex_dtype = _COMPLEX_DTYPE_MAP.get(self.dtype, torch.complex64)

        symmetric = torch.stack(symmetric_list)[:, 1:, :].to(complex_dtype)
        antisymmetric = torch.stack(antisymmetric_list).to(complex_dtype)
        gaussian_weights_t = torch.from_numpy(gaussian_weights).to(self.dtype)

        return symmetric, antisymmetric, gaussian_weights_t

    def longitudinal_rfft(self, x: torch.Tensor) -> torch.Tensor:
        """rFFT along longitude for each latitude ring (returned as padded matrix)."""
        fourier_out: list[torch.Tensor] = []
        n_rings = 2 * (self.truncation + 1)

        for ring in range(n_rings):
            start = self.cumsum_indices[ring]
            end = self.cumsum_indices[ring + 1]

            out = rfft(x[:, :, start:end, :], dim=2, norm="forward")
            max_m = int(self.highest_zonal_wavenumber_per_lat[ring]) + 1
            out = out[:, :, :max_m, :]

            pad = int(self.padding[ring])
            if pad > 0:
                out = torch.cat(
                    [out, torch.zeros((*out.shape[:2], pad, out.shape[-1]), device=out.device, dtype=out.dtype)],
                    dim=2,
                )
            fourier_out.append(out)

        return torch.stack(fourier_out, dim=2)

    def legendre(self, x: torch.Tensor) -> torch.Tensor:
        """Legendre transform, using NH/SH symmetry for efficiency."""
        fourier_sh_flipped = torch.flip(x[:, :, self.n_lat_nh :, :, :], dims=[2])
        fourier_norm_sym = x[:, :, : self.n_lat_nh, :, :] + fourier_sh_flipped
        fourier_norm_anti = x[:, :, : self.n_lat_nh, :, :] - fourier_sh_flipped

        bs, ens, _, mmax, nvars = fourier_norm_sym.shape

        spectrum = torch.empty((bs, ens, self.truncation + 1, mmax, nvars), dtype=x.dtype, device=x.device)

        spectrum[:, :, 1::2, :, :] = torch.einsum("beimv,mli->belmv", fourier_norm_sym, self.symmetric)
        spectrum[:, :, 0::2, :, :] = torch.einsum("beimv,mli->belmv", fourier_norm_anti, self.antisymmetric)

        return spectrum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fourier = self.longitudinal_rfft(x)
        return self.legendre(x_fourier)
