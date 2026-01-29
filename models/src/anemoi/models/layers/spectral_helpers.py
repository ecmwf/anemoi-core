# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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


class SphericalHarmonicTransform(Module):

    def __init__(
        self,
        nlat: int,
        nlon: list[int],
        lmax: int | None = None,
        mmax: int | None = None
    ) -> None:

        super().__init__()

        self.lmax = lmax or nlat
        self.mmax = mmax or nlat

        self.nlat = nlat
        self.nlon = nlon

        self.slon = [0] + list(np.cumsum(self.nlon))[:-1]
        self.rlon = [nlat + 8 - nlon // 2 for nlon in self.nlon]

        theta, weight = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        pct = precompute_legpoly(self.mmax, self.lmax, theta)
        pct = torch.from_numpy(pct)

        weight = torch.from_numpy(weight)
        weight = torch.einsum("mlk, k -> mlk", pct, weight)

        self.register_buffer("weight", weight, persistent=False)

    def rfft(self, x: Tensor) -> Tensor:

        return torch.fft.rfft(input=x, norm="forward")

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


class InverseSphericalHarmonicTransform(Module):

    def __init__(
        self,
        nlat: int,
        nlon: list[int],
        lmax: int | None = None,
        mmax: int | None = None
    ) -> None:

        super().__init__()

        self.lmax = lmax or nlat
        self.mmax = mmax or nlat

        self.nlat = nlat
        self.nlon = nlon

        theta, _ = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        pct = precompute_legpoly(self.mmax, self.lmax, theta, inverse=True)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def irfft(self, x: Tensor, nlon: int) -> Tensor:

        return torch.fft.irfft(
            input=x,
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
