# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# (C) Copyright 2022 The torch-harmonics Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# (C) Copyright 2023 Matthew Price, Jason McEwen and contributors (S2FFT).
#
# Differentiable and accelerated spherical harmonic and Wigner transforms,
# Journal of Computational Physics, 2024, arXiv:2311.14670.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch
from torch import Tensor
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
        l = len(N)
        m = n1 - l

        v = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
        v = 0 - v[:-1] - v[-1:0:-1]

        g0 = (-1) * np.ones(n1)
        g0[l] = g0[l] + n1
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

    # Initial values to start the recursion
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # Fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        vdm[l - 1, l, :] = np.sqrt(2 * l + 1) * x * vdm[l - 1, l - 1, :]
        vdm[l, l, :] = np.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * vdm[l - 1, l - 1, :]

    # Fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l - 1):
            vdm[m, l, :] = (
                +x * np.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m)) * vdm[m, l - 1, :]
                - np.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m)) * vdm[m, l - 2, :]
            )

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                vdm[:, l, :] = vdm[:, l, :] * np.sqrt(2 * l + 1)
            else:
                vdm[:, l, :] = vdm[:, l, :] / np.sqrt(2 * l + 1)

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
