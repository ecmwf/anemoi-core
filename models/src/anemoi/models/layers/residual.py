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


import numpy as np
import torch
import einops
import logging

from torch.nn import Parameter, Module

LOGGER = logging.getLogger(__name__)


def legendre_gauss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b].
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def clenshaw_curtiss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""
    Computation of the Clenshaw-Curtis quadrature nodes and weights.

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
        v = 0 - v[:-1] - v[-1: 0: -1]

        g0 = (-1) * np.ones(n1)
        g0[l] = g0[l] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1 ** 2 - 1 + (n1 % 2))
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
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
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
        
    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1. / norm_factor if inverse else norm_factor

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
                + x * np.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m)) * vdm[m, l - 1, :]
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
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)


class InverseRealSHT(Module):

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


class IdentityResidual(Module):

    def __init__(self, input_idx: list[int] = [], **_) -> None:

        super().__init__()
        self._internal_input_idx = input_idx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x[..., self._internal_input_idx]


class NoResidual(Module):

    def __init__(self, input_idx: list[int] = [], **_) -> None:

        super().__init__()
        self._internal_input_idx = input_idx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.zeros_like(x[..., self._internal_input_idx])


class OrnsteinResidual(Module):

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.05,
        theta_buff: float = 0.00,
        zmean_term: bool = False,
        regressors: list[str] = [],
        input_idx: list[int] = [],
        variables: dict[str, int] = {},
        statistics: dict[str, np.ndarray] = {},
    ) -> None:
        
        super().__init__()

        theta_init = self.build_first_theta(theta_init, theta_buff, statistics)
        theta_init = np.sqrt(4 * np.pi) * np.log(theta_init / (1 - theta_init))
        theta_init = np.array(theta_init)
        
        weight = torch.zeros(len(regressors) + 2, len(input_idx), lmax, lmax, 2)
        weight[0, :, 0, 0, 0] = torch.from_numpy(theta_init)

        self.weight = Parameter(weight)
        self.isht = InverseRealSHT(nlat, nlon, lmax, grid)

        self.node_order = node_order.replace("-", " ")
        self._regressors_input_idx = [variables[f] for f in regressors]
        self._internal_input_idx = input_idx

        muzero = torch.ones_like(weight)
        coords = slice(0, 1) if zmean_term else slice(None)
        muzero[1, :, coords, coords, :] = 0

        self.register_buffer("muzero", muzero)
        self.theta_buff = theta_buff

    def build_first_theta(
        self,
        theta_init: float,
        theta_buff: float,
        statistics: dict[str, np.ndarray],
    ) -> torch.Tensor:
        
        theta_init = (
            theta_init
            if (theta_init > 0) or any(s not in statistics for s in {"stdev", "stdev_tend"})
            else 0.5 * (statistics["stdev_tend"] / statistics["stdev"]) ** 2
        )
        
        theta_init = (theta_init - theta_buff) / (1 - theta_buff)
        theta_init = np.where(theta_init < 1, theta_init, 0.95)
        theta_init = np.where(theta_init > 0, theta_init, 0.05)

        return theta_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        weight = self.isht(torch.view_as_complex(self.weight * self.muzero))
        weight = einops.rearrange(weight, f"coef var lat lon -> coef ({self.node_order}) var")

        return (
            + (1 - torch.sigmoid(weight[0, ...]) * (1 - self.theta_buff) - self.theta_buff)
            * x[..., self._internal_input_idx]
            + weight[1, ...]
            + sum(
                weight[i + 2, ...] * x[..., k].unsqueeze(-1)
                for i, k in enumerate(self._regressors_input_idx)
            )
        )
