# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch
import einops

from torch.nn import Parameter, Module
from anemoi.models.layers.sht import (
    CartesianRealSHT,
    CartesianInverseRealSHT,
    OctahedralRealSHT,
    OctahedralInverseRealSHT,
)


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


class SimpleOrnsteinResidual(Module):

    def __init__(
        self,
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        theta_train: bool = True,
        input_idx: list[int] = [],
        statistics: dict[str, np.ndarray] = {},
        **_
    ) -> None:

        super().__init__()

        theta_init = self.init_theta(theta_init, theta_buff, statistics)
        theta_init = np.log(theta_init / (1 - theta_init))
        theta_init = np.array(theta_init)

        weight = torch.zeros(len(input_idx))
        weight[:] = torch.from_numpy(theta_init)

        self.weight = Parameter(weight, theta_train)
        self.theta_buff = theta_buff
        self._internal_input_idx = input_idx

    def init_theta(
        self,
        theta_init: float,
        theta_buff: float,
        statistics: dict[str, np.ndarray],
    ) -> np.ndarray:
        
        theta_init = (
            theta_init
            if (theta_init != 0) or any(s not in statistics for s in {"stdev", "stdev_tend"})
            else 0.5 * (statistics["stdev_tend"] / statistics["stdev"]) ** 2
        )
        
        theta_init = (theta_init - theta_buff) / (1 - theta_buff)
        theta_init = np.where(theta_init < 1, theta_init, 0.99)
        theta_init = np.where(theta_init > 0, theta_init, 0.01)

        return theta_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (
            + (1 - torch.sigmoid(self.weight) * (1 - self.theta_buff) - self.theta_buff)
            * x[..., self._internal_input_idx]
        )


class BasicOrnsteinResidual(Module):

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
        regressors: list[str] = [],
        input_idx: list[int] = [],
        variables: dict[str, int] = {},
        statistics: dict[str, np.ndarray] = {},
    ) -> None:
        
        super().__init__()

        theta_init = self.init_theta(theta_init, theta_buff, statistics)
        theta_init = np.sqrt(4 * np.pi) * np.log(theta_init / (1 - theta_init))
        theta_init = np.array(theta_init)
        
        weight = torch.zeros(len(regressors) + 2, len(input_idx), lmax, lmax, 2)
        weight[0, :, 0, 0, 0] = torch.from_numpy(theta_init)

        self.weight = Parameter(weight)

        if grid == "octahedral":
            self.isht = OctahedralInverseRealSHT(nlat, lmax, lmax)
            self.values_reshape_for = f"... values var -> ... var values"
            self.values_reshape_inv = f"... var values -> ... values var"
            self.kwargs_reshape_for = {}
        else:
            self.isht = CartesianInverseRealSHT(nlat, nlon, lmax, grid)
            self.values_reshape_for = f"... ({node_order.replace("-", " ")}) var -> ... var lat lon"
            self.values_reshape_inv = f"... var lat lon -> ... ({node_order.replace("-", " ")}) var"
            self.kwargs_reshape_for = {"lat": nlat, "lon": nlon}

        self._regressors_input_idx = [variables[f] for f in regressors]
        self._internal_input_idx = input_idx

        muzero = torch.ones_like(weight)
        coords = slice(0, 1) if zmean_term else slice(None)
        muzero[1, :, coords, coords, :] = 0

        self.register_buffer("muzero", muzero)
        self.theta_buff = theta_buff

    def init_theta(
        self,
        theta_init: float,
        theta_buff: float,
        statistics: dict[str, np.ndarray],
    ) -> np.ndarray:
        
        theta_init = (
            theta_init
            if (theta_init != 0) or any(s not in statistics for s in {"stdev", "stdev_tend"})
            else 0.5 * (statistics["stdev_tend"] / statistics["stdev"]) ** 2
        )
        
        theta_init = (theta_init - theta_buff) / (1 - theta_buff)
        theta_init = np.where(theta_init < 1, theta_init, 0.99)
        theta_init = np.where(theta_init > 0, theta_init, 0.01)

        return theta_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        weight = self.isht(torch.view_as_complex(self.weight * self.muzero))
        weight = einops.rearrange(weight, self.values_reshape_inv)

        return (
            + (1 - torch.sigmoid(weight[0, ...]) * (1 - self.theta_buff) - self.theta_buff)
            * x[..., self._internal_input_idx]
            + weight[1, ...]
            + sum(
                weight[i + 2, ...] * x[..., k].unsqueeze(-1)
                for i, k in enumerate(self._regressors_input_idx)
            )
        )


class CompleteOrnsteinResidual(BasicOrnsteinResidual):

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.05,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
        regressors: list[str] = [],
        anti_aliasing: bool = True,
        skip_blur: list[str] = [],
        input_idx: list[int] = [],
        variables: dict[str, int] = {},
        statistics: dict[str, np.ndarray] = {},
    ) -> None:
        
        super().__init__(
            nlat=nlat,
            nlon=nlon,
            grid=grid,
            lmax=lmax,
            node_order=node_order,
            theta_init=theta_init,
            theta_buff=theta_buff,
            zmean_term=zmean_term,
            regressors=regressors,
            input_idx=input_idx,
            variables=variables,
            statistics=statistics,
        )

        if grid == "octahedral":
            self.x_fsht = OctahedralRealSHT(nlat)
            self.x_isht = OctahedralInverseRealSHT(nlat, self.x_fsht.lmax, self.x_fsht.mmax)
        else:
            self.x_fsht = CartesianRealSHT(nlat, nlon, grid)
            self.x_isht = CartesianInverseRealSHT(nlat, nlon, self.x_fsht.lmax, grid)

        self._blurring_input_idx = [
            int(idx)
            for idx in input_idx
            if idx not in [variables.get(v) for v in skip_blur]
        ]

        self._var_axis_tail = (
            (slice(None),) * len(self.kwargs_reshape_for)
            if len(self.kwargs_reshape_for) > 0
            else (slice(None),)
        )

        filter = torch.ones(len(self._blurring_input_idx), self.x_fsht.lmax)
        filter = filter * max(theta_init, 0.01) / (0.5 - max(theta_init, 0.01))
        filter = torch.sqrt(filter / self.x_fsht.lmax)

        walias = torch.zeros(len(self._blurring_input_idx), lmax, lmax, 2)

        self.filter = Parameter(filter)
        self.walias = Parameter(walias)

        self.lpass_filter = (
            self.blur_with_anti_aliasing
            if anti_aliasing
            else self.blur_without_anti_aliasing
        )

    def x_filter(self) -> torch.Tensor:

        filter = torch.square(self.filter)
        filter = torch.cumsum(filter, -1)
        filter = filter / (1 + filter)

        return filter

    def w_filter(self) -> torch.Tensor:

        walias = self.isht(torch.view_as_complex(self.walias))

        return torch.sigmoid(walias)

    def blur_without_anti_aliasing(self, x_blur: torch.Tensor) -> torch.Tensor:

        x_blur = self.x_fsht(x_blur)
        filter = self.x_filter()

        x_blur = x_blur * (1 - filter.unsqueeze(-1))

        return self.x_isht(x_blur)

    def blur_with_anti_aliasing(self, x_blur: torch.Tensor) -> torch.Tensor:
        
        x_skip = self.x_fsht(x_blur)
        filter = self.x_filter()
        walias = self.w_filter()

        x_skip = x_skip * (1 - filter.unsqueeze(-1))

        return (
            + walias * x_blur
            + (1 - walias) * self.x_isht(x_skip)
        )

    def blurring(self, x: torch.Tensor) -> torch.Tensor:

        x = einops.rearrange(x, self.values_reshape_for, **self.kwargs_reshape_for)

        x[..., self._blurring_input_idx, *self._var_axis_tail] = self.lpass_filter(
            x[..., self._blurring_input_idx, *self._var_axis_tail]
        )

        x = einops.rearrange(x, self.values_reshape_inv)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return super().forward(self.blurring(x))
