# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from functools import cached_property
from typing import Any
from typing import Literal

from anemoi.training.losses.multivariate_kcrps import GroupedMultivariateKernelCRPS
from anemoi.training.losses.multivariate_kcrps import MultivariateKernelCRPS


class EnergyScore(MultivariateKernelCRPS):
    """Energy score (multivariate proper scoring rule) for ensemble forecasts.

    Implemented as a special case of `MultivariateKernelCRPS` with p_norm fixed to 2.0.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        fair: bool = True,
        implementation: Literal["vectorized", "low_mem"] = "vectorized",
        ignore_nans: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fair=fair,
            p_norm=2.0,
            beta=beta,
            implementation=implementation,
            ignore_nans=ignore_nans,
            **kwargs,
        )

    @cached_property
    def name(self) -> str:
        fair_str = "f" if self.fair else ""
        return f"{fair_str}energy_b{self.beta:g}"


class GroupedEnergyScore(GroupedMultivariateKernelCRPS):
    """Grouped energy score (feature grouping only).

    Supported patch/group methods:
    - group_by_variable
    - group_by_pressurelevel
    """

    def __init__(
        self,
        *,
        patch_method: str,
        beta: float = 1.0,
        fair: bool = True,
        implementation: Literal["vectorized", "low_mem"] = "vectorized",
        ignore_nans: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            patch_method=patch_method,
            fair=fair,
            p_norm=2.0,
            beta=beta,
            implementation=implementation,
            ignore_nans=ignore_nans,
            **kwargs,
        )

    @cached_property
    def name(self) -> str:
        fair_str = "f" if self.fair else ""
        return f"{fair_str}genergy_{self.patch_method}_b{self.beta:g}"
