# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class NoiseConditioningSettings:
    """Settings for the embedding that tells the model the current noise level or time."""

    channels: int = 32
    cond_dim: int = 16

    @classmethod
    def from_config(cls, config: Any) -> NoiseConditioningSettings:
        return cls(
            channels=int(_get_config_value(config, "noise_channels", cls.channels)),
            cond_dim=int(_get_config_value(config, "noise_cond_dim", cls.cond_dim)),
        )


@dataclass(frozen=True)
class EdmSettings:
    """Settings for the EDM diffusion objective and sampler schedule."""

    sigma_data: float = 1.0
    sigma_max: float = 100.0
    sigma_min: float = 0.02
    rho: float = 7.0

    @classmethod
    def from_config(cls, config: Any) -> EdmSettings:
        return cls(
            sigma_data=float(_get_config_value(config, "sigma_data", cls.sigma_data)),
            sigma_max=float(_get_config_value(config, "sigma_max", cls.sigma_max)),
            sigma_min=float(_get_config_value(config, "sigma_min", cls.sigma_min)),
            rho=float(_get_config_value(config, "rho", cls.rho)),
        )
