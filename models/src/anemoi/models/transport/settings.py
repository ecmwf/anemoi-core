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
from typing import Literal


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class NoiseConditioningSettings:
    """Settings for the embedding that tells the model the current noise or bridge time."""

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

    @classmethod
    def from_config(cls, config: Any) -> EdmSettings:
        return cls(
            sigma_data=float(_get_config_value(config, "sigma_data", cls.sigma_data)),
            sigma_max=float(_get_config_value(config, "sigma_max", cls.sigma_max)),
            sigma_min=float(_get_config_value(config, "sigma_min", cls.sigma_min)),
        )


@dataclass(frozen=True)
class TransportSourceSettings:
    """Settings that choose and modify the starting/source field for transport objectives."""

    kind: Literal["default", "zero", "gaussian", "reference_state"] = "default"
    scale: float = 1.0
    noise_scale: float = 0.0

    @classmethod
    def from_config(cls, config: Any) -> TransportSourceSettings:
        source_config = _get_config_value(config, "source", {})
        return cls(
            kind=str(_get_config_value(source_config, "kind", cls.kind)),
            scale=float(_get_config_value(source_config, "scale", cls.scale)),
            noise_scale=float(_get_config_value(source_config, "noise_scale", cls.noise_scale)),
        )


@dataclass(frozen=True)
class StochasticInterpolantSettings:
    """Settings for the stochastic-interpolant bridge used during training and sampling."""

    alpha_schedule: Literal["linear"] = "linear"
    beta_schedule: Literal["linear", "quadratic"] = "linear"
    sigma_schedule: Literal["brownian_bridge", "quadratic_bridge"] = "brownian_bridge"
    noise_scale: float = 1.0

    @classmethod
    def from_config(cls, config: Any) -> StochasticInterpolantSettings:
        return cls(
            alpha_schedule=str(_get_config_value(config, "si_alpha_schedule", cls.alpha_schedule)),
            beta_schedule=str(_get_config_value(config, "si_beta_schedule", cls.beta_schedule)),
            sigma_schedule=str(_get_config_value(config, "si_sigma_schedule", cls.sigma_schedule)),
            noise_scale=float(_get_config_value(config, "si_noise_scale", cls.noise_scale)),
        )
