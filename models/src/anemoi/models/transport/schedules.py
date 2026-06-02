# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import math
from abc import ABC
from abc import abstractmethod

import torch

from anemoi.models.distributed.random import use_synced_torch_rng
from anemoi.models.transport.paths import karras_sigma_from_unit_time

# Small tolerance used when a sigma schedule provides an explicit final noise value.
DEFAULT_FINAL_SIGMA_EPS = 1e-10


class SamplingSchedule(ABC):
    """Base class for inference schedules passed to transport samplers."""

    def __init__(self, num_steps: int) -> None:
        self._validate_num_steps(num_steps)
        self.num_steps = int(num_steps)

    def get_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Build the schedule values used by an inference sampler."""
        values = self._build_schedule(
            device=device,
            dtype_compute=dtype_compute,
        )
        self._validate_schedule(values)
        return self._finalize_schedule(values)

    @abstractmethod
    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Generate schedule-specific values."""
        pass

    @staticmethod
    def _validate_num_steps(num_steps: int) -> None:
        if int(num_steps) < 1:
            raise ValueError("num_steps must be at least 1.")

    def _validate_schedule(self, values: torch.Tensor) -> None:
        if values.ndim != 1:
            raise ValueError(f"Sampling schedule must be 1D, got shape {tuple(values.shape)}.")
        if values.numel() == 0:
            raise ValueError("Sampling schedule must contain at least one value.")
        if not torch.isfinite(values).all():
            raise ValueError("Sampling schedule must contain only finite values.")

    def _finalize_schedule(self, values: torch.Tensor) -> torch.Tensor:
        return values


class SigmaSchedule(SamplingSchedule):
    """Base class for EDM sigma schedules."""

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
    ) -> None:
        self._validate_sigma_parameters(sigma_max=sigma_max, sigma_min=sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        super().__init__(num_steps=num_steps)

    @staticmethod
    def _validate_sigma_parameters(sigma_max: float, sigma_min: float) -> None:
        if sigma_min <= 0:
            raise ValueError("sigma_min must be strictly positive; the final zero is added separately.")
        if sigma_max <= 0:
            raise ValueError("sigma_max must be strictly positive.")
        if sigma_max < sigma_min:
            raise ValueError("sigma_max must be greater than or equal to sigma_min.")

    def _validate_schedule(self, values: torch.Tensor) -> None:
        super()._validate_schedule(values)
        if values.numel() == self.num_steps + 1:
            last = values[-1].item()
            if last < 0 or last > DEFAULT_FINAL_SIGMA_EPS:
                raise ValueError("Sigma schedule with an explicit final value must end at zero.")

    def _finalize_schedule(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == self.num_steps + 1:
            if values[-1].item() != 0.0:
                values = values.clone()
                values[-1] = 0.0
            return values

        if values.numel() != self.num_steps:
            raise ValueError(
                f"Sigma schedule must contain {self.num_steps} values before the final zero, "
                f"or {self.num_steps + 1} including it; got {values.numel()}.",
            )

        return torch.cat((values, values.new_zeros(1)))


class KarrasSigmaSchedule(SigmaSchedule):
    """Karras et al. EDM sigma schedule."""

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
        rho: float = 7.0,
    ) -> None:
        super().__init__(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps)
        self.rho = float(rho)

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        if self.num_steps == 1:
            return torch.tensor([self.sigma_max], device=device, dtype=dtype_compute)

        step_indices = torch.arange(self.num_steps, device=device, dtype=dtype_compute)
        unit_time = step_indices / (self.num_steps - 1.0)
        return karras_sigma_from_unit_time(
            unit_time,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            rho=self.rho,
        )


class LinearSigmaSchedule(SigmaSchedule):
    """Linear schedule in sigma space."""

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return torch.linspace(
            self.sigma_max,
            self.sigma_min,
            self.num_steps,
            device=device,
            dtype=dtype_compute,
        )


class CosineSigmaSchedule(SigmaSchedule):
    """Cosine EDM sigma schedule."""

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
        s: float = 0.008,
    ) -> None:
        super().__init__(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps)
        self.s = float(s)

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return cosine_sigma_from_unit_time(
            torch.linspace(0.0, 1.0, self.num_steps, device=device, dtype=dtype_compute),
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            s=self.s,
        )


class ExponentialSigmaSchedule(SigmaSchedule):
    """Exponential schedule, linear in log-sigma space."""

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return exponential_sigma_from_unit_time(
            torch.linspace(0.0, 1.0, self.num_steps, device=device, dtype=dtype_compute),
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
        )


class TimeSchedule(SamplingSchedule):
    """Base class for time schedules used by vector-field transport samplers."""

    def _validate_schedule(self, values: torch.Tensor) -> None:
        super()._validate_schedule(values)
        if values.numel() != self.num_steps + 1:
            raise ValueError(
                f"Time schedule must contain {self.num_steps + 1} values; got {values.numel()}.",
            )
        if values[0].item() != 0.0 or values[-1].item() != 1.0:
            raise ValueError("Time schedule must start at 0 and end at 1.")
        if not torch.all(values[1:] >= values[:-1]):
            raise ValueError("Time schedule must be monotonically increasing.")


class UnitTimeSchedule(TimeSchedule):
    """Uniform time schedule from 0 to 1."""

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, self.num_steps + 1, device=device, dtype=dtype_compute)


class TrainingConditionDistribution(ABC):
    """Base class for one-shot training condition sampling."""

    def sample(
        self,
        shape: dict[str, tuple[int, ...]],
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample one scalar condition per batch and ensemble member."""
        batch_size, ensemble_size = _validate_condition_shape(shape)
        base = self._sample_base(
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            device=device,
            dtype=dtype,
        )
        base = base[:, None, :, None, None]
        return {dataset_name: base for dataset_name in shape}

    @abstractmethod
    def _sample_base(
        self,
        *,
        batch_size: int,
        ensemble_size: int,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        pass


class SigmaTrainingDistribution(TrainingConditionDistribution):
    """Base class for EDM training sigma distributions."""

    def __init__(self, sigma_max: float, sigma_min: float) -> None:
        SigmaSchedule._validate_sigma_parameters(sigma_max=sigma_max, sigma_min=sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)

    def _sample_base(
        self,
        *,
        batch_size: int,
        ensemble_size: int,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        with use_synced_torch_rng():
            unit_time = torch.rand((batch_size, ensemble_size), device=device, dtype=dtype)
        return self._sigma_from_unit_time(unit_time)

    @abstractmethod
    def _sigma_from_unit_time(self, unit_time: torch.Tensor) -> torch.Tensor:
        pass


class KarrasSigmaTrainingDistribution(SigmaTrainingDistribution):
    """Continuous Karras sigma distribution used by EDM training."""

    def __init__(self, sigma_max: float, sigma_min: float, rho: float = 7.0) -> None:
        super().__init__(sigma_max=sigma_max, sigma_min=sigma_min)
        self.rho = float(rho)

    def _sigma_from_unit_time(self, unit_time: torch.Tensor) -> torch.Tensor:
        return karras_sigma_from_unit_time(
            unit_time,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            rho=self.rho,
        )


class LinearSigmaTrainingDistribution(SigmaTrainingDistribution):
    """Continuous linear sigma distribution."""

    def _sigma_from_unit_time(self, unit_time: torch.Tensor) -> torch.Tensor:
        return self.sigma_max + unit_time * (self.sigma_min - self.sigma_max)


class ExponentialSigmaTrainingDistribution(SigmaTrainingDistribution):
    """Continuous log-linear sigma distribution."""

    def _sigma_from_unit_time(self, unit_time: torch.Tensor) -> torch.Tensor:
        return exponential_sigma_from_unit_time(
            unit_time,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
        )


class CosineSigmaTrainingDistribution(SigmaTrainingDistribution):
    """Continuous cosine sigma distribution."""

    def __init__(self, sigma_max: float, sigma_min: float, s: float = 0.008) -> None:
        super().__init__(sigma_max=sigma_max, sigma_min=sigma_min)
        self.s = float(s)

    def _sigma_from_unit_time(self, unit_time: torch.Tensor) -> torch.Tensor:
        return cosine_sigma_from_unit_time(
            unit_time,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            s=self.s,
        )


class UniformTimeTrainingDistribution(TrainingConditionDistribution):
    """Uniform bridge-time distribution used by stochastic-interpolant training."""

    def _sample_base(
        self,
        *,
        batch_size: int,
        ensemble_size: int,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        with use_synced_torch_rng():
            return torch.rand((batch_size, ensemble_size), device=device, dtype=dtype)


def exponential_sigma_from_unit_time(
    unit_time: torch.Tensor,
    *,
    sigma_max: float,
    sigma_min: float,
) -> torch.Tensor:
    """Map unit time to a log-linear EDM sigma path."""
    log_sigma_max = math.log(sigma_max)
    log_sigma_min = math.log(sigma_min)
    return torch.exp(log_sigma_max + unit_time * (log_sigma_min - log_sigma_max))


def cosine_sigma_from_unit_time(
    unit_time: torch.Tensor,
    *,
    sigma_max: float,
    sigma_min: float,
    s: float = 0.008,
) -> torch.Tensor:
    """Map unit time to the cosine EDM sigma path."""
    theta_max = math.atan(sigma_max)
    theta_min = math.atan(sigma_min)
    t_max = (2 * theta_max / math.pi) * (1 + s) - s
    t_min = (2 * theta_min / math.pi) * (1 + s) - s
    t = t_max + unit_time * (t_min - t_max)
    alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    return torch.sqrt((1 - alpha_bar) / alpha_bar)


def _validate_condition_shape(shape: dict[str, tuple[int, ...]]) -> tuple[int, int]:
    dataset_names = list(shape)
    if not dataset_names:
        raise ValueError("Condition distribution requires at least one dataset shape.")
    ref_shape = shape[dataset_names[0]]
    if len(ref_shape) != 5:
        raise ValueError("Expected 5D tensor shape (batch, time, ensemble, grid, vars) for transport conditions.")
    batch_size = ref_shape[0]
    ensemble_size = ref_shape[2]
    for dataset_name, shape_x in shape.items():
        if len(shape_x) != 5:
            raise ValueError(f"Expected 5D tensor shape for dataset '{dataset_name}'.")
        if shape_x[0] != batch_size or shape_x[2] != ensemble_size:
            raise ValueError("Batch or ensemble dimension mismatch across datasets when sampling conditions.")
    return batch_size, ensemble_size


SIGMA_SCHEDULES = {
    "karras": KarrasSigmaSchedule,
    "linear": LinearSigmaSchedule,
    "cosine": CosineSigmaSchedule,
    "exponential": ExponentialSigmaSchedule,
}

TIME_SCHEDULES = {
    "unit_time": UnitTimeSchedule,
}

SIGMA_TRAINING_DISTRIBUTIONS = {
    "karras": KarrasSigmaTrainingDistribution,
    "linear": LinearSigmaTrainingDistribution,
    "cosine": CosineSigmaTrainingDistribution,
    "exponential": ExponentialSigmaTrainingDistribution,
}

TIME_TRAINING_DISTRIBUTIONS = {
    "uniform_time": UniformTimeTrainingDistribution,
}
