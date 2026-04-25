# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import math
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup


DenoisingFunction = Callable[
    [
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        Optional[ProcessGroup],
        dict[str, Optional[list]],
    ],
    dict[str, torch.Tensor],
]

# Tolerance used when treating an explicitly provided final schedule value as zero.
DEFAULT_FINAL_SIGMA_EPS = 1e-10


def _expand_sigma(sigma: torch.Tensor, y: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Broadcast scalar sigma to per-dataset model-conditioning shape."""
    return {
        dataset_name: sigma.view(1, 1, 1, 1, 1).expand(y_data.shape[0], 1, y_data.shape[2], 1, 1).to(y_data.dtype)
        for dataset_name, y_data in y.items()
    }


class NoiseScheduler(ABC):
    """Base class for noise schedulers."""

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
    ):
        self._validate_scheduler_parameters(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps)

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_steps = num_steps

    def get_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        """Generate noise schedule.

        Parameters
        ----------
        device : torch.device
            Device to create tensors on
        dtype_compute : torch.dtype
            Data type for the noise schedule computation
        **kwargs
            Additional scheduler-specific parameters

        Returns
        -------
        torch.Tensor
            Noise schedule with shape (num_steps + 1,)
        """
        sigmas = self._build_schedule(
            device=device,
            dtype_compute=dtype_compute,
            **kwargs,
        )
        self._validate_schedule(sigmas)
        return self._finalize_schedule(sigmas)

    @abstractmethod
    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        """Generate the scheduler-specific path before final-zero finalization."""
        pass

    @staticmethod
    def _validate_scheduler_parameters(sigma_max: float, sigma_min: float, num_steps: int) -> None:
        if sigma_min <= 0:
            raise ValueError("sigma_min must be strictly positive; the final zero is added separately.")
        if sigma_max <= 0:
            raise ValueError("sigma_max must be strictly positive.")
        if sigma_max < sigma_min:
            raise ValueError("sigma_max must be greater than or equal to sigma_min.")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

    def _validate_schedule(self, sigmas: torch.Tensor) -> None:
        if sigmas.ndim != 1:
            raise ValueError(f"Sigma schedule must be 1D, got shape {tuple(sigmas.shape)}.")
        if sigmas.numel() == 0:
            raise ValueError("Sigma schedule must contain at least one value.")
        if not torch.isfinite(sigmas).all():
            raise ValueError("Sigma schedule must contain only finite values.")
        if sigmas.numel() == self.num_steps + 1:
            last = sigmas[-1].item()
            if last < 0 or last > DEFAULT_FINAL_SIGMA_EPS:
                raise ValueError("Sigma schedule with an explicit final value must end at zero.")

    def _finalize_schedule(self, sigmas: torch.Tensor) -> torch.Tensor:
        if sigmas.numel() == self.num_steps + 1:
            if sigmas[-1].item() != 0.0:
                sigmas = sigmas.clone()
                sigmas[-1] = 0.0
            return sigmas

        if sigmas.numel() != self.num_steps:
            raise ValueError(
                f"Sigma schedule must contain {self.num_steps} values before the final zero, "
                f"or {self.num_steps + 1} including it; got {sigmas.numel()}.",
            )

        return torch.cat((sigmas, sigmas.new_zeros(1)))


class KarrasScheduler(NoiseScheduler):
    """Karras et al. EDM schedule."""

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
        rho: float = 7.0,
        **kwargs,
    ):
        super().__init__(sigma_max, sigma_min, num_steps)
        self.rho = rho

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        if self.num_steps == 1:
            return torch.tensor([self.sigma_max], device=device, dtype=dtype_compute)

        step_indices = torch.arange(self.num_steps, device=device, dtype=dtype_compute)
        sigmas = (
            self.sigma_max ** (1.0 / self.rho)
            + step_indices
            / (self.num_steps - 1.0)
            * (self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho))
        ) ** self.rho
        sigmas = torch.cat([torch.as_tensor(sigmas), torch.zeros_like(sigmas[:1])])
        return sigmas


class LinearScheduler(NoiseScheduler):
    """Linear schedule in sigma space."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, **kwargs):
        super().__init__(sigma_max, sigma_min, num_steps)

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        sigmas = torch.linspace(
            self.sigma_max,
            self.sigma_min,
            self.num_steps,
            device=device,
            dtype=dtype_compute,
        )

        return sigmas


class CosineScheduler(NoiseScheduler):
    """Cosine schedule."""

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
        s: float = 0.008,
        **kwargs,
    ):
        super().__init__(sigma_max, sigma_min, num_steps)
        self.s = s  # small offset to prevent singularity

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        # Parameterize the cosine schedule over the sigma range we actually want,
        # so the schedule stays descending between sigma_max and sigma_min.
        theta_max = math.atan(self.sigma_max)
        theta_min = math.atan(self.sigma_min)
        t_max = (2 * theta_max / math.pi) * (1 + self.s) - self.s
        t_min = (2 * theta_min / math.pi) * (1 + self.s) - self.s

        t = torch.linspace(t_max, t_min, self.num_steps, device=device, dtype=dtype_compute)
        alpha_bar = torch.cos((t + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        sigmas = torch.sqrt((1 - alpha_bar) / alpha_bar)

        return sigmas


class ExponentialScheduler(NoiseScheduler):
    """Exponential schedule (linear in log space)."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, **kwargs):
        super().__init__(sigma_max, sigma_min, num_steps)

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        log_sigmas = torch.linspace(
            torch.log(torch.tensor(self.sigma_max, dtype=dtype_compute)),
            torch.log(torch.tensor(self.sigma_min, dtype=dtype_compute)),
            self.num_steps,
            device=device,
            dtype=dtype_compute,
        )
        sigmas = torch.exp(log_sigmas)

        return sigmas


def _karras_segment(
    sigma_start: float,
    sigma_end: float,
    num_points: int,
    *,
    rho: float,
    device: torch.device = None,
    dtype_compute: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a positive sigma segment including both endpoints."""
    if num_points <= 1:
        return torch.tensor([sigma_start], device=device, dtype=dtype_compute)

    step_indices = torch.arange(num_points, device=device, dtype=dtype_compute)
    return (
        sigma_start ** (1.0 / rho)
        + step_indices / (num_points - 1.0) * (sigma_end ** (1.0 / rho) - sigma_start ** (1.0 / rho))
    ) ** rho


def _exponential_segment(
    sigma_start: float,
    sigma_end: float,
    num_points: int,
    *,
    device: torch.device = None,
    dtype_compute: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a positive sigma segment including both endpoints."""
    if num_points <= 1:
        return torch.tensor([sigma_start], device=device, dtype=dtype_compute)

    log_sigmas = torch.linspace(
        torch.log(torch.tensor(sigma_start, device=device, dtype=dtype_compute)),
        torch.log(torch.tensor(sigma_end, device=device, dtype=dtype_compute)),
        num_points,
        device=device,
        dtype=dtype_compute,
    )
    return torch.exp(log_sigmas)


def _build_segment(
    schedule_type: str,
    sigma_start: float,
    sigma_end: float,
    num_points: int,
    *,
    rho: float = 7.0,
    device: torch.device = None,
    dtype_compute: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a positive sigma segment for a supported schedule type."""
    if schedule_type == "karras":
        return _karras_segment(
            sigma_start,
            sigma_end,
            num_points,
            rho=rho,
            device=device,
            dtype_compute=dtype_compute,
        )
    if schedule_type == "exponential":
        return _exponential_segment(
            sigma_start,
            sigma_end,
            num_points,
            device=device,
            dtype_compute=dtype_compute,
        )
    raise ValueError(f"Unsupported schedule_type for piecewise segment: {schedule_type}")


class ExperimentalSamplerScheduler(NoiseScheduler):
    """Piecewise scheduler for aggressive high-sigma collapse and denser low-sigma refinement.

    The schedule is split into two segments:
    - high segment: typically exponential from ``sigma_max`` to ``sigma_transition``
    - low segment: typically Karras/EDM from ``sigma_transition`` to ``sigma_min``, then terminal zero

    This is exposed through the noise-scheduler registry under ``experimental_sampler`` to match
    the existing inference config plumbing, even though it is semantically a scheduler.
    """

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
        num_steps: int,
        sigma_transition: float = 10.0,
        high_schedule_type: str = "exponential",
        low_schedule_type: str = "karras",
        num_steps_high: int | None = None,
        num_steps_low: int | None = None,
        rho: float = 7.0,
        rho_high: float | None = None,
        rho_low: float | None = None,
        **kwargs,
    ):
        super().__init__(sigma_max, sigma_min, num_steps)

        if sigma_transition <= sigma_min:
            raise ValueError(
                f"sigma_transition must be greater than sigma_min, got {sigma_transition} <= {sigma_min}"
            )
        if sigma_transition >= sigma_max:
            raise ValueError(
                f"sigma_transition must be smaller than sigma_max, got {sigma_transition} >= {sigma_max}"
            )

        default_low = num_steps // 2
        default_high = num_steps - default_low
        self.num_steps_high = default_high if num_steps_high is None else num_steps_high
        self.num_steps_low = default_low if num_steps_low is None else num_steps_low

        if self.num_steps_high < 1 or self.num_steps_low < 1:
            raise ValueError(
                f"num_steps_high and num_steps_low must both be >= 1, got {self.num_steps_high}, {self.num_steps_low}"
            )
        if self.num_steps_high + self.num_steps_low != num_steps:
            raise ValueError(
                "num_steps_high + num_steps_low must equal num_steps, got "
                f"{self.num_steps_high} + {self.num_steps_low} != {num_steps}"
            )

        self.sigma_transition = sigma_transition
        self.high_schedule_type = high_schedule_type
        self.low_schedule_type = low_schedule_type
        self.rho_high = rho if rho_high is None else rho_high
        self.rho_low = rho if rho_low is None else rho_low

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        high = _build_segment(
            self.high_schedule_type,
            self.sigma_max,
            self.sigma_transition,
            self.num_steps_high + 1,
            rho=self.rho_high,
            device=device,
            dtype_compute=dtype_compute,
        )
        low = _build_segment(
            self.low_schedule_type,
            self.sigma_transition,
            self.sigma_min,
            self.num_steps_low,
            rho=self.rho_low,
            device=device,
            dtype_compute=dtype_compute,
        )
        return torch.cat([high, low[1:], torch.zeros(1, device=device, dtype=dtype_compute)])


class DiffusionSampler(ABC):
    """Base class for diffusion samplers."""

    @abstractmethod
    def sample(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Perform diffusion sampling.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input conditioning data with shape (batch, time, ensemble, grid, vars)
        y : dict[str, torch.Tensor]
            Initial noise tensor with shape (batch, time, ensemble, grid, vars)
        sigmas : torch.Tensor
            Noise schedule with shape (num_steps + 1,). The final value is
            expected to be exact zero after NoiseScheduler finalization.
        denoising_fn : Callable
            Function that performs denoising
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : dict[str, Optional[list]]
            Grid shard shapes for distributed processing
        **kwargs
            Additional sampler-specific parameters

        Returns
        -------
        torch.Tensor
            Sampled output with shape (batch, time, ensemble, grid, vars)
        """
        pass


class EDMHeunSampler(DiffusionSampler):
    """EDM Heun sampler with stochastic churn following Karras et al."""

    def __init__(
        self,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        dtype: torch.dtype = torch.float64,
        eps_prec: float = 0.0,
        **kwargs,
    ):
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.dtype = dtype
        self.eps_prec = eps_prec

    def sample(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Override instance defaults with any kwargs
        S_churn = kwargs.get("S_churn", self.S_churn)
        S_min = kwargs.get("S_min", self.S_min)
        S_max = kwargs.get("S_max", self.S_max)
        S_noise = kwargs.get("S_noise", self.S_noise)
        dtype = kwargs.get("dtype", self.dtype)
        eps_prec = kwargs.get("eps_prec", self.eps_prec)
        step_callback = kwargs.get("step_callback")
        sigmas = sigmas.to(dtype)

        num_steps = len(sigmas) - 1
        # Persistent dtype-precision solver state; all Heun update arithmetic uses this buffer.
        y_solver = {dataset_name: y_data.to(dtype) for dataset_name, y_data in y.items()}

        # Heun sampling loop
        for i in range(num_steps):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]

            apply_churn = S_min <= sigma_i <= S_max and S_churn > 0.0
            if apply_churn:
                gamma = min(
                    S_churn / num_steps,
                    torch.sqrt(torch.tensor(2.0, dtype=sigma_i.dtype)) - 1,
                )
                sigma_effective = sigma_i + gamma * sigma_i

                for dataset_name in y_solver:
                    epsilon = torch.randn_like(y_solver[dataset_name]) * S_noise
                    y_solver[dataset_name] = (
                        y_solver[dataset_name] + torch.sqrt(sigma_effective**2 - sigma_i**2) * epsilon
                    )
            else:
                sigma_effective = sigma_i

            # Cast for model evaluation: run denoiser in model/input dtype.
            y_model = {dataset_name: y_data.to(x[dataset_name].dtype) for dataset_name, y_data in y_solver.items()}

            sigma_effective_expanded = _expand_sigma(sigma_effective, y_model)

            D1 = denoising_fn(
                x,
                y_model,
                sigma_effective_expanded,
                model_comm_group,
                grid_shard_shapes,
            )
            D1_solver = {dataset_name: den.to(dtype) for dataset_name, den in D1.items()}

            # Predictor state in solver precision; for Heun corrector evaluation.
            update_direction, y_next_solver = {}, {}
            for dataset_name in y_solver:
                update_direction[dataset_name] = (y_solver[dataset_name] - D1_solver[dataset_name]) / (
                    sigma_effective + eps_prec
                )
                y_next_solver[dataset_name] = (
                    y_solver[dataset_name] + (sigma_next - sigma_effective) * update_direction[dataset_name]
                )

            if sigma_next != 0:
                y_next_model = {
                    # Second denoiser call also runs in model/input dtype (Heun corrector stage).
                    dataset_name: y_next_data.to(x[dataset_name].dtype)
                    for dataset_name, y_next_data in y_next_solver.items()
                }
                sigma_next_expanded = _expand_sigma(sigma_next, y_next_model)

                D2 = denoising_fn(
                    x,
                    y_next_model,
                    sigma_next_expanded,
                    model_comm_group,
                    grid_shard_shapes,
                )
                D2_solver = {dataset_name: den.to(dtype) for dataset_name, den in D2.items()}

                for dataset_name in y_solver:
                    corrected_update_direction = (y_next_solver[dataset_name] - D2_solver[dataset_name]) / (
                        sigma_next + eps_prec
                    )
                    y_solver[dataset_name] = (
                        y_solver[dataset_name]
                        + (sigma_next - sigma_effective)
                        * (update_direction[dataset_name] + corrected_update_direction)
                        / 2
                    )
            else:
                y_solver = y_next_solver

            if step_callback is not None:
                step_callback(i, {dataset_name: y_data.to(x[dataset_name].dtype) for dataset_name, y_data in y_solver.items()})

        return {dataset_name: y_data.to(x[dataset_name].dtype) for dataset_name, y_data in y_solver.items()}


class DPMpp2MSampler(DiffusionSampler):
    """DPM++ 2M sampler (DPM-Solver++ with 2nd order multistep)."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ):
        self.dtype = dtype
        pass  # No parameters needed for DPM++ 2M

    def sample(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        dtype = kwargs.get("dtype", self.dtype)
        step_callback = kwargs.get("step_callback")

        # Keep model evaluations in model dtype, but run solver updates in sampler dtype.
        for dataset_name in y:
            y[dataset_name] = y[dataset_name].to(x[dataset_name].dtype)
        sigmas = sigmas.to(dtype)

        num_steps = len(sigmas) - 1

        # Storage for previous denoised predictions
        old_denoised = None

        # DPM++ 2M sampling loop
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_expanded = _expand_sigma(sigma, y)
            denoised = denoising_fn(x, y, sigma_expanded, model_comm_group, grid_shard_shapes)
            denoised_solver = {dataset_name: den.to(dtype) for dataset_name, den in denoised.items()}

            if sigma_next == 0:
                y = {dataset_name: den.to(x[dataset_name].dtype) for dataset_name, den in denoised_solver.items()}
                if step_callback is not None:
                    step_callback(i, y)
                break

            y_solver = {dataset_name: y_data.to(dtype) for dataset_name, y_data in y.items()}
            t = -torch.log(sigma + 1e-10)
            t_next = -torch.log(sigma_next + 1e-10) if sigma_next != 0 else float("inf")
            h = t_next - t

            if old_denoised is None:
                for dataset_name in y:
                    y_solver[dataset_name] = (sigma_next / sigma) * y_solver[dataset_name] - (
                        torch.exp(-h) - 1
                    ) * denoised_solver[dataset_name]
            else:
                # Second order multistep
                h_last = -torch.log(sigmas[i - 1] + 1e-10) - t if i > 0 else h
                r = h_last / h

                coeff1 = 1 + 1 / (2 * r)
                coeff2 = -1 / (2 * r)

                for dataset_name in y:
                    D = coeff1 * denoised_solver[dataset_name] + coeff2 * old_denoised[dataset_name]
                    y_solver[dataset_name] = (sigma_next / sigma) * y_solver[dataset_name] - (torch.exp(-h) - 1) * D

            old_denoised = denoised_solver
            y = {dataset_name: y_data.to(x[dataset_name].dtype) for dataset_name, y_data in y_solver.items()}

            if step_callback is not None:
                step_callback(i, y)

        return y


# Registry mappings for string-based selection
NOISE_SCHEDULERS = {
    "karras": KarrasScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "exponential": ExponentialScheduler,
    "experimental_sampler": ExperimentalSamplerScheduler,
    "experimental_piecewise": ExperimentalSamplerScheduler,
}

DIFFUSION_SAMPLERS = {
    "heun": EDMHeunSampler,
    "dpmpp_2m": DPMpp2MSampler,
}
