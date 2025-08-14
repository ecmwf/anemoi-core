# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

DenoisingFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[ProcessGroup], Optional[list]], torch.Tensor
]


class NoiseScheduler(ABC):
    """Base class for noise schedulers."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_steps = num_steps

    @abstractmethod
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
        pass


class KarrasScheduler(NoiseScheduler):
    """Karras et al. EDM schedule."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, rho: float = 7.0, **kwargs):
        super().__init__(sigma_max, sigma_min, num_steps)
        self.rho = rho

    def get_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        step_indices = torch.arange(self.num_steps, device=device, dtype=dtype_compute)
        sigmas = (
            self.sigma_max ** (1.0 / self.rho)
            + step_indices
            / (self.num_steps - 1.0)
            * (self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho))
        ) ** self.rho

        return sigmas


class LinearScheduler(NoiseScheduler):
    """Linear schedule in sigma space."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, **kwargs):
        super().__init__(sigma_max, sigma_min, num_steps)

    def get_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_steps, device=device, dtype=dtype_compute)

        return sigmas


class CosineScheduler(NoiseScheduler):
    """Cosine schedule."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, s: float = 0.008, **kwargs):
        super().__init__(sigma_max, sigma_min, num_steps)
        self.s = s  # small offset to prevent singularity

    def get_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        t = torch.linspace(0, 1, self.num_steps, device=device, dtype=dtype_compute)
        alpha_bar = torch.cos((t + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        sigmas = torch.sqrt((1 - alpha_bar) / alpha_bar) * self.sigma_max
        sigmas = torch.clamp(sigmas, min=self.sigma_min, max=self.sigma_max)

        return sigmas


class ExponentialScheduler(NoiseScheduler):
    """Exponential schedule (linear in log space)."""

    def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, **kwargs):
        super().__init__(sigma_max, sigma_min, num_steps)

    def get_schedule(
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


class DiffusionSampler(ABC):
    """Base class for diffusion samplers."""

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Perform diffusion sampling.

        Parameters
        ----------
        x : torch.Tensor
            Input conditioning data with shape (batch, time, ensemble, grid, vars)
        y : torch.Tensor
            Initial noise tensor with shape (batch, ensemble, grid, vars)
        sigmas : torch.Tensor
            Noise schedule with shape (num_steps + 1,)
        denoising_fn : Callable
            Function that performs denoising
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[list]
            Grid shard shapes for distributed processing
        **kwargs
            Additional sampler-specific parameters

        Returns
        -------
        torch.Tensor
            Sampled output with shape (batch, ensemble, grid, vars)
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
        eps_prec: float = 1e-10,
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
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Override instance defaults with any kwargs
        S_churn = kwargs.get("S_churn", self.S_churn)
        S_min = kwargs.get("S_min", self.S_min)
        S_max = kwargs.get("S_max", self.S_max)
        S_noise = kwargs.get("S_noise", self.S_noise)
        dtype = kwargs.get("dtype", self.dtype)
        eps_prec = kwargs.get("eps_prec", self.eps_prec)

        batch_size, ensemble_size = x.shape[0], x.shape[2]
        num_steps = len(sigmas) - 1

        # Heun sampling loop
        for i in range(num_steps):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]

            apply_churn = S_min <= sigma_i <= S_max and S_churn > 0.0
            if apply_churn:
                gamma = min(S_churn / num_steps, torch.sqrt(torch.tensor(2.0, dtype=sigma_i.dtype)) - 1)
                sigma_effective = sigma_i + gamma * sigma_i
                epsilon = torch.randn_like(y) * S_noise
                y = y + torch.sqrt(sigma_effective**2 - sigma_i**2) * epsilon
            else:
                sigma_effective = sigma_i

            D1 = denoising_fn(
                x,
                y.to(dtype=x.dtype),
                sigma_effective.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1).to(x.dtype),
                model_comm_group,
                grid_shard_shapes,
            ).to(dtype)

            d = (y - D1) / (sigma_effective + eps_prec)

            y_next = y + (sigma_next - sigma_effective) * d

            if sigma_next > eps_prec:
                D2 = denoising_fn(
                    x,
                    y_next.to(dtype=x.dtype),
                    sigma_next.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1).to(dtype=x.dtype),
                    model_comm_group,
                    grid_shard_shapes,
                ).to(dtype)
                d_prime = (y_next - D2) / (sigma_next + eps_prec)
                y = y + (sigma_next - sigma_effective) * (d + d_prime) / 2
            else:
                y = y_next

        return y


class DPMpp2MSampler(DiffusionSampler):
    """DPM++ 2M sampler (DPM-Solver++ with 2nd order multistep)."""

    def __init__(self, **kwargs):
        pass  # No parameters needed for DPM++ 2M

    def sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        # DPM++ sampler converts to x.dtype
        y = y.to(x.dtype)
        sigmas = sigmas.to(x.dtype)

        batch_size, ensemble_size = x.shape[0], x.shape[2]
        num_steps = len(sigmas) - 1

        # Storage for previous denoised predictions
        old_denoised = None

        # DPM++ 2M sampling loop
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_expanded = sigma.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
            denoised = denoising_fn(x, y, sigma_expanded, model_comm_group, grid_shard_shapes)

            if sigma_next == 0:
                y = denoised
                break

            t = -torch.log(sigma + 1e-10)
            t_next = -torch.log(sigma_next + 1e-10) if sigma_next > 0 else float("inf")
            h = t_next - t

            if old_denoised is None:
                x0 = denoised
                y = (sigma_next / sigma) * y - (torch.exp(-h) - 1) * x0
            else:
                # Second order multistep
                h_last = -torch.log(sigmas[i - 1] + 1e-10) - t if i > 0 else h
                r = h_last / h

                x0 = denoised
                x0_last = old_denoised

                coeff1 = 1 + 1 / (2 * r)
                coeff2 = -1 / (2 * r)

                D = coeff1 * x0 + coeff2 * x0_last
                y = (sigma_next / sigma) * y - (torch.exp(-h) - 1) * D

            old_denoised = denoised

        return y


# Registry mappings for string-based selection
NOISE_SCHEDULERS = {
    "karras": KarrasScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "exponential": ExponentialScheduler,
}

DIFFUSION_SAMPLERS = {
    "heun": EDMHeunSampler,
    "dpmpp_2m": DPMpp2MSampler,
}
