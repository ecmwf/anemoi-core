# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Callable
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup


def get_noise_schedule(
    num_steps: int,
    sigma_max: float,
    sigma_min: float,
    schedule_type: str = "karras",
    rho: float = 7.0,
    device: torch.device = None,
    dtype_compute: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate noise schedule for diffusion sampling.

    Parameters
    ----------
    num_steps : int
        Number of sampling steps
    sigma_max : float
        Maximum noise level
    sigma_min : float
        Minimum noise level
    schedule_type : str
        Type of schedule: "karras", "linear", "cosine", "exponential"
    rho : float
        Time discretization parameter (for Karras schedule)
    device : torch.device
        Device to create tensors on
    dtype_compute : torch.dtype
        Data type for the noise schedule computation (default: torch.float64)

    Returns
    -------
    torch.Tensor
        Noise schedule with shape (num_steps + 1,)
    """
    if schedule_type == "karras":
        # Karras et al. EDM schedule
        step_indices = torch.arange(num_steps, device=device, dtype=dtype_compute)
        sigmas = (
            sigma_max ** (1.0 / rho)
            + step_indices / (num_steps - 1.0) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho

    elif schedule_type == "linear":
        # Linear schedule in sigma space
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps, device=device, dtype=dtype_compute)

    elif schedule_type == "cosine":
        # Cosine schedule
        s = 0.008  # small offset to prevent singularity
        t = torch.linspace(0, 1, num_steps, device=device, dtype=dtype_compute)
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        sigmas = torch.sqrt((1 - alpha_bar) / alpha_bar) * sigma_max
        sigmas = torch.clamp(sigmas, min=sigma_min, max=sigma_max)

    elif schedule_type == "exponential":
        # Exponential schedule (linear in log space)
        log_sigmas = torch.linspace(
            torch.log(torch.tensor(sigma_max, dtype=dtype_compute)),
            torch.log(torch.tensor(sigma_min, dtype=dtype_compute)),
            num_steps,
            device=device,
            dtype=dtype_compute,
        )
        sigmas = torch.exp(log_sigmas)

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Append 0 for the final step
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    return sigmas


def edm_heun_sampler(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    denoising_fn: Callable,
    model_comm_group: Optional[ProcessGroup] = None,
    grid_shard_shapes: Optional[list] = None,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
    dtype: torch.dtype = torch.float64,
    eps_prec: float = 1e-10,
) -> torch.Tensor:
    """EDM Heun sampler with stochastic churn following Karras et al.

    Parameters
    ----------
    x : torch.Tensor
        Input conditioning data with shape (batch, time, ensemble, grid, vars)
    y : torch.Tensor
        Initial noise tensor with shape (batch, ensemble, grid, vars)
    sigmas : torch.Tensor
        Noise schedule with shape (num_steps + 1,)
    denoising_fn : Callable
        Function that performs denoising: denoising_fn(x, y, sigma, model_comm_group) -> denoised
    model_comm_group : Optional[ProcessGroup]
        Process group for distributed training
    S_churn : float
        Stochastic churn amount
    S_min : float
        Minimum sigma for applying churn
    S_max : float
        Maximum sigma for applying churn
    S_noise : float
        Noise scale multiplier
    dtype : torch.dtype
        Data type for computations (default: torch.float64)
    eps_prec : float
        Small epsilon value for numerical stability (default: 1e-10)

    Returns
    -------
    torch.Tensor
        Sampled output with shape (batch, ensemble, grid, vars)
    """
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


def dpmpp_2m_sampler(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    denoising_fn: Callable,
    model_comm_group: Optional[ProcessGroup] = None,
    grid_shard_shapes: Optional[list] = None,
) -> torch.Tensor:
    """DPM++ 2M sampler (DPM-Solver++ with 2nd order multistep).

    Parameters
    ----------
    x : torch.Tensor
        Input conditioning data with shape (batch, time, ensemble, grid, vars)
    y : torch.Tensor
        Initial noise tensor with shape (batch, ensemble, grid, vars)
    sigmas : torch.Tensor
        Noise schedule with shape (num_steps + 1,)
    denoising_fn : Callable
        Function that performs denoising: denoising_fn(x, y, sigma, model_comm_group) -> denoised
    model_comm_group : Optional[ProcessGroup]
        Process group for distributed training

    Returns
    -------
    torch.Tensor
        Sampled output with shape (batch, ensemble, grid, vars)
    """
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
