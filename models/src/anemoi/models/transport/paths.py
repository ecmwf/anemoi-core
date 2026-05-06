# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch

STOCHASTIC_INTERPOLANT_ENDPOINT_EPS = 1e-4


def karras_sigma_from_unit_time(
    t: torch.Tensor,
    *,
    sigma_max: float,
    sigma_min: float,
    rho: float,
) -> torch.Tensor:
    """Map a value between 0 and 1 to the Karras EDM noise schedule."""
    sigma_max_inv_rho = sigma_max ** (1.0 / rho)
    sigma_min_inv_rho = sigma_min ** (1.0 / rho)
    return (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** rho


def edm_loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """Return the EDM loss weight for a given noise level."""
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


def unit_time_grid(
    num_steps: int,
    *,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build the time grid used by deterministic and stochastic transport samplers."""
    if num_steps < 1:
        raise ValueError("Transport num_steps must be at least 1.")
    return torch.linspace(0.0, 1.0, int(num_steps) + 1, device=device, dtype=dtype)


def stochastic_interpolant_alpha(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    """Weight applied to the source field along the stochastic-interpolant bridge."""
    if schedule != "linear":
        raise ValueError(f"Unsupported stochastic interpolant alpha schedule: {schedule}")
    return 1.0 - t


def stochastic_interpolant_beta(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    """Weight applied to the target field along the stochastic-interpolant bridge."""
    if schedule == "linear":
        return t
    if schedule == "quadratic":
        return t.square()
    raise ValueError(f"Unsupported stochastic interpolant beta schedule: {schedule}")


def stochastic_interpolant_sigma(
    t: torch.Tensor, *, schedule: str = "brownian_bridge", noise_scale: float = 1.0
) -> torch.Tensor:
    """Noise amplitude used by the stochastic-interpolant bridge and sampler."""
    if schedule == "brownian_bridge":
        variance = torch.clamp(2.0 * t * (1.0 - t), min=0.0)
        return noise_scale * torch.sqrt(variance)
    if schedule == "quadratic_bridge":
        return noise_scale * t * (1.0 - t)
    raise ValueError(f"Unsupported stochastic interpolant sigma schedule: {schedule}")


def stochastic_interpolant_alpha_dot(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    """Rate of change of the source weight with interpolation time."""
    if schedule != "linear":
        raise ValueError(f"Unsupported stochastic interpolant alpha schedule: {schedule}")
    return -torch.ones_like(t)


def stochastic_interpolant_beta_dot(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    """Rate of change of the target weight with interpolation time."""
    if schedule == "linear":
        return torch.ones_like(t)
    if schedule == "quadratic":
        return 2.0 * t
    raise ValueError(f"Unsupported stochastic interpolant beta schedule: {schedule}")


def stochastic_interpolant_sigma_dot(
    t: torch.Tensor,
    *,
    schedule: str = "brownian_bridge",
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Rate of change of the bridge-noise amplitude with interpolation time."""
    if schedule == "brownian_bridge":
        variance = 2.0 * t * (1.0 - t)
        denom = torch.sqrt(torch.clamp(variance, min=STOCHASTIC_INTERPOLANT_ENDPOINT_EPS))
        return noise_scale * (1.0 - 2.0 * t) / denom
    if schedule == "quadratic_bridge":
        return noise_scale * (1.0 - 2.0 * t)
    raise ValueError(f"Unsupported stochastic interpolant sigma schedule: {schedule}")


def stochastic_interpolant_clean_mean(
    *,
    drift: torch.Tensor,
    interpolant: torch.Tensor,
    anchor: torch.Tensor,
    t: torch.Tensor,
    alpha_schedule: str = "linear",
    beta_schedule: str = "linear",
    sigma_schedule: str = "brownian_bridge",
    noise_scale: float = 1.0,
    reconstruction_eps: float = 1e-8,
) -> torch.Tensor:
    """Recover the clean target from the current bridge state and model prediction."""
    assert all(
        x.dtype in (torch.float32, torch.float64) for x in (drift, interpolant, anchor, t)
    ), "SI clean reconstruction requires at least float32 precision."
    alpha = stochastic_interpolant_alpha(t, alpha_schedule)
    beta = stochastic_interpolant_beta(t, beta_schedule)
    sigma = stochastic_interpolant_sigma(t, schedule=sigma_schedule, noise_scale=noise_scale)
    alpha_dot = stochastic_interpolant_alpha_dot(t, alpha_schedule)
    beta_dot = stochastic_interpolant_beta_dot(t, beta_schedule)
    sigma_dot = stochastic_interpolant_sigma_dot(t, schedule=sigma_schedule, noise_scale=noise_scale)
    if noise_scale == 0.0:
        use_drift = beta_dot.abs() > reconstruction_eps
        use_interpolant = beta.abs() > reconstruction_eps

        clean_from_drift = (drift - alpha_dot * anchor) / torch.clamp(beta_dot, min=reconstruction_eps)
        clean_from_interpolant = (interpolant - alpha * anchor) / torch.clamp(beta, min=reconstruction_eps)

        fallback_clean = torch.where(use_interpolant, clean_from_interpolant, anchor)
        return torch.where(use_drift, clean_from_drift, fallback_clean)

    denom = torch.clamp(beta_dot * sigma - beta * sigma_dot, min=reconstruction_eps)
    return (sigma * (drift - alpha_dot * anchor) - sigma_dot * (interpolant - alpha * anchor)) / denom
