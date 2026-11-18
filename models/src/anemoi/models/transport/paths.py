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


def flow_matching_noisy_target(noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The noisy target at time ``t``: the point at ``t`` on the straight path from the noise to the target."""
    return (1.0 - t) * noise + t * target


def flow_matching_velocity(noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Constant velocity of the straight path from the noise to the target."""
    return target - noise


def flow_matching_clean(noisy_target: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Estimate the target by following ``velocity`` from ``noisy_target`` at time ``t`` to time 1."""
    return noisy_target + (1.0 - t) * velocity
