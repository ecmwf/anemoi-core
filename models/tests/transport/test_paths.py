# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.samplers.transport_samplers import KarrasScheduler
from anemoi.models.transport.paths import karras_sigma_from_unit_time
from anemoi.models.transport.paths import stochastic_interpolant_beta
from anemoi.models.transport.paths import stochastic_interpolant_beta_dot
from anemoi.models.transport.paths import stochastic_interpolant_bridge_noise_velocity_ratio
from anemoi.models.transport.paths import stochastic_interpolant_clean_mean
from anemoi.models.transport.paths import stochastic_interpolant_sigma
from anemoi.models.transport.paths import unit_time_grid


def test_karras_sigma_helper_matches_scheduler_prefinal_values() -> None:
    num_steps = 6
    unit_time = torch.linspace(0.0, 1.0, num_steps, dtype=torch.float64)

    helper_sigmas = karras_sigma_from_unit_time(
        unit_time,
        sigma_max=1.0,
        sigma_min=0.02,
        rho=7.0,
    )
    scheduler_sigmas = KarrasScheduler(
        sigma_max=1.0,
        sigma_min=0.02,
        num_steps=num_steps,
        rho=7.0,
    ).get_schedule(
        dtype_compute=torch.float64
    )[:-1]

    torch.testing.assert_close(helper_sigmas, scheduler_sigmas)


def test_unit_time_grid_validates_num_steps() -> None:
    grid = unit_time_grid(4, dtype=torch.float64)

    torch.testing.assert_close(grid, torch.linspace(0.0, 1.0, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="num_steps"):
        unit_time_grid(0)


def test_stochastic_interpolant_brownian_bridge_ratio_matches_noise_velocity_interior() -> None:
    time = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    noise_scale = 2.0

    sigma = stochastic_interpolant_sigma(time, schedule="brownian_bridge", noise_scale=noise_scale)
    ratio = stochastic_interpolant_bridge_noise_velocity_ratio(time, schedule="brownian_bridge", eps=0.0)
    expected_velocity = noise_scale * (1.0 - 2.0 * time) / torch.sqrt(2.0 * time * (1.0 - time))

    torch.testing.assert_close(ratio * sigma, expected_velocity)


def test_stochastic_interpolant_quadratic_bridge_ratio_matches_noise_velocity_interior() -> None:
    time = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    noise_scale = 2.0

    sigma = stochastic_interpolant_sigma(time, schedule="quadratic_bridge", noise_scale=noise_scale)
    ratio = stochastic_interpolant_bridge_noise_velocity_ratio(time, schedule="quadratic_bridge", eps=0.0)

    torch.testing.assert_close(
        ratio * sigma,
        noise_scale * (1.0 - 2.0 * time),
    )


def test_quadratic_bridge_reconstructs_clean_endpoint_near_source_endpoint() -> None:
    anchor = torch.tensor([[[[[1.0]]]]], dtype=torch.float64)
    clean = torch.tensor([[[[[3.0]]]]], dtype=torch.float64)
    noise = torch.tensor([[[[[2.0]]]]], dtype=torch.float64)
    time_level = torch.full_like(anchor, 1e-4)

    beta = stochastic_interpolant_beta(time_level)
    sigma = stochastic_interpolant_sigma(time_level, schedule="quadratic_bridge")
    interpolant = (1.0 - time_level) * anchor + beta * clean + sigma * noise
    drift_noise = (
        stochastic_interpolant_bridge_noise_velocity_ratio(time_level, schedule="quadratic_bridge", eps=1e-8)
        * sigma
        * noise
    )
    drift = -anchor + clean + drift_noise

    reconstructed = stochastic_interpolant_clean_mean(
        drift=drift,
        interpolant=interpolant,
        anchor=anchor,
        t=time_level,
        sigma_schedule="quadratic_bridge",
        noise_scale=1.0,
    )

    torch.testing.assert_close(reconstructed, clean)


def test_deterministic_stochastic_interpolant_reconstructs_clean_endpoint() -> None:
    anchor = torch.tensor([[[[[1.0, -2.0]]]]])
    clean = torch.tensor([[[[[3.0, 4.0]]]]])
    time_level = torch.tensor([[[[[0.25]]]]])
    interpolant = (1.0 - time_level) * anchor + time_level * clean
    drift = clean - anchor

    reconstructed = stochastic_interpolant_clean_mean(
        drift=drift,
        interpolant=interpolant,
        anchor=anchor,
        t=time_level,
        noise_scale=0.0,
    )

    torch.testing.assert_close(reconstructed, clean)


@pytest.mark.parametrize("beta_schedule", ["linear", "quadratic"])
@pytest.mark.parametrize("sigma_schedule", ["brownian_bridge", "quadratic_bridge"])
def test_stochastic_interpolant_reconstructs_clean_endpoint_with_bridge_noise(
    beta_schedule: str,
    sigma_schedule: str,
) -> None:
    torch.manual_seed(0)
    anchor = torch.randn((2, 3, 1, 4, 5), dtype=torch.float64)
    clean = torch.randn_like(anchor)
    noise = torch.randn_like(anchor)
    time_level = torch.rand((2, 1, 1, 1, 1), dtype=torch.float64).clamp(1e-4, 1.0 - 1e-4)
    noise_scale = 0.7

    alpha = 1.0 - time_level
    beta = stochastic_interpolant_beta(time_level, beta_schedule)
    sigma = stochastic_interpolant_sigma(time_level, schedule=sigma_schedule, noise_scale=noise_scale)
    interpolant = alpha * anchor + beta * clean + sigma * noise

    alpha_dot = -torch.ones_like(time_level)
    beta_dot = stochastic_interpolant_beta_dot(time_level, beta_schedule)
    ratio = stochastic_interpolant_bridge_noise_velocity_ratio(time_level, schedule=sigma_schedule, eps=1e-8)
    drift = alpha_dot * anchor + beta_dot * clean + ratio * sigma * noise

    reconstructed = stochastic_interpolant_clean_mean(
        drift=drift,
        interpolant=interpolant,
        anchor=anchor,
        t=time_level,
        beta_schedule=beta_schedule,
        sigma_schedule=sigma_schedule,
        noise_scale=noise_scale,
    )

    torch.testing.assert_close(reconstructed, clean)
