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

from anemoi.models.transport.paths import flow_matching_clean
from anemoi.models.transport.paths import flow_matching_state
from anemoi.models.transport.paths import flow_matching_velocity
from anemoi.models.transport.paths import karras_sigma_from_unit_time
from anemoi.models.transport.schedules import KarrasSigmaSchedule


def test_karras_sigma_helper_matches_scheduler_prefinal_values() -> None:
    num_steps = 6
    unit_time = torch.linspace(0.0, 1.0, num_steps, dtype=torch.float64)

    helper_sigmas = karras_sigma_from_unit_time(
        unit_time,
        sigma_max=1.0,
        sigma_min=0.02,
        rho=7.0,
    )
    scheduler_sigmas = KarrasSigmaSchedule(
        sigma_max=1.0,
        sigma_min=0.02,
        num_steps=num_steps,
        rho=7.0,
    ).get_schedule(dtype_compute=torch.float64)[:-1]

    torch.testing.assert_close(helper_sigmas, scheduler_sigmas)


def test_flow_matching_path_runs_from_source_to_target() -> None:
    source = torch.tensor([1.0, -2.0], dtype=torch.float64)
    target = torch.tensor([3.0, 4.0], dtype=torch.float64)

    torch.testing.assert_close(flow_matching_state(source, target, torch.tensor(0.0, dtype=torch.float64)), source)
    torch.testing.assert_close(flow_matching_state(source, target, torch.tensor(1.0, dtype=torch.float64)), target)
    torch.testing.assert_close(flow_matching_velocity(source, target), torch.tensor([2.0, 6.0], dtype=torch.float64))


@pytest.mark.parametrize("time", [0.0, 0.25, 0.9])
def test_flow_matching_clean_recovers_the_target_from_the_true_velocity(time: float) -> None:
    source = torch.tensor([1.0, -2.0], dtype=torch.float64)
    target = torch.tensor([3.0, 4.0], dtype=torch.float64)
    t = torch.tensor(time, dtype=torch.float64)

    clean = flow_matching_clean(flow_matching_state(source, target, t), flow_matching_velocity(source, target), t)

    torch.testing.assert_close(clean, target)


def test_flow_matching_clean_error_shrinks_with_time() -> None:
    """An error in the velocity only counts for the part of the path still to go."""
    state = torch.zeros(1, dtype=torch.float64)
    velocity_error = torch.ones(1, dtype=torch.float64)

    error = flow_matching_clean(state, velocity_error, torch.tensor(0.75, dtype=torch.float64))

    torch.testing.assert_close(error, torch.tensor([0.25], dtype=torch.float64))
