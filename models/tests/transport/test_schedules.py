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

from anemoi.models.transport.schedules import KarrasSigmaTrainingDistribution
from anemoi.models.transport.schedules import PiecewiseSigmaSchedule
from anemoi.models.transport.schedules import UniformTimeTrainingDistribution
from anemoi.models.transport.schedules import UnitTimeSchedule


def test_unit_time_schedule_returns_monotonic_time_grid() -> None:
    times = UnitTimeSchedule(num_steps=4).get_schedule(dtype_compute=torch.float64)

    torch.testing.assert_close(times, torch.linspace(0.0, 1.0, 5, dtype=torch.float64))


def test_unit_time_schedule_validates_num_steps() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        UnitTimeSchedule(num_steps=0)


def test_time_schedule_rejects_sigma_parameters() -> None:
    with pytest.raises(TypeError):
        UnitTimeSchedule(num_steps=4, sigma_max=2.0)

    with pytest.raises(TypeError):
        UniformTimeTrainingDistribution(sigma_max=2.0)


def test_piecewise_sigma_schedule_has_one_transition_and_terminal_zero() -> None:
    schedule = PiecewiseSigmaSchedule(
        sigma_max=100.0,
        sigma_min=0.02,
        sigma_transition=10.0,
        num_steps=8,
        num_steps_high=5,
        num_steps_low=3,
    ).get_schedule(dtype_compute=torch.float64)

    assert schedule.shape == (9,)
    assert schedule[-1].item() == 0.0
    assert torch.count_nonzero(schedule == 10.0).item() == 1
    assert torch.all(schedule[:-1] > 0)
    assert torch.all(schedule[1:] <= schedule[:-1])


def test_piecewise_sigma_schedule_supports_one_solver_step() -> None:
    schedule = PiecewiseSigmaSchedule(
        sigma_max=100.0,
        sigma_min=0.02,
        sigma_transition=10.0,
        num_steps=1,
    ).get_schedule()

    torch.testing.assert_close(schedule, torch.tensor([100.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_transition": 0.02}, "sigma_transition"),
        ({"sigma_transition": 100.0}, "sigma_transition"),
        ({"num_steps_high": 1, "num_steps_low": 1}, "must equal"),
        ({"high_schedule_type": "linear"}, "Unsupported"),
    ],
)
def test_piecewise_sigma_schedule_validates_configuration(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PiecewiseSigmaSchedule(sigma_max=100.0, sigma_min=0.02, num_steps=8, **kwargs)


@pytest.mark.parametrize(
    "distribution",
    [
        KarrasSigmaTrainingDistribution(sigma_max=2.0, sigma_min=0.5, rho=7.0),
        UniformTimeTrainingDistribution(),
    ],
)
def test_training_condition_distribution_samples_one_shared_condition_per_sample_and_ensemble(
    distribution,
) -> None:
    shapes = {
        "a": (2, 1, 3, 5, 4),
        "b": (2, 2, 3, 7, 1),
    }

    condition = distribution.sample(shapes, device=torch.device("cpu"), dtype=torch.float64)

    assert set(condition) == set(shapes)
    assert condition["a"].shape == condition["b"].shape == (2, 1, 3, 1, 1)
    assert condition["a"].dtype == condition["b"].dtype == torch.float64
    torch.testing.assert_close(condition["a"], condition["b"])
