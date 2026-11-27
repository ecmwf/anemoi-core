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

from anemoi.models.transport.schedules import SIGMA_TRAINING_DISTRIBUTIONS
from anemoi.models.transport.schedules import KarrasSigmaTrainingDistribution
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


@pytest.mark.parametrize("name", sorted(SIGMA_TRAINING_DISTRIBUTIONS))
def test_sigma_training_distributions_map_unit_time_onto_the_noise_range(name: str) -> None:
    """Unit time 0 is the largest noise level and 1 the smallest; noise falls steadily in between."""
    settings = {"sigma_max": 80.0, "sigma_min": 0.02}
    if name == "karras":
        settings["rho"] = 7.0
    distribution = SIGMA_TRAINING_DISTRIBUTIONS[name](**settings)

    sigma = distribution._sigma_from_unit_time(torch.linspace(0.0, 1.0, 101, dtype=torch.float64))

    assert sigma[0].item() == pytest.approx(80.0)
    assert sigma[-1].item() == pytest.approx(0.02)
    assert torch.all(sigma[1:] < sigma[:-1])


@pytest.mark.parametrize("name", sorted(SIGMA_TRAINING_DISTRIBUTIONS))
def test_sigma_training_distributions_sample_within_the_noise_range(name: str) -> None:
    settings = {"sigma_max": 80.0, "sigma_min": 0.02}
    if name == "karras":
        settings["rho"] = 7.0
    distribution = SIGMA_TRAINING_DISTRIBUTIONS[name](**settings)

    sigma = distribution.sample({"a": (64, 1, 4, 5, 3)}, device=torch.device("cpu"), dtype=torch.float64)["a"]

    assert sigma.shape == (64, 1, 4, 1, 1)
    assert torch.all((sigma >= 0.02 * (1 - 1e-12)) & (sigma <= 80.0 * (1 + 1e-12)))
