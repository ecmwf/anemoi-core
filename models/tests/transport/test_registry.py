# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.samplers.transport_samplers import DIFFUSION_SAMPLERS
from anemoi.models.transport.registry import build_registered
from anemoi.models.transport.schedules import SIGMA_SCHEDULES
from anemoi.models.transport.schedules import SIGMA_TRAINING_DISTRIBUTIONS
from anemoi.models.transport.schedules import LinearSigmaTrainingDistribution

NOISE_RANGE = {"sigma_max": 80.0, "sigma_min": 0.02}


@pytest.mark.parametrize("name", sorted(SIGMA_TRAINING_DISTRIBUTIONS))
def test_every_training_distribution_builds_from_the_noise_range(name: str) -> None:
    distribution = build_registered(SIGMA_TRAINING_DISTRIBUTIONS, name, NOISE_RANGE, "distribution")

    assert distribution.sigma_max == 80.0
    assert distribution.sigma_min == 0.02


@pytest.mark.parametrize("name", sorted(SIGMA_SCHEDULES))
def test_every_sigma_schedule_builds_from_the_noise_range(name: str) -> None:
    schedule = build_registered(SIGMA_SCHEDULES, name, {**NOISE_RANGE, "num_steps": 5}, "schedule")

    assert schedule.get_schedule().shape == (6,)


def test_a_setting_of_another_entry_is_rejected() -> None:
    """The Karras ``rho`` with a linear schedule is an error: the linear schedule would not use it."""
    with pytest.raises(TypeError, match="rho"):
        build_registered(SIGMA_SCHEDULES, "linear", {**NOISE_RANGE, "num_steps": 5, "rho": 7.0}, "schedule")

    with pytest.raises(TypeError, match="S_churn"):
        build_registered(DIFFUSION_SAMPLERS, "dpmpp_2m", {"S_churn": 0.5}, "sampler")


def test_a_setting_no_entry_takes_is_rejected() -> None:
    with pytest.raises(TypeError, match="sigma_mx"):
        build_registered(SIGMA_TRAINING_DISTRIBUTIONS, "linear", {"sigma_max": 1.0, "sigma_mx": 0.1}, "distribution")


def test_an_unknown_name_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="Available: cosine, exponential, karras, linear"):
        build_registered(SIGMA_TRAINING_DISTRIBUTIONS, "karas", NOISE_RANGE, "distribution")


def test_fixed_arguments_are_passed_on() -> None:
    distribution = build_registered(
        SIGMA_TRAINING_DISTRIBUTIONS, "linear", {}, "distribution", sigma_max=2.0, sigma_min=1.0
    )

    assert isinstance(distribution, LinearSigmaTrainingDistribution)
    assert distribution.sigma_max == 2.0
