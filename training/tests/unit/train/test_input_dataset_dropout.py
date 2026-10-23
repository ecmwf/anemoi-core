# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""training.input_dataset_dropout: leaving a first_step_only input dataset out of some training samples."""

from types import SimpleNamespace

import pytest

from anemoi.training.train.methods.ensemble import EnsembleTraining

validated = EnsembleTraining._validated_input_dataset_dropout


def _dropped(dropout: dict[str, float], global_step: int) -> tuple[str, ...]:
    module = SimpleNamespace(input_dataset_dropout=dropout, global_step=global_step)
    return EnsembleTraining._dropped_input_datasets(module)


def test_no_dropout_by_default() -> None:
    assert validated(None, frozenset()) == {}
    assert _dropped({}, global_step=5) == ()


def test_probabilities_are_read_as_floats() -> None:
    assert validated({"std": 1}, {"std"}) == {"std": 1.0}


def test_only_first_step_only_datasets_can_be_dropped() -> None:
    with pytest.raises(ValueError, match="not a first_step_only input dataset"):
        validated({"era5": 0.5}, {"std"})


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_probability_must_be_in_the_unit_interval(probability: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        validated({"std": probability}, {"std"})


def test_probability_zero_never_drops_and_one_always_does() -> None:
    assert all(_dropped({"std": 0.0}, step) == () for step in range(200))
    assert all(_dropped({"std": 1.0}, step) == ("std",) for step in range(200))


def test_the_draw_depends_only_on_the_global_step(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every rank, and every member of an ensemble spread over GPUs, must make the same choice.
    monkeypatch.setenv("ANEMOI_BASE_SEED", "199")
    first = [_dropped({"std": 0.5}, step) for step in range(200)]
    second = [_dropped({"std": 0.5}, step) for step in range(200)]

    assert first == second


def test_about_the_configured_share_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANEMOI_BASE_SEED", "199")
    share = sum(bool(_dropped({"std": 0.3}, step)) for step in range(4000)) / 4000

    assert share == pytest.approx(0.3, abs=0.03)


def test_the_seed_changes_which_steps_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANEMOI_BASE_SEED", "199")
    seed_199 = [_dropped({"std": 0.5}, step) for step in range(200)]
    monkeypatch.setenv("ANEMOI_BASE_SEED", "132")
    seed_132 = [_dropped({"std": 0.5}, step) for step in range(200)]

    assert seed_199 != seed_132
