# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
from unittest.mock import Mock

import pytest
import torch
from pytorch_lightning import Callback
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.connectors.callback_connector import _validate_callbacks_list
from pytorch_lightning.utilities.model_helpers import is_overridden

from anemoi.training.diagnostics.callbacks.dropout_scheduler import DropoutScheduler


class _FakeProcessor:
    """Stands in for RandomSpatialDropout: only dropout_prob and dropout_indices are used."""

    def __init__(self, dropout_prob: float = 0.5, n_indices: int = 3) -> None:
        self.dropout_prob = dropout_prob
        self.dropout_indices = torch.arange(n_indices)


def _make_module(processors: dict | None = None, dataset_name: str = "data") -> Mock:
    """Build a pl_module whose model.pre_processors[ds].processors is a plain dict."""
    if processors is None:
        processors = {"spatial_dropout": _FakeProcessor()}

    # Not spec'd to LightningModule: `model` is set by the task at runtime, so it is
    # not part of the class spec and a spec'd Mock would reject the attribute.
    pl_module = Mock()
    pl_module.model.pre_processors = {dataset_name: Mock(processors=processors)}
    pl_module.logger_enabled = False
    return pl_module


def _make_trainer(global_step: int = 0) -> Mock:
    trainer = Mock(Trainer)
    trainer.global_step = global_step
    return trainer


# ── construction validation ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start_prob": 1.5}, "start_prob"),
        ({"end_prob": -0.1}, "end_prob"),
        ({"total_steps": 0}, "total_steps"),
        ({"schedule": "quadratic"}, "Unknown schedule"),
        ({"step_milestones": [0.3, 1.0]}, "strictly within"),
        ({"step_milestones": [0.0, 0.5]}, "strictly within"),
    ],
)
def test_rejects_invalid_config(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DropoutScheduler(**kwargs)


def test_default_milestones_are_all_reachable() -> None:
    """A milestone at 1.0 would never be applied, so the default must exclude it."""
    scheduler = DropoutScheduler(schedule="step")
    assert all(0.0 < milestone < 1.0 for milestone in scheduler.step_milestones)


# ── schedule maths ────────────────────────────────────────────────────────


@pytest.mark.parametrize("schedule", ["cosine", "linear", "step"])
def test_schedule_endpoints(schedule: str) -> None:
    scheduler = DropoutScheduler(start_prob=0.9, end_prob=0.2, total_steps=1000, schedule=schedule)
    assert scheduler._compute_dropout(0) == pytest.approx(0.9)
    assert scheduler._compute_dropout(1000) == pytest.approx(0.2)
    assert scheduler._compute_dropout(5000) == pytest.approx(0.2)  # clamped past the end


@pytest.mark.parametrize("schedule", ["cosine", "linear", "step"])
def test_schedule_is_monotonically_decreasing(schedule: str) -> None:
    scheduler = DropoutScheduler(start_prob=0.9, end_prob=0.2, total_steps=1000, schedule=schedule)
    probs = [scheduler._compute_dropout(step) for step in range(0, 1100, 25)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(probs))
    assert all(0.2 <= prob <= 0.9 for prob in probs)


def test_linear_schedule_midpoint() -> None:
    scheduler = DropoutScheduler(start_prob=1.0, end_prob=0.0, total_steps=1000, schedule="linear")
    assert scheduler._compute_dropout(500) == pytest.approx(0.5)


def test_step_schedule_stages() -> None:
    """Each milestone drops one equal stage, landing on end_prob at the last."""
    scheduler = DropoutScheduler(
        start_prob=0.9,
        end_prob=0.3,
        total_steps=1000,
        schedule="step",
        step_milestones=[0.3, 0.6],
    )
    assert scheduler._compute_dropout(0) == pytest.approx(0.9)
    assert scheduler._compute_dropout(299) == pytest.approx(0.9)
    assert scheduler._compute_dropout(300) == pytest.approx(0.6)  # first milestone
    assert scheduler._compute_dropout(599) == pytest.approx(0.6)
    assert scheduler._compute_dropout(600) == pytest.approx(0.3)  # second, reaches end_prob


def test_step_schedule_reaches_end_prob_before_total_steps() -> None:
    """Regression: the final stage must be applied via a milestone, not the clamp."""
    scheduler = DropoutScheduler(
        start_prob=0.8,
        end_prob=0.1,
        total_steps=1000,
        schedule="step",
        step_milestones=[0.5],
    )
    assert scheduler._compute_dropout(999) == pytest.approx(0.1)


# ── processor resolution ──────────────────────────────────────────────────


def test_raises_on_missing_dataset() -> None:
    scheduler = DropoutScheduler(dataset_name="absent")
    with pytest.raises(ValueError, match="dataset 'absent' not found"):
        scheduler.setup(_make_trainer(), _make_module(), stage="fit")


def test_raises_on_missing_processor() -> None:
    scheduler = DropoutScheduler(processor_name="typo_dropout")
    with pytest.raises(ValueError, match="processor 'typo_dropout' not found"):
        scheduler.setup(_make_trainer(), _make_module(), stage="fit")


def test_raises_on_processor_with_no_variables() -> None:
    """dropout_prob: 0.0 in the data config leaves dropout_indices empty."""
    pl_module = _make_module({"spatial_dropout": _FakeProcessor(n_indices=0)})
    scheduler = DropoutScheduler()
    with pytest.raises(ValueError, match="no variables to drop"):
        scheduler.setup(_make_trainer(), pl_module, stage="fit")


def test_setup_ignores_non_fit_stages() -> None:
    scheduler = DropoutScheduler(processor_name="typo_dropout")
    scheduler.setup(_make_trainer(), _make_module(), stage="test")  # must not raise
    assert scheduler._processor is None


# ── applying the schedule ─────────────────────────────────────────────────


def test_on_train_start_applies_initial_prob() -> None:
    processor = _FakeProcessor(dropout_prob=0.0)
    pl_module = _make_module({"spatial_dropout": processor})
    scheduler = DropoutScheduler(start_prob=0.9, end_prob=0.2, total_steps=1000)

    scheduler.on_train_start(_make_trainer(global_step=0), pl_module)

    assert processor.dropout_prob == pytest.approx(0.9)


def test_on_train_start_resumes_mid_schedule() -> None:
    """Resume is driven purely by global_step, with no callback state involved."""
    processor = _FakeProcessor()
    pl_module = _make_module({"spatial_dropout": processor})
    scheduler = DropoutScheduler(start_prob=1.0, end_prob=0.0, total_steps=1000, schedule="linear")

    scheduler.on_train_start(_make_trainer(global_step=750), pl_module)

    assert processor.dropout_prob == pytest.approx(0.25)


def test_on_train_batch_start_updates_prob() -> None:
    processor = _FakeProcessor()
    pl_module = _make_module({"spatial_dropout": processor})
    scheduler = DropoutScheduler(start_prob=1.0, end_prob=0.0, total_steps=1000, schedule="linear")
    scheduler.setup(_make_trainer(), pl_module, stage="fit")

    scheduler.on_train_batch_start(_make_trainer(global_step=200), pl_module, batch=None, batch_idx=0)
    assert processor.dropout_prob == pytest.approx(0.8)

    scheduler.on_train_batch_start(_make_trainer(global_step=900), pl_module, batch=None, batch_idx=1)
    assert processor.dropout_prob == pytest.approx(0.1)


def test_logs_rate_under_processor_keyed_name() -> None:
    """The two schedulers in a run must not collide on one metric name."""
    pl_module = _make_module({"spatial_dropout2": _FakeProcessor()})
    scheduler = DropoutScheduler(processor_name="spatial_dropout2", start_prob=0.6, total_steps=1000)
    scheduler.setup(_make_trainer(), pl_module, stage="fit")

    scheduler.on_train_batch_start(_make_trainer(global_step=0), pl_module, batch=None, batch_idx=0)

    name, value = pl_module.log.call_args.args
    assert name == "dropout/spatial_dropout2"
    assert value == pytest.approx(0.6)


# ── checkpoint state ──────────────────────────────────────────────────────


def test_holds_no_checkpoint_state() -> None:
    """The schedule must come from the config on resume, not from a stale checkpoint.

    The base Callback.state_dict returns {} and load_state_dict is a no-op, so a
    checkpoint written by the old implementation cannot override the new config.
    """
    scheduler = DropoutScheduler(start_prob=0.4, end_prob=0.1, total_steps=2000, schedule="linear")
    assert scheduler.state_dict() == {}

    stale = {"start_prob": 0.95, "end_prob": 0.3, "total_steps": 250000, "schedule": "step"}
    scheduler.load_state_dict(stale)

    assert scheduler.start_prob == pytest.approx(0.4)
    assert scheduler.end_prob == pytest.approx(0.1)
    assert scheduler.total_steps == 2000
    assert scheduler.schedule == "linear"


def test_two_schedulers_can_coexist_in_one_trainer() -> None:
    """A run schedules spatial_dropout and spatial_dropout2 together.

    Lightning rejects duplicate instances of a *stateful* callback type, since they
    would share a state_key in the checkpoint. Holding no state is what makes two
    instances legal, so this pins that dependency: reintroducing state_dict on this
    callback would break every config that schedules more than one processor.
    """
    schedulers = [
        DropoutScheduler(processor_name="spatial_dropout", start_prob=0.9, end_prob=0.3),
        DropoutScheduler(processor_name="spatial_dropout2", start_prob=0.3, end_prob=0.1),
    ]
    assert not is_overridden("state_dict", instance=schedulers[0], parent=Callback)
    _validate_callbacks_list(schedulers)  # must not raise
