# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the TrainingState dataclass."""

from anemoi.training.checkpoint.loading.state import TrainingState


def test_training_state_default_values() -> None:
    state = TrainingState()
    assert state.epoch == 0
    assert state.global_step == 0
    assert state.best_metric is None
    assert state.metrics_history == {}


def test_training_state_from_checkpoint() -> None:
    """Extract training state from a Lightning-format checkpoint dict."""
    checkpoint = {
        "epoch": 42,
        "global_step": 10000,
        "state_dict": {"layer.weight": "..."},
        "optimizer_states": [{"state": {}}],
        "lr_schedulers": [{"last_epoch": 42}],
    }
    state = TrainingState.from_checkpoint(checkpoint)
    assert state.epoch == 42
    assert state.global_step == 10000


def test_training_state_from_checkpoint_missing_keys() -> None:
    """Gracefully handle checkpoints without training state metadata."""
    checkpoint = {"state_dict": {"layer.weight": "..."}}
    state = TrainingState.from_checkpoint(checkpoint)
    assert state.epoch == 0
    assert state.global_step == 0


def test_training_state_to_dict() -> None:
    state = TrainingState(epoch=5, global_step=500)
    d = state.to_dict()
    assert d["epoch"] == 5
    assert d["global_step"] == 500
