"""Gate loading-g1: TrainingState.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

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
