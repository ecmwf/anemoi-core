# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity coverage for the checkpoint loading-strategy layer.

These tests close genuine assertion gaps in the loading strategies
(``WeightsOnlyLoader``, ``TransferLearningLoader``, ``WarmStartLoader``,
``ColdStartLoader``), the ``TrainingState`` dataclass, ``filter_state_dict``
and the shared ``apply_trainable_edge_perm_migration`` parity helper.

They exercise the real strategy code paths on small CPU modules with
deterministic synthetic checkpoints, asserting the concrete observable signal
(loaded tensor values, discarded optimizer/scheduler, exact metadata keys and
values, raised exception types) rather than a weaker proxy such as "weights
changed" or "key is present".
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading.base import LoadingStrategy
from anemoi.training.checkpoint.loading.state import TrainingState
from anemoi.training.checkpoint.loading.strategies import ColdStartLoader
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader
from anemoi.training.checkpoint.loading.utils import filter_state_dict


class _LinearModel(nn.Module):
    """Single ``linear`` submodule; state-dict keys ``linear.weight`` / ``linear.bias``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


class _Layer1Model(nn.Module):
    """Single ``layer1`` submodule; state-dict keys ``layer1.weight`` / ``layer1.bias``."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(10, 5)


class _SharedModel(nn.Module):
    """Transfer-learning target: only a ``shared`` layer (no head)."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(10, 5)


def _fake_index_collection(name_to_index: dict[str, int]) -> object:
    """Build a stand-in ``IndexCollection`` exposing ``.name_to_index``."""
    return type("IndexCollection", (), {"name_to_index": name_to_index})()


def _exact_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Deterministic state dict matching ``model``'s keys/shapes (distinct constant per key)."""
    return {key: torch.full_like(value, float(idx + 1)) for idx, (key, value) in enumerate(model.state_dict().items())}


# ---------------------------------------------------------------------------
# WeightsOnlyLoader
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weights_only_strict_loads_exact_checkpoint_tensors() -> None:
    model = _LinearModel()
    weight = torch.arange(50, dtype=torch.float32).reshape(5, 10)
    bias = torch.arange(5, dtype=torch.float32)
    checkpoint_data = {"state_dict": {"linear.weight": weight, "linear.bias": bias}}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WeightsOnlyLoader().process(context)

    assert torch.equal(result.model.linear.weight, weight)
    assert torch.equal(result.model.linear.bias, bias)


@pytest.mark.asyncio
async def test_weights_only_strict_raises_on_extra_checkpoint_key() -> None:
    model = _Layer1Model()
    checkpoint_data = {
        "state_dict": {
            "layer1.weight": torch.full((5, 10), 1.0),
            "layer1.bias": torch.full((5,), 2.0),
            "extra_layer.weight": torch.full((2, 2), 3.0),
        },
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    with pytest.raises(CheckpointLoadError):
        await WeightsOnlyLoader(strict=True).process(context)


@pytest.mark.asyncio
async def test_weights_only_non_strict_tolerates_extra_key() -> None:
    model = _Layer1Model()
    weight = torch.full((5, 10), 4.0)
    bias = torch.full((5,), 5.0)
    checkpoint_data = {
        "state_dict": {
            "layer1.weight": weight,
            "layer1.bias": bias,
            "extra_layer.weight": torch.full((2, 2), 6.0),
        },
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WeightsOnlyLoader(strict=False).process(context)

    # No exception; the model's own keys loaded, extra key ignored.
    assert torch.equal(result.model.layer1.weight, weight)
    assert torch.equal(result.model.layer1.bias, bias)


@pytest.mark.asyncio
async def test_weights_only_non_strict_still_raises_on_shape_mismatch() -> None:
    model = _Layer1Model()  # layer1.weight is (5, 10)
    checkpoint_data = {
        "state_dict": {
            "layer1.weight": torch.zeros(10, 10),  # same key, mismatched shape
            "layer1.bias": torch.zeros(5),
        },
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    with pytest.raises(CheckpointLoadError):
        await WeightsOnlyLoader(strict=False).process(context)


@pytest.mark.asyncio
async def test_weights_only_does_not_write_training_progress_metadata() -> None:
    model = _LinearModel()
    checkpoint_data = {
        "state_dict": _exact_state_dict(model),
        "epoch": 42,
        "global_step": 10000,
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WeightsOnlyLoader().process(context)

    assert "epoch" not in result.metadata
    assert "global_step" not in result.metadata


@pytest.mark.asyncio
async def test_weights_only_populates_variables_metadata() -> None:
    model = _LinearModel()
    checkpoint_data = {
        "state_dict": _exact_state_dict(model),
        "hyper_parameters": {
            "data_indices": {"era5": _fake_index_collection({"2t": 0})},
            "metadata": {"dataset": {"era5": {"variables_metadata": {"2t": {"units": "K"}}}}},
        },
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WeightsOnlyLoader().process(context)

    assert getattr(result.model, "_ckpt_variables_metadata", None) is not None


@pytest.mark.asyncio
async def test_weights_only_marks_weights_initialized() -> None:
    model = _LinearModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": _exact_state_dict(model)})

    result = await WeightsOnlyLoader().process(context)

    assert result.model.weights_initialized is True


# ---------------------------------------------------------------------------
# ColdStartLoader
# ---------------------------------------------------------------------------


def test_cold_start_inherits_strict_parameter() -> None:
    assert ColdStartLoader(strict=False).strict is False
    assert ColdStartLoader().strict is True


@pytest.mark.asyncio
async def test_cold_start_records_pretrained_from_path_value() -> None:
    model = _LinearModel()
    context = CheckpointContext(
        model=model,
        checkpoint_data={"state_dict": _exact_state_dict(model)},
        checkpoint_path="/path/to/pretrained.ckpt",
    )

    result = await ColdStartLoader().process(context)

    assert result.metadata["pretrained_from"] == str(result.checkpoint_path)
    assert result.metadata["pretrained_from"] == "/path/to/pretrained.ckpt"


@pytest.mark.asyncio
async def test_cold_start_records_pretrained_from_none_when_path_unset() -> None:
    model = _LinearModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": _exact_state_dict(model)})

    result = await ColdStartLoader().process(context)

    assert result.metadata["pretrained_from"] is None


@pytest.mark.asyncio
async def test_cold_start_sets_loading_strategy_label() -> None:
    model = _LinearModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": _exact_state_dict(model)})

    result = await ColdStartLoader().process(context)

    assert result.metadata["loading_strategy"] == "cold_start"


@pytest.mark.asyncio
async def test_cold_start_loads_weights_and_clears_optimizer_scheduler() -> None:
    torch.manual_seed(0)
    model = _LinearModel()
    original_weight = model.linear.weight.detach().clone()
    ckpt_state = _exact_state_dict(model)

    context = CheckpointContext(
        model=model,
        checkpoint_data={"state_dict": ckpt_state},
        optimizer=object(),
        scheduler=object(),
    )
    result = await ColdStartLoader().process(context)

    assert not torch.equal(result.model.linear.weight, original_weight)
    assert torch.equal(result.model.linear.weight, ckpt_state["linear.weight"])
    assert result.optimizer is None
    assert result.scheduler is None


# ---------------------------------------------------------------------------
# TransferLearningLoader
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transfer_learning_records_missing_key_reason() -> None:
    target = _SharedModel()
    source_state = {
        "shared.weight": torch.full((5, 10), 1.0),
        "shared.bias": torch.full((5,), 2.0),
        "old_head.weight": torch.full((3, 5), 3.0),  # absent from target
        "old_head.bias": torch.full((3,), 4.0),
    }

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert result.metadata["skipped_params"]["old_head.weight"] == "Key not in target"


@pytest.mark.asyncio
async def test_transfer_learning_records_shape_mismatch_reason() -> None:
    target = _SharedModel()  # shared.weight is (5, 10)
    source_state = {
        "shared.weight": torch.full((7, 10), 1.0),  # same key, different shape
        "shared.bias": torch.full((5,), 2.0),
    }

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert result.metadata["skipped_params"]["shared.weight"].startswith("Shape mismatch")


@pytest.mark.asyncio
async def test_transfer_learning_strict_raises_on_shape_mismatch() -> None:
    target = _SharedModel()  # shared.weight is (5, 10)
    source_state = {
        "shared.weight": torch.full((7, 10), 1.0),  # same key, different shape
        "shared.bias": torch.full((5,), 2.0),
    }

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    with pytest.raises(CheckpointIncompatibleError, match="Shape mismatch"):
        await TransferLearningLoader(skip_mismatched=False).process(context)


@pytest.mark.asyncio
async def test_transfer_learning_strict_does_not_raise_on_missing_key() -> None:
    target = _SharedModel()
    source_state = {
        "shared.weight": torch.full((5, 10), 1.0),  # matches target
        "shared.bias": torch.full((5,), 2.0),
        "old_head.weight": torch.full((3, 5), 3.0),  # absent from target, not a shape mismatch
    }

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await TransferLearningLoader(skip_mismatched=False).process(context)

    assert result.model is target
    assert result.metadata["skipped_params"]["old_head.weight"] == "Key not in target"


@pytest.mark.asyncio
async def test_transfer_learning_skipped_params_maps_keys_to_reason_strings() -> None:
    target = _SharedModel()  # shared.weight is (5, 10)
    source_state = {
        "shared.weight": torch.full((7, 10), 1.0),  # same key, different shape
        "shared.bias": torch.full((5,), 2.0),  # matches
        "old_head.weight": torch.full((3, 5), 3.0),  # absent from target
    }

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    skipped = result.metadata["skipped_params"]
    assert isinstance(skipped, dict)
    assert all(isinstance(reason, str) for reason in skipped.values())
    assert skipped["old_head.weight"] == "Key not in target"
    assert skipped["shared.weight"].startswith("Shape mismatch")


@pytest.mark.asyncio
async def test_transfer_learning_discards_optimizer() -> None:
    target = _SharedModel()
    source_state = {"shared.weight": torch.full((5, 10), 1.0), "shared.bias": torch.full((5,), 2.0)}

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state}, optimizer=object())
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert result.optimizer is None


@pytest.mark.asyncio
async def test_transfer_learning_discards_scheduler() -> None:
    target = _SharedModel()
    source_state = {"shared.weight": torch.full((5, 10), 1.0), "shared.bias": torch.full((5,), 2.0)}

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state}, scheduler=object())
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert result.scheduler is None


@pytest.mark.asyncio
async def test_transfer_learning_sets_loading_strategy_label() -> None:
    target = _SharedModel()
    source_state = {"shared.weight": torch.full((5, 10), 1.0), "shared.bias": torch.full((5,), 2.0)}

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert result.metadata["loading_strategy"] == "transfer_learning"


@pytest.mark.asyncio
async def test_transfer_learning_loads_when_no_keys_match() -> None:
    target = _SharedModel()  # keys: shared.weight, shared.bias
    source_state = {
        "encoder.weight": torch.full((4, 4), 1.0),
        "encoder.bias": torch.full((4,), 2.0),
    }

    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert result.metadata["transferred_params"] == []
    assert set(source_state) <= set(result.metadata["skipped_params"])


# ---------------------------------------------------------------------------
# WarmStartLoader
# ---------------------------------------------------------------------------


def test_warm_start_has_no_strict_parameter() -> None:
    assert "strict" not in inspect.signature(WarmStartLoader.__init__).parameters


@pytest.mark.asyncio
async def test_warm_start_raises_on_unexpected_key() -> None:
    model = _LinearModel()
    checkpoint_data = {
        "state_dict": {
            "linear.weight": torch.full((5, 10), 1.0),
            "linear.bias": torch.full((5,), 2.0),
            "extra.weight": torch.full((2, 2), 3.0),  # unexpected under strict=True
        },
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    with pytest.raises(CheckpointIncompatibleError):
        await WarmStartLoader().process(context)


@pytest.mark.asyncio
async def test_warm_start_surfaces_best_metric_when_present() -> None:
    model = _LinearModel()
    checkpoint_data = {"state_dict": _exact_state_dict(model), "best_metric": 0.95}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert result.metadata["best_metric"] == 0.95


@pytest.mark.asyncio
async def test_warm_start_omits_best_metric_when_absent() -> None:
    model = _LinearModel()
    checkpoint_data = {"state_dict": _exact_state_dict(model)}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert "best_metric" not in result.metadata


@pytest.mark.asyncio
async def test_warm_start_surfaces_metrics_history_when_present() -> None:
    model = _LinearModel()
    metrics_history = {"loss": [1.0, 0.5]}
    checkpoint_data = {"state_dict": _exact_state_dict(model), "metrics_history": metrics_history}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert result.metadata["metrics_history"] == metrics_history


@pytest.mark.asyncio
async def test_warm_start_omits_metrics_history_when_empty() -> None:
    model = _LinearModel()
    checkpoint_data = {"state_dict": _exact_state_dict(model), "metrics_history": {}}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert "metrics_history" not in result.metadata


@pytest.mark.asyncio
async def test_warm_start_leaves_scheduler_none() -> None:
    model = _LinearModel()
    checkpoint_data = {
        "state_dict": _exact_state_dict(model),
        "lr_schedulers": [{"last_epoch": 5}],
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert result.scheduler is None


# ---------------------------------------------------------------------------
# LoadingStrategy base contract
# ---------------------------------------------------------------------------


def test_restores_training_state_defaults_false_for_non_warm_start() -> None:
    assert LoadingStrategy.restores_training_state is False
    assert WeightsOnlyLoader.restores_training_state is False
    assert TransferLearningLoader.restores_training_state is False
    assert ColdStartLoader.restores_training_state is False


# ---------------------------------------------------------------------------
# apply_trainable_edge_perm_migration parity helper
# ---------------------------------------------------------------------------


def test_edge_perm_migration_noop_when_checkpoint_none() -> None:
    model = _LinearModel()

    with patch.object(loading_base, "_load_trainable_edge_perm_migration") as resolve:
        result = loading_base.apply_trainable_edge_perm_migration(None, model)

    assert result is None
    resolve.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingState
# ---------------------------------------------------------------------------


def test_training_state_from_checkpoint_extracts_all_fields() -> None:
    checkpoint = {
        "epoch": 42,
        "global_step": 10000,
        "best_metric": 0.95,
        "metrics_history": {"loss": [1.0]},
        "state_dict": {"layer.weight": "..."},
    }
    state = TrainingState.from_checkpoint(checkpoint)

    assert state.epoch == 42
    assert state.global_step == 10000
    assert state.best_metric == 0.95
    assert state.metrics_history == {"loss": [1.0]}


def test_training_state_from_checkpoint_defaults_when_missing() -> None:
    state = TrainingState.from_checkpoint({"state_dict": {"layer.weight": "..."}})

    assert state.epoch == 0
    assert state.global_step == 0
    assert state.best_metric is None
    assert state.metrics_history == {}


def test_training_state_to_dict_serializes_all_fields() -> None:
    state = TrainingState(epoch=5, global_step=500, best_metric=0.5, metrics_history={"loss": [1]})

    assert state.to_dict() == {
        "epoch": 5,
        "global_step": 500,
        "best_metric": 0.5,
        "metrics_history": {"loss": [1]},
    }


@pytest.mark.parametrize("epoch", [0, 7])
def test_training_state_apply_to_always_writes_epoch(epoch: int) -> None:
    context = CheckpointContext()

    TrainingState(epoch=epoch).apply_to(context)

    assert context.metadata["epoch"] == epoch


@pytest.mark.parametrize("global_step", [0, 500])
def test_training_state_apply_to_always_writes_global_step(global_step: int) -> None:
    context = CheckpointContext()

    TrainingState(global_step=global_step).apply_to(context)

    assert context.metadata["global_step"] == global_step


def test_training_state_apply_to_writes_best_metric_only_when_not_none() -> None:
    populated = CheckpointContext()
    TrainingState(best_metric=0.95).apply_to(populated)
    assert populated.metadata["best_metric"] == 0.95

    empty = CheckpointContext()
    TrainingState(best_metric=None).apply_to(empty)
    assert "best_metric" not in empty.metadata


def test_training_state_apply_to_writes_metrics_history_only_when_non_empty() -> None:
    populated = CheckpointContext()
    TrainingState(metrics_history={"loss": [1]}).apply_to(populated)
    assert populated.metadata["metrics_history"] == {"loss": [1]}

    empty = CheckpointContext()
    TrainingState(metrics_history={}).apply_to(empty)
    assert "metrics_history" not in empty.metadata


# ---------------------------------------------------------------------------
# filter_state_dict
# ---------------------------------------------------------------------------


def test_filter_state_dict_maps_keys_to_specific_reasons() -> None:
    source: dict[str, Any] = {
        "shared.weight": torch.full((5, 10), 1.0),  # matches -> filtered
        "missing_key": torch.full((3,), 2.0),  # absent from target
        "mismatch_key": torch.full((7, 10), 3.0),  # present but wrong shape
    }
    target: dict[str, Any] = {
        "shared.weight": torch.full((5, 10), 0.0),
        "mismatch_key": torch.full((5, 10), 0.0),
    }

    filtered, skipped = filter_state_dict(source, target)

    assert "shared.weight" in filtered
    assert skipped["missing_key"] == "Key not in target"
    assert skipped["mismatch_key"].startswith("Shape mismatch:")
