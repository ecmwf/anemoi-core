# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TransferLearningLoader."""

import logging

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader


class SourceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(10, 5)
        self.old_head = nn.Linear(5, 3)


class TargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(10, 5)
        self.new_head = nn.Linear(5, 7)  # Different output size


def test_transfer_learning_skips_mismatched_shapes() -> None:
    """Shape mismatches should be skipped, not crash."""
    target = TargetModel()
    source_state = {
        "shared.weight": torch.randn(5, 10),
        "shared.bias": torch.randn(5),
        "old_head.weight": torch.randn(3, 5),  # Not in target
        "old_head.bias": torch.randn(3),
    }

    loader = TransferLearningLoader(skip_mismatched=True)
    context = CheckpointContext(
        model=target,
        checkpoint_data={"state_dict": source_state},
    )
    result = loader.process(context)

    # Shared layer loaded, old_head skipped (not in target)
    assert "skipped_params" in result.metadata
    assert result.model is not None


def test_transfer_learning_filter_is_non_mutating() -> None:
    """The loader leaves ``checkpoint_data`` untouched; ``filter_state_dict`` returns a new mapping."""
    target = TargetModel()
    source_state = {
        "shared.weight": torch.randn(5, 10),
        "shared.bias": torch.randn(5),
        "old_head.weight": torch.randn(3, 5),
        "old_head.bias": torch.randn(3),
    }
    checkpoint_data = {"state_dict": source_state}
    original_keys = set(source_state.keys())

    loader = TransferLearningLoader(skip_mismatched=True)
    context = CheckpointContext(model=target, checkpoint_data=checkpoint_data)
    loader.process(context)

    # Original state dict NOT mutated
    assert set(checkpoint_data["state_dict"].keys()) == original_keys


def test_transfer_learning_tracks_transferred_params() -> None:
    target = TargetModel()
    source_state = {
        "shared.weight": torch.randn(5, 10),
        "shared.bias": torch.randn(5),
    }

    loader = TransferLearningLoader(skip_mismatched=True)
    context = CheckpointContext(
        model=target,
        checkpoint_data={"state_dict": source_state},
    )
    result = loader.process(context)

    assert "transferred_params" in result.metadata
    assert "shared.weight" in result.metadata["transferred_params"]


def test_transfer_learning_sets_weights_initialized() -> None:
    target = TargetModel()
    source_state = {"shared.weight": torch.randn(5, 10), "shared.bias": torch.randn(5)}

    loader = TransferLearningLoader(skip_mismatched=True)
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = loader.process(context)

    assert getattr(result.model, "weights_initialized", False) is True


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


def test_transfer_learning_warns_naming_each_skipped_param(caplog: pytest.LogCaptureFixture) -> None:
    """The skip is loud: one WARNING names every dropped tensor with its reason."""
    target = TargetModel()
    source_state = {
        "shared.weight": torch.randn(5, 10),
        "shared.bias": torch.randn(5),
        "old_head.weight": torch.randn(3, 5),  # not in target
        "new_head.weight": torch.randn(3, 5),  # in target, but (7, 5) there
    }
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    with caplog.at_level(logging.WARNING):
        TransferLearningLoader(skip_mismatched=True).process(context)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "2 checkpoint parameters did not transfer" in warnings[0]
    assert "1 shape-mismatched" in warnings[0]
    assert "1 not present in the model" in warnings[0]
    assert "new_head.weight: Shape mismatch" in warnings[0]
    assert "old_head.weight: Key not in target" in warnings[0]
    assert "more than" not in warnings[0]  # two transferred, two skipped


def test_transfer_learning_warns_when_most_of_the_checkpoint_did_not_transfer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A width change leaves almost nothing to transfer; the warning says so outright."""
    target = TargetModel()
    source_state = {
        "shared.weight": torch.randn(5, 10),
        "shared.bias": torch.randn(9),  # wrong shape
        "new_head.weight": torch.randn(3, 5),  # wrong shape
        "new_head.bias": torch.randn(3),  # wrong shape
    }
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    with caplog.at_level(logging.WARNING):
        result = TransferLearningLoader(skip_mismatched=True).process(context)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "3 shape-mismatched" in warnings[0]
    assert "more than the 1 that did" in warnings[0]
    assert result.metadata["transferred_params"] == ["shared.weight"]


def test_transfer_learning_is_quiet_when_everything_transfers(caplog: pytest.LogCaptureFixture) -> None:
    target = TargetModel()
    source_state = {key: torch.randn_like(value) for key, value in target.state_dict().items()}
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    with caplog.at_level(logging.WARNING):
        result = TransferLearningLoader(skip_mismatched=True).process(context)

    assert _warnings(caplog) == []
    assert result.metadata["skipped_params"] == {}


class _SharedModel(nn.Module):
    """Transfer-learning target with a single ``shared`` layer: (5, 10) weight, (5,) bias."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(10, 5)


def _mixed_source_state() -> dict[str, torch.Tensor]:
    """One shape mismatch, one matching key and one key the target does not have."""
    return {
        "shared.weight": torch.full((7, 10), 1.0),
        "shared.bias": torch.full((5,), 2.0),
        "old_head.weight": torch.full((3, 5), 3.0),
    }


def test_transfer_learning_records_a_reason_string_per_skipped_key() -> None:
    """skipped_params maps each skipped key to why: absent from the target, or a shape mismatch."""
    context = CheckpointContext(model=_SharedModel(), checkpoint_data={"state_dict": _mixed_source_state()})

    result = TransferLearningLoader(skip_mismatched=True).process(context)

    skipped = result.metadata["skipped_params"]
    assert all(isinstance(reason, str) for reason in skipped.values())
    assert skipped["old_head.weight"] == "Key not in target"
    assert skipped["shared.weight"].startswith("Shape mismatch")
    assert "shared.bias" not in skipped


def test_transfer_learning_without_skip_mismatched_raises_on_a_shape_mismatch() -> None:
    state = _mixed_source_state()
    del state["old_head.weight"]
    context = CheckpointContext(model=_SharedModel(), checkpoint_data={"state_dict": state})

    with pytest.raises(CheckpointIncompatibleError, match="Shape mismatch"):
        TransferLearningLoader(skip_mismatched=False).process(context)


def test_transfer_learning_without_skip_mismatched_accepts_a_missing_key() -> None:
    """A key the target does not have is skipped, not an incompatibility, even with skip_mismatched=False."""
    target = _SharedModel()
    state = {
        "shared.weight": torch.full((5, 10), 1.0),
        "shared.bias": torch.full((5,), 2.0),
        "old_head.weight": torch.full((3, 5), 3.0),
    }

    result = TransferLearningLoader(skip_mismatched=False).process(
        CheckpointContext(model=target, checkpoint_data={"state_dict": state}),
    )

    assert result.model is target
    assert result.metadata["skipped_params"]["old_head.weight"] == "Key not in target"


def test_transfer_learning_records_its_strategy_label() -> None:
    state = {"shared.weight": torch.full((5, 10), 1.0), "shared.bias": torch.full((5,), 2.0)}

    result = TransferLearningLoader(skip_mismatched=True).process(
        CheckpointContext(model=_SharedModel(), checkpoint_data={"state_dict": state}),
    )

    assert result.metadata["loading_strategy"] == "transfer_learning"


def test_transfer_learning_with_no_matching_keys_transfers_nothing() -> None:
    state = {"encoder.weight": torch.full((4, 4), 1.0), "encoder.bias": torch.full((4,), 2.0)}

    result = TransferLearningLoader(skip_mismatched=True).process(
        CheckpointContext(model=_SharedModel(), checkpoint_data={"state_dict": state}),
    )

    assert result.metadata["transferred_params"] == []
    assert set(state) <= set(result.metadata["skipped_params"])


def test_transfer_learning_does_not_warn_about_different_hparams(caplog: pytest.LogCaptureFixture) -> None:
    """Transfer learning expects a different model config, so it does not warn about it."""
    target = TargetModel()
    checkpoint = {
        "state_dict": {key: torch.randn_like(value) for key, value in target.state_dict().items()},
        "hyper_parameters": {"config": {"model": {"num_channels": 64}}},
    }
    config = OmegaConf.create({"model": {"num_channels": 128}})
    context = CheckpointContext(model=target, checkpoint_data=checkpoint, config=config)

    with caplog.at_level(logging.WARNING):
        TransferLearningLoader(skip_mismatched=True).process(context)

    assert "hparams differ" not in caplog.text


def test_transfer_learning_sets_variables_metadata() -> None:
    """A transfer-learning load sets ``_ckpt_variables_metadata`` from the checkpoint, keyed by dataset."""
    target = TargetModel()
    variables_metadata = {"2t": {"units": "K"}}
    checkpoint = {
        "state_dict": {key: torch.randn_like(value) for key, value in target.state_dict().items()},
        "hyper_parameters": {
            "data_indices": {"era5": type("IndexCollection", (), {"name_to_index": {"2t": 0}})()},
            "metadata": {"dataset": {"era5": {"variables_metadata": variables_metadata}}},
        },
    }

    result = TransferLearningLoader().process(CheckpointContext(model=target, checkpoint_data=checkpoint))

    assert result.model._ckpt_variables_metadata == {"era5": variables_metadata}
