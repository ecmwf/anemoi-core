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

from anemoi.training.checkpoint.base import CheckpointContext
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


@pytest.mark.asyncio
async def test_transfer_learning_skips_mismatched_shapes() -> None:
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
    result = await loader.process(context)

    # Shared layer loaded, old_head skipped (not in target)
    assert "skipped_params" in result.metadata
    assert result.model is not None


@pytest.mark.asyncio
async def test_transfer_learning_filter_is_non_mutating() -> None:
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
    await loader.process(context)

    # Original state dict NOT mutated
    assert set(checkpoint_data["state_dict"].keys()) == original_keys


@pytest.mark.asyncio
async def test_transfer_learning_tracks_transferred_params() -> None:
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
    result = await loader.process(context)

    assert "transferred_params" in result.metadata
    assert "shared.weight" in result.metadata["transferred_params"]


@pytest.mark.asyncio
async def test_transfer_learning_sets_weights_initialized() -> None:
    target = TargetModel()
    source_state = {"shared.weight": torch.randn(5, 10), "shared.bias": torch.randn(5)}

    loader = TransferLearningLoader(skip_mismatched=True)
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})
    result = await loader.process(context)

    assert getattr(result.model, "weights_initialized", False) is True


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_transfer_learning_warns_naming_each_skipped_param(caplog: pytest.LogCaptureFixture) -> None:
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
        await TransferLearningLoader(skip_mismatched=True).process(context)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "2 checkpoint parameters did not transfer" in warnings[0]
    assert "1 shape-mismatched" in warnings[0]
    assert "1 not present in the model" in warnings[0]
    assert "new_head.weight: Shape mismatch" in warnings[0]
    assert "old_head.weight: Key not in target" in warnings[0]
    assert "more than" not in warnings[0]  # two transferred, two skipped


@pytest.mark.asyncio
async def test_transfer_learning_warns_when_most_of_the_checkpoint_did_not_transfer(
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
        result = await TransferLearningLoader(skip_mismatched=True).process(context)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "3 shape-mismatched" in warnings[0]
    assert "more than the 1 that did" in warnings[0]
    assert result.metadata["transferred_params"] == ["shared.weight"]


@pytest.mark.asyncio
async def test_transfer_learning_is_quiet_when_everything_transfers(caplog: pytest.LogCaptureFixture) -> None:
    target = TargetModel()
    source_state = {key: torch.randn_like(value) for key, value in target.state_dict().items()}
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    with caplog.at_level(logging.WARNING):
        result = await TransferLearningLoader(skip_mismatched=True).process(context)

    assert _warnings(caplog) == []
    assert result.metadata["skipped_params"] == {}
