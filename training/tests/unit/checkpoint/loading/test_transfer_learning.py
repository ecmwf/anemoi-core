# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TransferLearningLoader."""

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
    """CRITICAL: Must not mutate the checkpoint_data dict. PR #458 approach, NOT legacy."""
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
