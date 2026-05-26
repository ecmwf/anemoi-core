"""Gate loading-g4: TransferLearningLoader.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

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
