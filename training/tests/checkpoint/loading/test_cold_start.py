"""Gate loading-g3: ColdStartLoader.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import ColdStartLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


@pytest.mark.asyncio
async def test_cold_start_loads_weights() -> None:
    model = SimpleModel()
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = ColdStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.model is not None


@pytest.mark.asyncio
async def test_cold_start_resets_training_state() -> None:
    """Epoch and global_step must be 0 regardless of checkpoint content."""
    model = SimpleModel()
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "epoch": 42,
        "global_step": 10000,
    }

    loader = ColdStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.metadata.get("epoch", 0) == 0
    assert result.metadata.get("global_step", 0) == 0
    assert result.optimizer is None
    assert result.scheduler is None


@pytest.mark.asyncio
async def test_cold_start_tracks_pretrained_source() -> None:
    model = SimpleModel()
    ckpt_path = "/path/to/pretrained.ckpt"
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = ColdStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data, checkpoint_path=ckpt_path)
    result = await loader.process(context)

    assert "pretrained_from" in result.metadata
