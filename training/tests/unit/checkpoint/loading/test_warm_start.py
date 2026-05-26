"""Gate loading-g5: WarmStartLoader.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


@pytest.mark.asyncio
async def test_warm_start_restores_model_weights() -> None:
    model = SimpleModel()
    saved_weight = torch.randn(5, 10)
    checkpoint_data = {"state_dict": {"linear.weight": saved_weight, "linear.bias": torch.randn(5)}}

    loader = WarmStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert torch.equal(result.model.linear.weight, saved_weight)


@pytest.mark.asyncio
async def test_warm_start_restores_optimizer_state() -> None:
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simulate a checkpoint with optimizer state
    optimizer_state = optimizer.state_dict()
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "optimizer_states": [optimizer_state],
    }

    loader = WarmStartLoader()
    context = CheckpointContext(model=model, optimizer=optimizer, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.optimizer is not None


@pytest.mark.asyncio
async def test_warm_start_restores_epoch_and_step() -> None:
    model = SimpleModel()
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "epoch": 42,
        "global_step": 10000,
    }

    loader = WarmStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.metadata["epoch"] == 42
    assert result.metadata["global_step"] == 10000


@pytest.mark.asyncio
async def test_warm_start_restores_scheduler_state() -> None:
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "lr_schedulers": [scheduler.state_dict()],
    }

    loader = WarmStartLoader()
    context = CheckpointContext(model=model, optimizer=optimizer, scheduler=scheduler, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.scheduler is not None
