"""Gate loading-g2: WeightsOnlyLoader.

CANONICAL GATE TEST — DO NOT MODIFY.
"""

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_weights_only_extends_pipeline_stage() -> None:
    assert issubclass(WeightsOnlyLoader, PipelineStage)


@pytest.mark.asyncio
async def test_weights_only_loads_state_dict() -> None:
    model = SimpleModel()
    original_weight = model.linear.weight.clone()

    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    # Weights changed
    assert not torch.equal(result.model.linear.weight, original_weight)


@pytest.mark.asyncio
async def test_weights_only_discards_optimizer() -> None:
    """Optimizer state must be explicitly set to None."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "optimizer_states": [{"some": "state"}],
    }

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, optimizer=optimizer, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.optimizer is None
    assert result.scheduler is None


@pytest.mark.asyncio
async def test_weights_only_sets_metadata() -> None:
    model = SimpleModel()
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await loader.process(context)

    assert result.metadata.get("loading_strategy") == "weights_only"
