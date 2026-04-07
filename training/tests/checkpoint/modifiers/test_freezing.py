import pytest
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 3)


def test_freezing_adapter_extends_pipeline_stage() -> None:
    assert issubclass(FreezingModifierStage, PipelineStage)


@pytest.mark.asyncio
async def test_freezing_adapter_freezes_specified_module() -> None:
    model = SimpleModel()
    assert model.encoder.weight.requires_grad is True

    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    assert result.model.encoder.weight.requires_grad is False
    assert result.model.decoder.weight.requires_grad is True  # Untouched


@pytest.mark.asyncio
async def test_freezing_adapter_updates_metadata() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    assert "modifier" in result.metadata or "modifiers_applied" in result.metadata
