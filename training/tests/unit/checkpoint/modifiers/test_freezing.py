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


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])
        self.head = nn.Linear(10, 3)


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


@pytest.mark.asyncio
async def test_freezing_strict_mode_raises_on_missing_submodule() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["nonexistent_module"], strict=True)
    context = CheckpointContext(model=model)

    with pytest.raises((ValueError, AttributeError)):
        await adapter.process(context)


@pytest.mark.asyncio
async def test_freezing_non_strict_skips_missing_submodule() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["nonexistent_module", "encoder"], strict=False)
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    # encoder is still frozen despite the missing module being skipped
    assert result.model.encoder.weight.requires_grad is False


@pytest.mark.asyncio
async def test_freezing_dot_notation_submodule() -> None:
    model = NestedModel()
    assert model.processor[0].weight.requires_grad is True
    assert model.processor[1].weight.requires_grad is True

    adapter = FreezingModifierStage(submodules_to_freeze=["processor.0"])
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    assert result.model.processor[0].weight.requires_grad is False
    assert result.model.processor[1].weight.requires_grad is True  # Untouched
    assert result.model.head.weight.requires_grad is True  # Untouched


@pytest.mark.asyncio
async def test_freezing_gradient_validation() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"], validate_gradients=True)
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    for param in result.model.encoder.parameters():
        assert param.requires_grad is False
        assert param.grad is None  # No stale gradients on frozen parameters


@pytest.mark.asyncio
async def test_freezing_metadata_includes_param_counts() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    meta = result.metadata.get("modifier", result.metadata.get("modifiers_applied", {}))
    meta_str = str(meta).lower()
    assert "frozen" in meta_str or "parameters" in meta_str
