# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage
from anemoi.training.utils.checkpoint import freeze_submodule_by_name


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


class TwoBranchModel(nn.Module):
    """Two branches each containing a child named ``data``."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({"data": nn.Linear(4, 4)})
        self.decoder = nn.ModuleDict({"data": nn.Linear(4, 4)})


class BranchModel(nn.Module):
    """A sub-model with a direct child and a nested ``Sequential``.

    Used under a two-entry ``ModuleDict`` to mirror the legacy
    ``freeze_submodule_by_name`` fixture from #1159, so the stage is exercised
    on a deep dot-path (``a.sequential.0``) and a bare branch name (``a``).
    """

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.sequential = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))


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
async def test_freezing_bare_name_does_not_match_nested() -> None:
    """A bare name resolves only a direct child, never nested submodules.

    Same path semantics as ``freeze_submodule_by_name`` after #1159.
    """
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["data"], strict=False)
    result = await adapter.process(CheckpointContext(model=model))

    assert result.model.encoder["data"].weight.requires_grad is True
    assert result.model.decoder["data"].weight.requires_grad is True


@pytest.mark.asyncio
async def test_freezing_bare_name_nested_strict_raises() -> None:
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["data"], strict=True)

    with pytest.raises(ValueError, match="not found"):
        await adapter.process(CheckpointContext(model=model))


@pytest.mark.asyncio
async def test_freezing_already_frozen_module_is_not_missing() -> None:
    """A found module whose parameters are already frozen is not an error."""
    model = SimpleModel()
    for param in model.encoder.parameters():
        param.requires_grad = False

    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"], strict=True)
    result = await adapter.process(CheckpointContext(model=model))

    applied = result.metadata["modifiers_applied"][0]
    assert applied["frozen_modules"] == [{"name": "encoder", "frozen_params": 0}]


@pytest.mark.asyncio
async def test_freezing_stage_matches_legacy_helper() -> None:
    """Stage and legacy helper produce identical requires_grad maps (#1159)."""
    targets = ["encoder.data", "missing.path"]

    legacy_model = TwoBranchModel()
    for name in targets:
        freeze_submodule_by_name(legacy_model, name)

    stage_model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=targets, strict=False)
    await adapter.process(CheckpointContext(model=stage_model))

    legacy_map = {name: param.requires_grad for name, param in legacy_model.named_parameters()}
    stage_map = {name: param.requires_grad for name, param in stage_model.named_parameters()}
    assert stage_map == legacy_map


@pytest.mark.asyncio
async def test_freezing_deep_dot_path_targets_single_branch() -> None:
    """A deep dot-path freezes only that submodule; siblings and the other branch stay trainable.

    Ports the legacy ``freeze_submodule_by_name`` test for ``a.sequential.0`` (#1159).
    """
    model = nn.ModuleDict({"a": BranchModel(), "b": BranchModel()})
    adapter = FreezingModifierStage(submodules_to_freeze=["a.sequential.0"])
    result = await adapter.process(CheckpointContext(model=model))

    assert result.model["a"].lin1.weight.requires_grad
    assert not result.model["a"].sequential[0].weight.requires_grad
    assert result.model["a"].sequential[1].weight.requires_grad
    assert result.model["b"].lin1.weight.requires_grad
    assert result.model["b"].sequential[0].weight.requires_grad
    assert result.model["b"].sequential[1].weight.requires_grad


@pytest.mark.asyncio
async def test_freezing_branch_name_freezes_whole_subtree() -> None:
    """A bare branch name freezes the entire branch, leaving the sibling branch trainable.

    Ports the legacy ``freeze_submodule_by_name`` test for ``a`` (#1159).
    """
    model = nn.ModuleDict({"a": BranchModel(), "b": BranchModel()})
    adapter = FreezingModifierStage(submodules_to_freeze=["a"])
    result = await adapter.process(CheckpointContext(model=model))

    assert not result.model["a"].lin1.weight.requires_grad
    assert not result.model["a"].sequential[0].weight.requires_grad
    assert not result.model["a"].sequential[1].weight.requires_grad
    assert result.model["b"].lin1.weight.requires_grad
    assert result.model["b"].sequential[0].weight.requires_grad
    assert result.model["b"].sequential[1].weight.requires_grad


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
async def test_freezing_metadata_accumulates_across_modifiers() -> None:
    model = SimpleModel()
    context = CheckpointContext(model=model)

    context = await FreezingModifierStage(submodules_to_freeze=["encoder"]).process(context)
    context = await FreezingModifierStage(submodules_to_freeze=["decoder"]).process(context)

    applied = context.metadata["modifiers_applied"]
    assert len(applied) == 2
    assert applied[0]["submodules"] == ["encoder"]
    assert applied[1]["submodules"] == ["decoder"]


@pytest.mark.asyncio
async def test_freezing_metadata_includes_param_counts() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = await adapter.process(context)

    meta = result.metadata.get("modifier", result.metadata.get("modifiers_applied", {}))
    meta_str = str(meta).lower()
    assert "frozen" in meta_str or "parameters" in meta_str
