# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
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


class TwoBranchModel(nn.Module):
    """Two branches each containing a child named ``data``."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({"data": nn.Linear(4, 4)})
        self.decoder = nn.ModuleDict({"data": nn.Linear(4, 4)})


class BranchModel(nn.Module):
    """A sub-model with a direct child and a nested ``Sequential``.

    Used in a two-entry ``ModuleDict`` to test dot-paths (#1159).
    """

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.sequential = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))


def test_freezing_adapter_extends_pipeline_stage() -> None:
    assert issubclass(FreezingModifierStage, PipelineStage)


def test_freezing_adapter_freezes_specified_module() -> None:
    model = SimpleModel()
    assert model.encoder.weight.requires_grad is True

    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = adapter.process(context)

    assert result.model.encoder.weight.requires_grad is False
    assert result.model.decoder.weight.requires_grad is True  # Untouched


def test_freezing_adapter_updates_metadata() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = adapter.process(context)

    assert "modifier" in result.metadata or "modifiers_applied" in result.metadata


def test_freezing_strict_mode_raises_on_missing_submodule() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["nonexistent_module"], strict=True)
    context = CheckpointContext(model=model)

    with pytest.raises((ValueError, AttributeError)):
        adapter.process(context)


def test_freezing_non_strict_skips_missing_submodule() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["nonexistent_module", "encoder"], strict=False)
    context = CheckpointContext(model=model)
    result = adapter.process(context)

    # encoder is still frozen despite the missing module being skipped
    assert result.model.encoder.weight.requires_grad is False


def test_freezing_dot_notation_submodule() -> None:
    model = NestedModel()
    assert model.processor[0].weight.requires_grad is True
    assert model.processor[1].weight.requires_grad is True

    adapter = FreezingModifierStage(submodules_to_freeze=["processor.0"])
    context = CheckpointContext(model=model)
    result = adapter.process(context)

    assert result.model.processor[0].weight.requires_grad is False
    assert result.model.processor[1].weight.requires_grad is True  # Untouched
    assert result.model.head.weight.requires_grad is True  # Untouched


def test_freezing_bare_name_does_not_match_nested() -> None:
    """A bare name resolves only a direct child, never nested submodules.

    ``encoder.data`` resolves, so that the config is not fully inert.
    """
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["data", "encoder.data"], strict=False)
    result = adapter.process(CheckpointContext(model=model))

    assert result.model.encoder["data"].weight.requires_grad is False
    assert result.model.decoder["data"].weight.requires_grad is True


def test_freezing_bare_name_nested_strict_raises() -> None:
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["data"], strict=True)

    with pytest.raises(ValueError, match="not found"):
        adapter.process(CheckpointContext(model=model))


def test_freezing_already_frozen_module_is_not_missing() -> None:
    """A found module whose parameters are already frozen is not an error."""
    model = SimpleModel()
    for param in model.encoder.parameters():
        param.requires_grad = False

    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"], strict=True)
    result = adapter.process(CheckpointContext(model=model))

    applied = result.metadata["modifiers_applied"][0]
    assert applied["frozen_modules"] == [{"name": "encoder", "frozen_params": 0}]


def test_freezing_dot_path_into_branch_skips_missing() -> None:
    """A dot-path freezes only its ModuleDict entry. A missing path is skipped."""
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder.data", "missing.path"], strict=False)
    adapter.process(CheckpointContext(model=model))

    assert model.encoder["data"].weight.requires_grad is False
    assert model.decoder["data"].weight.requires_grad is True


def test_freezing_deep_dot_path_targets_single_branch() -> None:
    """A deep dot-path freezes only that submodule; siblings and the other branch stay trainable."""
    model = nn.ModuleDict({"a": BranchModel(), "b": BranchModel()})
    adapter = FreezingModifierStage(submodules_to_freeze=["a.sequential.0"])
    result = adapter.process(CheckpointContext(model=model))

    assert result.model["a"].lin1.weight.requires_grad
    assert not result.model["a"].sequential[0].weight.requires_grad
    assert result.model["a"].sequential[1].weight.requires_grad
    assert result.model["b"].lin1.weight.requires_grad
    assert result.model["b"].sequential[0].weight.requires_grad
    assert result.model["b"].sequential[1].weight.requires_grad


def test_freezing_branch_name_freezes_whole_subtree() -> None:
    """A bare branch name freezes the entire branch, leaving the sibling branch trainable."""
    model = nn.ModuleDict({"a": BranchModel(), "b": BranchModel()})
    adapter = FreezingModifierStage(submodules_to_freeze=["a"])
    result = adapter.process(CheckpointContext(model=model))

    assert not result.model["a"].lin1.weight.requires_grad
    assert not result.model["a"].sequential[0].weight.requires_grad
    assert not result.model["a"].sequential[1].weight.requires_grad
    assert result.model["b"].lin1.weight.requires_grad
    assert result.model["b"].sequential[0].weight.requires_grad
    assert result.model["b"].sequential[1].weight.requires_grad


def test_freezing_gradient_validation() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"], validate_gradients=True)
    context = CheckpointContext(model=model)
    result = adapter.process(context)

    for param in result.model.encoder.parameters():
        assert param.requires_grad is False
        assert param.grad is None  # No stale gradients on frozen parameters


def test_freezing_metadata_accumulates_across_modifiers() -> None:
    model = SimpleModel()
    context = CheckpointContext(model=model)

    context = FreezingModifierStage(submodules_to_freeze=["encoder"]).process(context)
    context = FreezingModifierStage(submodules_to_freeze=["decoder"]).process(context)

    applied = context.metadata["modifiers_applied"]
    assert len(applied) == 2
    assert applied[0]["submodules"] == ["encoder"]
    assert applied[1]["submodules"] == ["decoder"]


def test_freezing_metadata_includes_param_counts() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
    context = CheckpointContext(model=model)
    result = adapter.process(context)

    meta = result.metadata.get("modifier", result.metadata.get("modifiers_applied", {}))
    meta_str = str(meta).lower()
    assert "frozen" in meta_str or "parameters" in meta_str


class WrappedModel(nn.Module):
    """Copies the real nesting: LightningModule -> interface -> graph model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.model = TwoBranchModel()
        self.model.pre_processors = nn.Linear(4, 4)


def test_freezing_resolves_paths_against_submodule_root() -> None:
    """With a root set, paths start at the graph model, not the LightningModule."""
    model = WrappedModel()
    adapter = FreezingModifierStage(
        submodules_to_freeze=["encoder.data"],
        submodule_root="model.model",
    )

    result = adapter.process(CheckpointContext(model=model))

    assert result.model.model.model.encoder["data"].weight.requires_grad is False
    assert result.model.model.model.decoder["data"].weight.requires_grad is True
    # The root is not frozen.
    assert result.model.model.pre_processors.weight.requires_grad is True


def test_freezing_records_frozen_params_under_root() -> None:
    """The recorded parameter count is that of the real submodule."""
    model = WrappedModel()
    adapter = FreezingModifierStage(
        submodules_to_freeze=["encoder.data"],
        submodule_root="model.model",
    )

    result = adapter.process(CheckpointContext(model=model))

    (record,) = result.metadata["modifiers_applied"]
    assert record["frozen_modules"] == [{"name": "encoder.data", "frozen_params": 2}]
    assert record["total_frozen_params"] == 2


def test_freezing_raises_when_nothing_resolves() -> None:
    """A config that freezes nothing raises, also when strict is False."""
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder", "decoder"], submodule_root="", strict=False)
    # Control: these paths resolve.
    adapter.process(CheckpointContext(model=TwoBranchModel()))

    inert = FreezingModifierStage(submodules_to_freeze=["processor", "nope"], strict=False)
    with pytest.raises(CheckpointConfigError, match="not one path resolved"):
        inert.process(CheckpointContext(model=model))


def test_freezing_raises_when_submodule_root_does_not_resolve() -> None:
    """A root that does not exist raises a config error."""
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"], submodule_root="model.model")

    with pytest.raises(CheckpointConfigError, match="submodule_root"):
        adapter.process(CheckpointContext(model=SimpleModel()))


def test_freezing_raises_without_a_model() -> None:
    """No model on the context raises a config error."""
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])

    with pytest.raises(CheckpointConfigError, match=r"context\.model is None"):
        adapter.process(CheckpointContext(model=None))


def test_gradient_validation_sees_the_rooted_modules(caplog: pytest.LogCaptureFixture) -> None:
    """Gradient validation uses the same root as the freeze.

    A stub makes one parameter trainable again before validation, so validation must warn.
    """
    model = WrappedModel()

    class _ReArmingFreeze(FreezingModifierStage):
        """Freezes, then makes one parameter trainable again."""

        def _freeze_submodule_by_name(self, module: nn.Module, target_name: str) -> int | None:
            frozen = super()._freeze_submodule_by_name(module, target_name)
            if frozen is not None:
                module.get_submodule(target_name).weight.requires_grad = True
            return frozen

    adapter = _ReArmingFreeze(
        submodules_to_freeze=["encoder.data"],
        submodule_root="model.model",
        validate_gradients=True,
    )

    with caplog.at_level(logging.WARNING):
        adapter.process(CheckpointContext(model=model))

    assert "still has trainable parameters" in caplog.text
    assert "encoder.data" in caplog.text
