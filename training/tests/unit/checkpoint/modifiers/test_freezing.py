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
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage
from anemoi.training.checkpoint.pipeline import CheckpointPipeline


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

    Used under a two-entry ``ModuleDict`` to mirror the #1159 dot-path fixture,
    so the stage is exercised on a deep dot-path (``a.sequential.0``) and a bare
    branch name (``a``).
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

    Same dot-path semantics introduced in #1159. The resolving ``encoder.data``
    is paired with the non-resolving bare ``data`` so the config as a whole is
    not inert: a run where *nothing* resolves is now refused outright, which is
    asserted separately.
    """
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["data", "encoder.data"], strict=False)
    result = adapter.process(CheckpointContext(model=model))

    # The dot path froze its target; the bare name matched nothing, so the
    # identically-named child under the other branch is untouched.
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
    """A dot-path into a ModuleDict branch freezes only that entry; a missing path is skipped (#1159)."""
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder.data", "missing.path"], strict=False)
    adapter.process(CheckpointContext(model=model))

    assert model.encoder["data"].weight.requires_grad is False  # frozen via dot-path
    assert model.decoder["data"].weight.requires_grad is True  # untouched sibling branch


def test_freezing_deep_dot_path_targets_single_branch() -> None:
    """A deep dot-path freezes only that submodule; siblings and the other branch stay trainable.

    Ports the #1159 dot-path test for ``a.sequential.0``.
    """
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
    """A bare branch name freezes the entire branch, leaving the sibling branch trainable.

    Ports the #1159 dot-path test for ``a``.
    """
    model = nn.ModuleDict({"a": BranchModel(), "b": BranchModel()})
    adapter = FreezingModifierStage(submodules_to_freeze=["a"])
    result = adapter.process(CheckpointContext(model=model))

    assert not result.model["a"].lin1.weight.requires_grad
    assert not result.model["a"].sequential[0].weight.requires_grad
    assert not result.model["a"].sequential[1].weight.requires_grad
    assert result.model["b"].lin1.weight.requires_grad
    assert result.model["b"].sequential[0].weight.requires_grad
    assert result.model["b"].sequential[1].weight.requires_grad


def test_freezing_leaves_frozen_params_without_grad() -> None:
    model = SimpleModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])
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


class WrappedModel(nn.Module):
    """Mirrors the real nesting: LightningModule -> interface -> graph model.

    ``encoder`` / ``processor`` / ``decoder`` live two attribute hops below the
    object handed to the stage, so their paths start with ``model.model.``.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.model = TwoBranchModel()
        self.model.pre_processors = nn.Linear(4, 4)


def test_freezing_resolves_full_paths_from_the_lightning_module() -> None:
    """Paths start at the object handed to the stage, so the graph model needs the full prefix."""
    model = WrappedModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["model.model.encoder.data"])

    result = adapter.process(CheckpointContext(model=model))

    assert result.model.model.model.encoder["data"].weight.requires_grad is False
    assert result.model.model.model.decoder["data"].weight.requires_grad is True
    # Only the named path is frozen: modules along the prefix keep their siblings trainable.
    assert result.model.model.pre_processors.weight.requires_grad is True


def test_freezing_records_frozen_params_by_full_path() -> None:
    """The recorded parameter count reflects the real submodule, not zero."""
    model = WrappedModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["model.model.encoder.data"])

    result = adapter.process(CheckpointContext(model=model))

    (record,) = result.metadata["modifiers_applied"]
    assert record["frozen_modules"] == [{"name": "model.model.encoder.data", "frozen_params": 2}]
    assert record["total_frozen_params"] == 2


def test_freezing_raises_when_nothing_resolves() -> None:
    """A config that freezes nothing is an error even when strict is False.

    ``strict`` tolerates *one* missing name. It has never meant "tolerate a
    config that is entirely inert" — that is the silent no-op which let every
    documented freezing config train all parameters unnoticed.
    """
    model = TwoBranchModel()
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder", "decoder"], strict=False)
    # Sanity: these DO resolve here, so the negative case below is not vacuous.
    adapter.process(CheckpointContext(model=TwoBranchModel()))

    inert = FreezingModifierStage(submodules_to_freeze=["processor", "nope"], strict=False)
    with pytest.raises(CheckpointConfigError, match="not one path resolved"):
        inert.process(CheckpointContext(model=model))


def test_freezing_graph_model_name_without_prefix_names_the_full_path_form() -> None:
    """A graph-model name without the ``model.model.`` prefix freezes nothing on a LightningModule.

    The error names the full-path form, so the fix is in the message.
    """
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder.data"])

    with pytest.raises(CheckpointConfigError, match=r"model\.model\.encoder\.<dataset>") as excinfo:
        adapter.process(CheckpointContext(model=WrappedModel()))
    assert "submodule_root" not in str(excinfo.value)


def test_freezing_raises_without_a_model() -> None:
    """No model on the context is a config error rather than an AttributeError."""
    adapter = FreezingModifierStage(submodules_to_freeze=["encoder"])

    with pytest.raises(CheckpointConfigError, match=r"context\.model is None"):
        adapter.process(CheckpointContext(model=None))


@pytest.mark.parametrize(
    "removed_kwarg",
    [{"submodule_root": "model.model"}, {"validate_gradients": True}],
)
def test_freezing_config_with_removed_parameter_fails_at_instantiation(removed_kwarg: dict) -> None:
    """A config still carrying a removed parameter fails when the stage is built, naming it."""
    (name,) = removed_kwarg
    stage_cfg = {
        "_target_": "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage",
        "submodules_to_freeze": ["model.model.encoder.data"],
        **removed_kwarg,
    }

    with pytest.raises(CheckpointConfigError, match=name):
        CheckpointPipeline(stages=[OmegaConf.create(stage_cfg)])


def test_freezing_preset_example_path_freezes_the_graph_model_child() -> None:
    """The shipped preset instantiates and its documented path form resolves on the real nesting."""
    from importlib.resources import files

    preset = files("anemoi.training") / "config" / "training" / "checkpoint" / "modifiers" / "freezing.yaml"
    (stage_cfg,) = OmegaConf.create(preset.read_text())
    assert set(stage_cfg) == {"_target_", "submodules_to_freeze", "strict"}
    stage_cfg.submodules_to_freeze = ["model.model.encoder.data"]

    (stage,) = CheckpointPipeline(stages=[stage_cfg]).stages
    result = stage.process(CheckpointContext(model=WrappedModel()))

    assert result.model.model.model.encoder["data"].weight.requires_grad is False
    assert result.model.model.model.decoder["data"].weight.requires_grad is True


class ThreeChildModel(nn.Module):
    """Three freezable children plus one unlisted child that must stay trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 3)
        self.head = nn.Linear(3, 2)
        self.extra = nn.Linear(2, 2)


def test_freezing_several_submodules_records_one_entry_with_per_module_counts() -> None:
    """One stage freezing several submodules writes one record: per-module counts and their total."""
    model = ThreeChildModel()
    names = ["encoder", "decoder", "head"]
    expected = {name: sum(p.requires_grad for p in model.get_submodule(name).parameters()) for name in names}

    result = FreezingModifierStage(submodules_to_freeze=names).process(CheckpointContext(model=model))

    for name in names:
        assert all(not p.requires_grad for p in result.model.get_submodule(name).parameters())
    assert all(p.requires_grad for p in result.model.extra.parameters())

    applied = result.metadata["modifiers_applied"]
    assert len(applied) == 1
    entry = applied[0]
    assert entry["type"] == "freezing"
    assert entry["submodules"] == names
    assert [record["name"] for record in entry["frozen_modules"]] == names
    assert {record["name"]: record["frozen_params"] for record in entry["frozen_modules"]} == expected
    assert all(count > 0 for count in expected.values())
    assert entry["total_frozen_params"] == sum(expected.values())


def test_empty_freeze_list_is_a_no_op() -> None:
    """An empty list freezes nothing and records nothing."""
    model = SimpleModel()

    result = FreezingModifierStage(submodules_to_freeze=[]).process(CheckpointContext(model=model))

    assert all(p.requires_grad for p in result.model.parameters())
    assert "modifiers_applied" not in result.metadata
