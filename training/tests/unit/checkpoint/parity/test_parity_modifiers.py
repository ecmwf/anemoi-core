# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity coverage for model modifiers and pipeline composition/validation.

Every test drives the real checkpoint classes on CPU with synthetic models and
in-memory (or tmp-path) checkpoints. No GPU, no network, no RNG-dependent
assertions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint import validate_pipeline_health
from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.loading.base import LoadingStrategy
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader
from anemoi.training.checkpoint.modifiers.base import ModelModifier
from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage
from anemoi.training.checkpoint.pipeline import CheckpointPipeline
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class TwoChildModel(nn.Module):
    """Two named children, each an ``nn.Linear`` (weight + bias)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 3)


class ThreeChildModel(nn.Module):
    """Three freezable children plus one unlisted child that must stay trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 3)
        self.head = nn.Linear(3, 2)
        self.extra = nn.Linear(2, 2)


class MismatchTargetModel(nn.Module):
    """Target whose ``encoder`` shape differs from the source checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(8, 4)


# ---------------------------------------------------------------------------
# Real pipeline-stage subclasses (layer bases, deliberately off-convention names)
# ---------------------------------------------------------------------------


class _InMemorySource(CheckpointSource):
    """A real ``CheckpointSource`` that attaches in-memory checkpoint data."""

    def __init__(self, checkpoint_data: dict | None = None) -> None:
        self._data = checkpoint_data if checkpoint_data is not None else {"state_dict": {}}

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        context.checkpoint_data = self._data
        return context


class _WeightMarkingLoader(LoadingStrategy):
    """A real ``LoadingStrategy`` that only marks the model weights as loaded."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        if context.model is not None:
            self._mark_weights_loaded(context.model)
        return context


class _NoOpStage(PipelineStage):
    """A bare stage that neither loads weights nor belongs to any layer."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _CustomModifier(ModelModifier):
    """An inline modifier that freezes the whole model and records itself."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        for param in context.model.parameters():
            param.requires_grad = False
        context.metadata.setdefault("modifiers_applied", []).append({"type": "custom_recording"})
        return context


class _CustomRestorer(CheckpointSource):
    """A ``CheckpointSource`` subclass whose name contains no 'Source'."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _Restorer(LoadingStrategy):
    """A ``LoadingStrategy`` subclass whose name contains no 'Loader'."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _LoRAAdapter(ModelModifier):
    """A ``ModelModifier`` subclass whose name contains no 'Modifier'/'Freez'."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _UnknownStage(PipelineStage):
    """A bare ``PipelineStage`` that belongs to no checkpoint layer."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


# ---------------------------------------------------------------------------
# Freezing modifier behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_freezing_multiple_submodules_in_single_call() -> None:
    model = ThreeChildModel()
    names = ["encoder", "decoder", "head"]
    stage = FreezingModifierStage(submodules_to_freeze=names)

    result = await stage.process(CheckpointContext(model=model))

    for name in names:
        submodule = result.model.get_submodule(name)
        assert all(not p.requires_grad for p in submodule.parameters())
    # Unlisted submodule is untouched.
    assert all(p.requires_grad for p in result.model.extra.parameters())

    applied = result.metadata["modifiers_applied"]
    assert len(applied) == 1
    assert len(applied[0]["frozen_modules"]) == 3
    assert [record["name"] for record in applied[0]["frozen_modules"]] == names


@pytest.mark.asyncio
async def test_validate_gradients_false_skips_validation_but_still_freezes() -> None:
    model = TwoChildModel()

    with patch.object(FreezingModifierStage, "_validate_gradient_flow") as spy:
        stage = FreezingModifierStage(submodules_to_freeze=["encoder"], validate_gradients=False)
        result = await stage.process(CheckpointContext(model=model))

    spy.assert_not_called()
    assert all(not p.requires_grad for p in result.model.encoder.parameters())


@pytest.mark.asyncio
async def test_validate_gradients_true_invokes_validation() -> None:
    model = TwoChildModel()

    with patch.object(FreezingModifierStage, "_validate_gradient_flow") as spy:
        stage = FreezingModifierStage(submodules_to_freeze=["encoder"], validate_gradients=True)
        await stage.process(CheckpointContext(model=model))

    spy.assert_called_once()


@pytest.mark.asyncio
async def test_modifiers_applied_records_param_counts_and_total() -> None:
    model = TwoChildModel()
    expected_encoder = sum(p.requires_grad for p in model.encoder.parameters())
    expected_decoder = sum(p.requires_grad for p in model.decoder.parameters())

    stage = FreezingModifierStage(submodules_to_freeze=["encoder", "decoder"])
    result = await stage.process(CheckpointContext(model=model))

    entry = result.metadata["modifiers_applied"][0]
    assert entry["type"] == "freezing"
    assert entry["submodules"] == ["encoder", "decoder"]

    counts = {record["name"]: record["frozen_params"] for record in entry["frozen_modules"]}
    assert counts["encoder"] == expected_encoder
    assert counts["decoder"] == expected_decoder
    assert expected_encoder > 0
    assert expected_decoder > 0
    assert entry["total_frozen_params"] == expected_encoder + expected_decoder


@pytest.mark.asyncio
async def test_empty_freeze_list_is_noop() -> None:
    model = TwoChildModel()
    stage = FreezingModifierStage(submodules_to_freeze=[])

    result = await stage.process(CheckpointContext(model=model))

    assert all(p.requires_grad for p in result.model.parameters())
    # Early return happens before any metadata is appended.
    assert "modifiers_applied" not in result.metadata


# ---------------------------------------------------------------------------
# Pipeline composition warnings/suggestions
# ---------------------------------------------------------------------------


def test_empty_pipeline_warns_and_has_zero_length(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        pipeline = CheckpointPipeline([])

    assert len(pipeline) == 0
    assert any("no stages" in record.getMessage().lower() for record in caplog.records)


def test_loader_without_source_suggests_adding_a_source(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        CheckpointPipeline([WeightsOnlyLoader()])

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert any("source" in message.lower() for message in info_messages)


def test_source_without_loader_suggests_adding_a_loading_stage(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        CheckpointPipeline([LocalSource(path="model.ckpt")])

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert any("loading" in message.lower() for message in info_messages)


# ---------------------------------------------------------------------------
# Pipeline weights-loaded gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_without_loaded_weights_raises_even_with_continue_on_error() -> None:
    for continue_on_error in (False, True):
        model = TwoChildModel()
        pipeline = CheckpointPipeline(
            [_InMemorySource(), _NoOpStage()],
            continue_on_error=continue_on_error,
        )
        with pytest.raises(CheckpointLoadError):
            await pipeline.execute(CheckpointContext(model=model))


@pytest.mark.asyncio
async def test_source_with_loaded_weights_completes() -> None:
    model = TwoChildModel()
    pipeline = CheckpointPipeline([_InMemorySource(), _WeightMarkingLoader()])

    result = await pipeline.execute(CheckpointContext(model=model))

    assert result.model.weights_initialized is True


# ---------------------------------------------------------------------------
# Hydra instantiation of stages
# ---------------------------------------------------------------------------


def test_invalid_target_raises_config_error() -> None:
    with pytest.raises(CheckpointConfigError, match="Failed to instantiate pipeline stage 0"):
        CheckpointPipeline([{"_target_": "nonexistent.module.Class", "param": "value"}])


def test_pipeline_instantiates_real_stages_from_hydra_targets() -> None:
    pipeline = CheckpointPipeline(
        [
            {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": "model.ckpt"},
            {"_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"},
            {
                "_target_": "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage",
                "submodules_to_freeze": ["encoder"],
            },
        ],
    )

    assert isinstance(pipeline.stages[0], LocalSource)
    assert isinstance(pipeline.stages[1], WeightsOnlyLoader)
    assert isinstance(pipeline.stages[2], FreezingModifierStage)


# ---------------------------------------------------------------------------
# Stage-role classification by isinstance (name-independent)
# ---------------------------------------------------------------------------


def test_stage_role_classifies_source_by_isinstance() -> None:
    assert CheckpointPipeline._stage_role(_CustomRestorer()) == "source"


def test_stage_role_classifies_loader_by_isinstance() -> None:
    assert CheckpointPipeline._stage_role(_Restorer()) == "loader"


def test_stage_role_classifies_modifier_by_isinstance() -> None:
    assert CheckpointPipeline._stage_role(_LoRAAdapter()) == "modifier"


def test_stage_role_returns_none_for_unrecognized_stage() -> None:
    assert CheckpointPipeline._stage_role(_UnknownStage()) is None


# ---------------------------------------------------------------------------
# Post-execution health validation
# ---------------------------------------------------------------------------


def test_health_check_flags_scheduler_without_optimizer() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)

    ctx = CheckpointContext(model=model, optimizer=optimizer, scheduler=scheduler)
    ctx.optimizer = None
    ctx.update_metadata(stage_0_X="completed")

    with pytest.raises(CheckpointValidationError) as excinfo:
        validate_pipeline_health(ctx)
    assert any("Scheduler" in error and "optimizer is None" in error for error in excinfo.value.validation_errors)


def test_health_check_flags_config_validation_error() -> None:
    ctx = CheckpointContext()
    ctx.update_metadata(stage_0_X="completed", validation_config_status="error")

    with pytest.raises(CheckpointValidationError) as excinfo:
        validate_pipeline_health(ctx)
    assert any("configuration" in error.lower() for error in excinfo.value.validation_errors)


# ---------------------------------------------------------------------------
# Cross-layer executed pipelines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_inline_modifier_runs_and_classifies_as_modifier() -> None:
    model = TwoChildModel()
    modifier = _CustomModifier()
    pipeline = CheckpointPipeline([_InMemorySource(), _WeightMarkingLoader(), modifier])

    result = await pipeline.execute(CheckpointContext(model=model))

    assert all(not p.requires_grad for p in result.model.parameters())
    assert {"type": "custom_recording"} in result.metadata["modifiers_applied"]
    assert CheckpointPipeline._stage_role(modifier) == "modifier"


@pytest.mark.asyncio
async def test_transfer_learning_then_freezing_compose_in_pipeline(tmp_path: Path) -> None:
    target = MismatchTargetModel()
    source_state = {
        "encoder.weight": torch.randn(7, 10),  # shape mismatch vs (5, 10)
        "encoder.bias": torch.randn(7),  # shape mismatch vs (5,)
        "decoder.weight": torch.randn(4, 8),  # matches target
        "decoder.bias": torch.randn(4),  # matches target
    }
    ckpt_path = tmp_path / "pretrained.ckpt"
    torch.save({"state_dict": source_state}, ckpt_path)

    pipeline = CheckpointPipeline(
        [
            LocalSource(path=str(ckpt_path)),
            TransferLearningLoader(skip_mismatched=True),
            FreezingModifierStage(submodules_to_freeze=["encoder"]),
        ],
    )

    result = await pipeline.execute(CheckpointContext(model=target))

    # Modifier froze the encoder.
    assert all(not p.requires_grad for p in result.model.encoder.parameters())

    # Transfer-learning loader recorded transferred and skipped params.
    assert result.metadata["loading_strategy"] == "transfer_learning"
    assert "decoder.weight" in result.metadata["transferred_params"]
    assert "encoder.weight" in result.metadata["skipped_params"]
    assert "Shape mismatch" in result.metadata["skipped_params"]["encoder.weight"]

    # Freezing record composed alongside the loader state.
    freezing_records = [entry for entry in result.metadata["modifiers_applied"] if entry["type"] == "freezing"]
    assert freezing_records
    assert freezing_records[0]["frozen_modules"][0]["name"] == "encoder"
