# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint pipeline validation."""

from __future__ import annotations

import sys

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint import CheckpointContext
from anemoi.training.checkpoint import CheckpointPipeline
from anemoi.training.checkpoint import CheckpointPipelineValidator
from anemoi.training.checkpoint import PipelineStage
from anemoi.training.checkpoint import validate_pipeline_health
from anemoi.training.checkpoint.exceptions import CheckpointValidationError


class _Source(PipelineStage):
    """Stage whose class name contains 'Source' so the health check recognises it."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        context.checkpoint_data = {"state_dict": {}}
        return context


class _LoaderMarksWeights(PipelineStage):
    """Stage that simulates a loading strategy setting weights_initialized=True."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        if context.model is not None:
            context.model.weights_initialized = True
        return context


class TestValidateEnvironmentSetup:
    """``CheckpointPipelineValidator.validate_environment_setup``."""

    def test_returns_expected_shape(self) -> None:
        result = CheckpointPipelineValidator.validate_environment_setup()
        assert set(result) >= {"status", "issues", "warnings", "info"}
        assert result["status"] in {"ok", "warning", "error"}
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["info"], list)

    def test_passes_on_current_runtime(self) -> None:
        # The test process itself satisfies the Python/torch floor.
        result = CheckpointPipelineValidator.validate_environment_setup()
        assert result["status"] == "ok", result

    def test_optional_deps_are_info_only(self) -> None:
        result = CheckpointPipelineValidator.validate_environment_setup()
        # Missing optional deps (e.g. Lightning) must never become an issue.
        joined = " ".join(result["issues"])
        assert "lightning" not in joined.lower()

    def test_missing_torch_reported_as_issue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        result = CheckpointPipelineValidator.validate_environment_setup()
        assert result["status"] == "error"
        assert any("torch" in issue.lower() for issue in result["issues"])

    def test_old_python_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        fake = SimpleNamespace(major=3, minor=9, micro=0)
        monkeypatch.setattr(sys, "version_info", fake)
        result = CheckpointPipelineValidator.validate_environment_setup()
        assert result["status"] == "error"
        assert any("Python" in issue for issue in result["issues"])


class TestValidateConfiguration:
    """``CheckpointPipelineValidator.validate_configuration``."""

    def test_none_is_error(self) -> None:
        result = CheckpointPipelineValidator.validate_configuration(None)
        assert result["status"] == "error"
        assert result["issues"]

    def test_empty_config_warns_about_missing_training(self) -> None:
        result = CheckpointPipelineValidator.validate_configuration(OmegaConf.create({}))
        assert result["status"] == "warning"
        assert any("training" in w for w in result["warnings"])

    def test_well_formed_config_is_ok(self) -> None:
        cfg = OmegaConf.create(
            {
                "training": {
                    "checkpoint": {
                        "source": {"_target_": "anemoi.training.checkpoint.sources.LocalSource"},
                        "loading": {"_target_": "anemoi.training.checkpoint.loading.WeightsOnlyLoader"},
                    },
                },
            },
        )
        result = CheckpointPipelineValidator.validate_configuration(cfg)
        assert result["status"] == "ok", result

    def test_empty_checkpoint_block_warns(self) -> None:
        cfg = OmegaConf.create({"training": {"checkpoint": {}}})
        result = CheckpointPipelineValidator.validate_configuration(cfg)
        assert result["status"] == "warning"
        assert any("source" in w and "loading" in w for w in result["warnings"])

    def test_subblock_without_target_warns(self) -> None:
        cfg = OmegaConf.create(
            {"training": {"checkpoint": {"source": {"path": "./model.ckpt"}}}},
        )
        result = CheckpointPipelineValidator.validate_configuration(cfg)
        assert result["status"] == "warning"
        assert any("_target_" in w for w in result["warnings"])


class TestValidatePipelineHealth:
    """``validate_pipeline_health`` post-execution checks."""

    def test_empty_context_is_unhealthy(self) -> None:
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(CheckpointContext())
        assert any("did not execute" in e for e in excinfo.value.validation_errors)

    def test_all_stages_completed_passes(self) -> None:
        ctx = CheckpointContext()
        ctx.update_metadata(
            stage_0_LocalSource="completed",
            stage_1_WeightsOnlyLoader="completed",
            validation_performed=True,
        )
        ctx.model = nn.Linear(2, 2)
        ctx.model.weights_initialized = True
        assert validate_pipeline_health(ctx) is True

    def test_failed_stage_is_error(self) -> None:
        ctx = CheckpointContext()
        ctx.update_metadata(stage_0_X="completed", stage_1_Y="failed: boom")
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(ctx)
        joined = " ".join(excinfo.value.validation_errors)
        assert "stage_1_Y" in joined
        assert "boom" in joined

    def test_source_without_loaded_weights_is_error(self) -> None:
        ctx = CheckpointContext(model=nn.Linear(2, 2))
        ctx.update_metadata(stage_0_LocalSource="completed")
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(ctx)
        assert any("weights_initialized" in e for e in excinfo.value.validation_errors)

    def test_source_with_missing_model_is_error(self) -> None:
        ctx = CheckpointContext()
        ctx.update_metadata(stage_0_S3Source="completed")
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(ctx)
        assert any("model is None" in e for e in excinfo.value.validation_errors)

    def test_optimizer_without_model_is_error(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = __import__("torch").optim.SGD(model.parameters(), lr=0.1)
        # Construct a context with optimizer present but model cleared, so
        # we have to bypass the consistency warning by editing post-init.
        ctx = CheckpointContext(model=model, optimizer=optimizer)
        ctx.model = None
        ctx.update_metadata(stage_0_X="completed")
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(ctx)
        assert any("Optimizer" in e and "model is None" in e for e in excinfo.value.validation_errors)

    def test_pl_module_with_wrong_format_is_error(self) -> None:
        ctx = CheckpointContext(
            pl_module=object(),
            checkpoint_format="pytorch",
        )
        ctx.update_metadata(stage_0_X="completed")
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(ctx)
        assert any("pl_module" in e for e in excinfo.value.validation_errors)

    def test_pre_execution_error_propagates(self) -> None:
        ctx = CheckpointContext()
        ctx.update_metadata(
            stage_0_X="completed",
            validation_environment_status="error",
        )
        with pytest.raises(CheckpointValidationError) as excinfo:
            validate_pipeline_health(ctx)
        assert any("environment" in e.lower() for e in excinfo.value.validation_errors)

    def test_raise_on_error_false_returns_false(self) -> None:
        ctx = CheckpointContext()
        ctx.update_metadata(stage_0_X="failed: nope")
        assert validate_pipeline_health(ctx, raise_on_error=False) is False


class TestPipelineIntegration:
    """End-to-end: pipeline now finds the validation module."""

    @pytest.mark.asyncio
    async def test_validation_performed_in_metadata(self) -> None:
        pipeline = CheckpointPipeline([])
        result = await pipeline.execute(
            CheckpointContext(config=OmegaConf.create({"training": {}})),
        )
        # Pipeline.execute calls _perform_pre_execution_validation, which
        # used to log "module_unavailable" — now it should actually run.
        assert result.metadata.get("validation_performed") is True
        assert "validation_skipped" not in result.metadata
        assert result.metadata.get("validation_environment_status") in {"ok", "warning"}

    @pytest.mark.asyncio
    async def test_health_check_passes_for_realistic_pipeline(self) -> None:
        model = nn.Linear(4, 2)
        pipeline = CheckpointPipeline([_Source(), _LoaderMarksWeights()])
        result = await pipeline.execute(CheckpointContext(model=model))
        assert validate_pipeline_health(result) is True
