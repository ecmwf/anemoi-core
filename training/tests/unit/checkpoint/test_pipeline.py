# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint pipeline orchestrator."""

import logging
from typing import Any

import pytest

from anemoi.training.checkpoint import CheckpointContext
from anemoi.training.checkpoint import CheckpointError
from anemoi.training.checkpoint import CheckpointPipeline
from anemoi.training.checkpoint import PipelineStage


class MockStage(PipelineStage):
    """Mock pipeline stage for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.process_called = False
        self.context_received = None

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Process the context."""
        self.process_called = True
        self.context_received = context

        if self.should_fail:
            error_msg = f"Stage {self.name} failed"
            raise CheckpointError(error_msg)

        # Add marker to metadata
        context.update_metadata(**{f"stage_{self.name}": "processed"})
        return context


class TestCheckpointPipeline:
    """Test CheckpointPipeline class."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = CheckpointPipeline(stages)

        assert len(pipeline) == 2
        assert pipeline.stages == stages
        assert pipeline.async_execution is True
        assert pipeline.continue_on_error is False

    def test_pipeline_initialization_with_options(self) -> None:
        """Test pipeline initialization with options."""
        stages = [MockStage("stage1")]
        pipeline = CheckpointPipeline(stages, async_execution=False, continue_on_error=True)

        assert pipeline.async_execution is False
        assert pipeline.continue_on_error is True

    @pytest.mark.asyncio
    async def test_pipeline_execution_async(self) -> None:
        """Test async pipeline execution."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")
        stage3 = MockStage("stage3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3])

        context = CheckpointContext()
        result = await pipeline.execute_async(context)

        # Check all stages were called
        assert stage1.process_called
        assert stage2.process_called
        assert stage3.process_called

        # Check metadata was updated
        assert result.metadata["stage_stage1"] == "processed"
        assert result.metadata["stage_stage2"] == "processed"
        assert result.metadata["stage_stage3"] == "processed"

    def test_pipeline_execution_sync(self) -> None:
        """Test sync pipeline execution."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")

        pipeline = CheckpointPipeline([stage1, stage2])

        context = CheckpointContext()
        result = pipeline.execute_sync(context)

        assert stage1.process_called
        assert stage2.process_called
        assert "stage_stage1" in result.metadata
        assert "stage_stage2" in result.metadata

    @pytest.mark.asyncio
    async def test_pipeline_execution_with_error(self) -> None:
        """Test pipeline execution with error."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2", should_fail=True)
        stage3 = MockStage("stage3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3])

        context = CheckpointContext()

        with pytest.raises(CheckpointError) as exc_info:
            await pipeline.execute_async(context)

        assert "Stage stage2 failed" in str(exc_info.value)

        # Check first stage was called but third wasn't
        assert stage1.process_called
        assert stage2.process_called
        assert not stage3.process_called

    @pytest.mark.asyncio
    async def test_pipeline_continue_on_error(self) -> None:
        """Test pipeline continues on error when configured."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2", should_fail=True)
        stage3 = MockStage("stage3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3], continue_on_error=True)

        context = CheckpointContext()
        result = await pipeline.execute_async(context)

        # All stages should be called
        assert stage1.process_called
        assert stage2.process_called
        assert stage3.process_called

        # Check metadata shows failure
        assert "stage_1_MockStage" in result.metadata
        assert "failed" in result.metadata["stage_1_MockStage"]

    def test_add_stage(self) -> None:
        """Test adding stage to pipeline."""
        pipeline = CheckpointPipeline([])
        assert len(pipeline) == 0

        stage = MockStage("new_stage")
        pipeline.add_stage(stage)

        assert len(pipeline) == 1
        assert pipeline.stages[0] == stage

    def test_remove_stage(self) -> None:
        """Test removing stage from pipeline."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")

        pipeline = CheckpointPipeline([stage1, stage2])
        assert len(pipeline) == 2

        pipeline.remove_stage(stage1)
        assert len(pipeline) == 1
        assert pipeline.stages[0] == stage2

    def test_remove_nonexistent_stage(self) -> None:
        """Test removing non-existent stage."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")

        pipeline = CheckpointPipeline([stage1])

        # Should not raise, just log warning
        pipeline.remove_stage(stage2)
        assert len(pipeline) == 1

    def test_clear_stages(self) -> None:
        """Test clearing all stages."""
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = CheckpointPipeline(stages)

        assert len(pipeline) == 2

        pipeline.clear_stages()
        assert len(pipeline) == 0
        assert pipeline.stages == []

    def test_pipeline_repr(self) -> None:
        """Test pipeline string representation."""
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = CheckpointPipeline(stages, async_execution=False)

        repr_str = repr(pipeline)
        assert "CheckpointPipeline" in repr_str
        assert "MockStage" in repr_str
        assert "async=False" in repr_str

    @pytest.mark.asyncio
    async def test_empty_pipeline(self) -> None:
        """Test executing empty pipeline."""
        pipeline = CheckpointPipeline([])
        context = CheckpointContext()

        result = await pipeline.execute_async(context)

        # Should return context unchanged
        assert result == context

    @pytest.mark.asyncio
    async def test_context_passing(self) -> None:
        """Test context is passed correctly between stages."""

        class ModifyingStage(PipelineStage):
            def __init__(self, key: str, value: Any):
                self.key = key
                self.value = value

            async def process(self, context: CheckpointContext) -> CheckpointContext:
                context.update_metadata(**{self.key: self.value})
                return context

        stage1 = ModifyingStage("key1", "value1")
        stage2 = ModifyingStage("key2", "value2")
        stage3 = ModifyingStage("key3", "value3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3])

        context = CheckpointContext()
        result = await pipeline.execute_async(context)

        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == "value2"
        assert result.metadata["key3"] == "value3"

    def test_pipeline_mixed_stages(self) -> None:
        """Test pipeline with mix of instantiated stages.

        Note: We test with pre-instantiated stages only, since test modules
        are not installed as packages in CI environments. The Hydra instantiation
        logic is tested elsewhere.
        """
        stage1 = MockStage("instantiated")
        stage2 = MockStage("from_list")

        pipeline = CheckpointPipeline([stage1, stage2])

        assert len(pipeline) == 2
        assert pipeline.stages[0] == stage1
        assert isinstance(pipeline.stages[1], MockStage)
        assert pipeline.stages[1].name == "from_list"

    def test_add_stage_from_config(self) -> None:
        """Test adding stage to pipeline.

        Note: We test with pre-instantiated stages since test modules
        are not installed as packages in CI. Hydra instantiation with
        _target_ paths is tested with installed package classes.
        """
        pipeline = CheckpointPipeline([])

        stage = MockStage("added_stage")
        pipeline.add_stage(stage)

        assert len(pipeline) == 1
        assert isinstance(pipeline.stages[0], MockStage)
        assert pipeline.stages[0].name == "added_stage"

    @pytest.mark.asyncio
    async def test_hydra_configured_pipeline_execution(self) -> None:
        """Test executing pipeline created from config.

        Note: We test with pre-instantiated stages passed to the constructor
        since test modules are not installed as packages in CI. This still
        tests the full pipeline execution flow.
        """
        stage1 = MockStage("config_stage1")
        stage2 = MockStage("config_stage2")

        # Create pipeline directly with instantiated stages
        pipeline = CheckpointPipeline([stage1, stage2])
        context = CheckpointContext()

        result = await pipeline.execute(context)

        assert result.metadata["stage_config_stage1"] == "processed"
        assert result.metadata["stage_config_stage2"] == "processed"


class TestComposition:
    """Stage-composition validation: ordering/conflict violations raise at construction."""

    @staticmethod
    def _source() -> object:
        from anemoi.training.checkpoint.sources.local import LocalSource

        return LocalSource(path="model.ckpt")

    @staticmethod
    def _loader() -> object:
        from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

        return WeightsOnlyLoader()

    @staticmethod
    def _modifier() -> object:
        modifiers = pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing")
        return modifiers.FreezingModifierStage(submodules_to_freeze=[])

    def test_loader_before_source_raises(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match="comes after a loader"):
            CheckpointPipeline([self._loader(), self._source()])

    def test_two_loaders_raise(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match="loading strategies"):
            CheckpointPipeline([self._source(), self._loader(), self._loader()])

    def test_modifier_before_loader_raises(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match="comes before a loader"):
            CheckpointPipeline([self._source(), self._modifier(), self._loader()])

    def test_canonical_order_constructs(self) -> None:
        pipeline = CheckpointPipeline([self._source(), self._loader(), self._modifier()])
        assert len(pipeline) == 3

    def test_empty_pipeline_constructs(self) -> None:
        assert len(CheckpointPipeline([])) == 0

    def test_multiple_sources_warn_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            pipeline = CheckpointPipeline([self._source(), self._source(), self._loader()])
        assert len(pipeline) == 3
        assert any("Multiple checkpoint sources" in record.getMessage() for record in caplog.records)
