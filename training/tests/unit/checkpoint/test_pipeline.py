# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint pipeline orchestrator."""

from __future__ import annotations

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

    def process(self, context: CheckpointContext) -> CheckpointContext:
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

    def test_pipeline_execution(self) -> None:
        """Test pipeline execution."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")
        stage3 = MockStage("stage3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3])

        context = CheckpointContext()
        result = pipeline.execute(context)

        # Check all stages were called
        assert stage1.process_called
        assert stage2.process_called
        assert stage3.process_called

        # Check metadata was updated
        assert result.metadata["stage_stage1"] == "processed"
        assert result.metadata["stage_stage2"] == "processed"
        assert result.metadata["stage_stage3"] == "processed"

    def test_pipeline_execution_with_error(self) -> None:
        """Test pipeline execution with error."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2", should_fail=True)
        stage3 = MockStage("stage3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3])

        context = CheckpointContext()

        with pytest.raises(CheckpointError) as exc_info:
            pipeline.execute(context)

        assert "Stage stage2 failed" in str(exc_info.value)

        # Check first stage was called but third wasn't
        assert stage1.process_called
        assert stage2.process_called
        assert not stage3.process_called
        assert exc_info.value.details["pipeline_stage_index"] == 1

    def test_non_checkpoint_error_is_wrapped(self) -> None:
        """A stage error that is not a CheckpointError becomes a CheckpointError."""

        class BrokenStage(PipelineStage):
            def process(self, context: CheckpointContext) -> CheckpointContext:
                del context
                msg = "boom"
                raise ValueError(msg)

        with pytest.raises(CheckpointError, match="boom") as exc_info:
            CheckpointPipeline([BrokenStage()]).execute(CheckpointContext())
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_memory_error_is_not_wrapped(self) -> None:
        """A MemoryError propagates unchanged, so that the caller sees the real cause."""

        class OutOfMemoryStage(PipelineStage):
            def process(self, context: CheckpointContext) -> CheckpointContext:
                del context
                raise MemoryError

        with pytest.raises(MemoryError):
            CheckpointPipeline([OutOfMemoryStage()]).execute(CheckpointContext())

    def test_add_stage_instantiates_a_config(self) -> None:
        from anemoi.training.checkpoint.sources.local import LocalSource

        pipeline = CheckpointPipeline([])
        pipeline.add_stage({"_target_": "anemoi.training.checkpoint.sources.LocalSource", "path": "model.ckpt"})

        assert isinstance(pipeline.stages[0], LocalSource)

    @pytest.mark.parametrize("stage", [object(), {"_target_": "nonexistent.module.DoesNotExist"}])
    def test_add_stage_rejects_an_invalid_stage(self, stage: object) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError):
            CheckpointPipeline([]).add_stage(stage)

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
        pipeline = CheckpointPipeline(stages)

        assert repr(pipeline) == "CheckpointPipeline(stages=['MockStage', 'MockStage'])"

    def test_empty_pipeline(self) -> None:
        """Test executing empty pipeline."""
        pipeline = CheckpointPipeline([])
        context = CheckpointContext()

        result = pipeline.execute(context)

        # Should return context unchanged
        assert result == context

    def test_context_passing(self) -> None:
        """Test context is passed correctly between stages."""

        class ModifyingStage(PipelineStage):
            def __init__(self, key: str, value: Any):
                self.key = key
                self.value = value

            def process(self, context: CheckpointContext) -> CheckpointContext:
                context.update_metadata(**{self.key: self.value})
                return context

        stage1 = ModifyingStage("key1", "value1")
        stage2 = ModifyingStage("key2", "value2")
        stage3 = ModifyingStage("key3", "value3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3])

        context = CheckpointContext()
        result = pipeline.execute(context)

        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == "value2"
        assert result.metadata["key3"] == "value3"


class TestComposition:
    """Bad stage order or conflicts raise at construction."""

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
        from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage

        return FreezingModifierStage(submodules_to_freeze=[])

    def test_loader_before_source_raises(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match="order source"):
            CheckpointPipeline([self._loader(), self._source()])

    def test_two_loaders_raise(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match="one loader"):
            CheckpointPipeline([self._source(), self._loader(), self._loader()])

    def test_modifier_before_loader_raises(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match="order source"):
            CheckpointPipeline([self._source(), self._modifier(), self._loader()])

    def test_canonical_order_constructs(self) -> None:
        pipeline = CheckpointPipeline([self._source(), self._loader(), self._modifier()])
        assert len(pipeline) == 3

    def test_empty_pipeline_constructs(self) -> None:
        assert len(CheckpointPipeline([])) == 0
