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

from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint import CheckpointContext
from anemoi.training.checkpoint import CheckpointError
from anemoi.training.checkpoint import CheckpointPipeline
from anemoi.training.checkpoint import PipelineStage
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.base import LoadingStrategy
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.checkpoint.modifiers.base import ModelModifier
from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from pathlib import Path


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
        assert pipeline.continue_on_error is False

    def test_pipeline_initialization_with_options(self) -> None:
        """Test pipeline initialization with options."""
        stages = [MockStage("stage1")]
        pipeline = CheckpointPipeline(stages, continue_on_error=True)

        assert pipeline.continue_on_error is True

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

    def test_pipeline_execution_sync(self) -> None:
        """Test sync pipeline execution."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")

        pipeline = CheckpointPipeline([stage1, stage2])

        context = CheckpointContext()
        result = pipeline.execute(context)

        assert stage1.process_called
        assert stage2.process_called
        assert "stage_stage1" in result.metadata
        assert "stage_stage2" in result.metadata

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

    def test_pipeline_continue_on_error(self) -> None:
        """Test pipeline continues on error when configured."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2", should_fail=True)
        stage3 = MockStage("stage3")

        pipeline = CheckpointPipeline([stage1, stage2, stage3], continue_on_error=True)

        context = CheckpointContext()
        result = pipeline.execute(context)

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
        pipeline = CheckpointPipeline(stages, continue_on_error=True)

        repr_str = repr(pipeline)
        assert "CheckpointPipeline" in repr_str
        assert "MockStage" in repr_str
        assert "continue_on_error=True" in repr_str

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

    def test_hydra_configured_pipeline_execution(self) -> None:
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

        result = pipeline.execute(context)

        assert result.metadata["stage_config_stage1"] == "processed"
        assert result.metadata["stage_config_stage2"] == "processed"


class TestStageOrder:
    """The stage-order check: source -> one loader -> modifiers, raised at construction."""

    _ORDER_ERROR = "use the order source -> one loader -> modifiers"

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

    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param("source loader modifier modifier", id="canonical"),
            pytest.param("source loader", id="source-loader"),
            pytest.param("source", id="source-only"),
            pytest.param("loader modifier", id="loader-modifier"),
            pytest.param("modifier", id="modifier-only"),
            pytest.param("source source loader", id="two-sources"),
            pytest.param("", id="empty"),
        ],
    )
    def test_valid_orders_construct(self, layout: str) -> None:
        make = {"source": self._source, "loader": self._loader, "modifier": self._modifier}
        stages = [make[kind]() for kind in layout.split()]
        assert len(CheckpointPipeline(stages)) == len(stages)

    def test_stage_of_no_known_layer_is_not_placed(self) -> None:
        stages = [MockStage("custom"), self._source(), self._loader(), MockStage("other"), self._modifier()]
        assert len(CheckpointPipeline(stages)) == 5

    def test_source_after_loader_raises(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match=self._ORDER_ERROR):
            CheckpointPipeline([self._loader(), self._source()])

    def test_modifier_before_loader_raises(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match=self._ORDER_ERROR):
            CheckpointPipeline([self._source(), self._modifier(), self._loader()])

    def test_two_loaders_raise(self) -> None:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        with pytest.raises(CheckpointConfigError, match=self._ORDER_ERROR) as exc_info:
            CheckpointPipeline([self._source(), self._loader(), self._loader()])
        assert "WeightsOnlyLoader" in str(exc_info.value)


# --- a source with no loader is refused -------------------------------------------


def _write_checkpoint(tmp_path: Path) -> Path:
    import torch

    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}}, ckpt)
    return ckpt


def test_verify_weights_loaded_still_rejects_a_bare_source_without_loader(tmp_path: Path) -> None:
    """Positive control: a source that loaded data with no loader to apply it is still refused."""
    import torch

    from anemoi.training.checkpoint.exceptions import CheckpointLoadError
    from anemoi.training.checkpoint.sources.local import LocalSource

    ckpt = _write_checkpoint(tmp_path)
    pipeline = CheckpointPipeline([LocalSource(path=ckpt)])

    with pytest.raises(CheckpointLoadError, match="Refusing to proceed with random weights"):
        pipeline.execute(CheckpointContext(model=torch.nn.Linear(2, 2)))


# --- stage construction and error propagation ---------------------------------------


class _MemoryErrorStage(PipelineStage):
    """Stage whose process raises MemoryError."""

    def process(self, context: CheckpointContext) -> CheckpointContext:  # noqa: ARG002
        msg = "out of memory"
        raise MemoryError(msg)


def test_unimportable_stage_target_fails_at_construction() -> None:
    """A stage config whose ``_target_`` cannot be imported names the failing stage index."""
    with pytest.raises(CheckpointConfigError, match="Failed to instantiate pipeline stage 0"):
        CheckpointPipeline([{"_target_": "nonexistent.module.Class", "param": "value"}])


def test_memory_error_propagates_even_with_continue_on_error() -> None:
    """continue_on_error swallows checkpoint errors, never a MemoryError."""
    pipeline = CheckpointPipeline([_MemoryErrorStage()], continue_on_error=True)

    with pytest.raises(MemoryError):
        pipeline.execute(CheckpointContext(model=nn.Linear(2, 2)))


# --- stage roles and executed cross-layer pipelines ----------------------------------


class _TwoChildModel(nn.Module):
    """Two named children, each an ``nn.Linear``."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 3)


class _MismatchTargetModel(nn.Module):
    """Target whose ``encoder`` shape differs from the source checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(8, 4)


class _InMemorySource(CheckpointSource):
    """A source that attaches in-memory checkpoint data."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        context.checkpoint_data = {"state_dict": {}}
        return context


class _WeightMarkingLoader(LoadingStrategy):
    """A loader that only marks the model weights as loaded."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        if context.model is not None:
            self._mark_weights_loaded(context.model)
        return context


class _NoOpStage(PipelineStage):
    """A bare stage that neither loads weights nor belongs to any layer."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _CustomModifier(ModelModifier):
    """An inline modifier that freezes the whole model and records itself."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        for param in context.model.parameters():
            param.requires_grad = False
        context.metadata.setdefault("modifiers_applied", []).append({"type": "custom_recording"})
        return context


class _CustomRestorer(CheckpointSource):
    """A source whose class name does not contain 'Source'."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _Restorer(LoadingStrategy):
    """A loader whose class name does not contain 'Loader'."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


class _LoRAAdapter(ModelModifier):
    """A modifier whose class name does not contain 'Modifier' or 'Freez'."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_source_without_a_loader_is_refused_even_with_continue_on_error(continue_on_error: bool) -> None:
    """Checkpoint data that no loader applied stops the pipeline whatever continue_on_error says."""
    pipeline = CheckpointPipeline([_InMemorySource(), _NoOpStage()], continue_on_error=continue_on_error)

    with pytest.raises(CheckpointLoadError):
        pipeline.execute(CheckpointContext(model=_TwoChildModel()))


@pytest.mark.parametrize(
    ("stage", "role"),
    [(_CustomRestorer(), "source"), (_Restorer(), "loader"), (_LoRAAdapter(), "modifier")],
    ids=["source", "loader", "modifier"],
)
def test_stage_role_comes_from_the_base_class_not_the_name(stage: PipelineStage, role: str) -> None:
    assert CheckpointPipeline._stage_role(stage) == role


def test_custom_inline_modifier_runs_after_the_loader() -> None:
    """A third-party ModelModifier subclass runs as a modifier stage in an executed pipeline."""
    modifier = _CustomModifier()
    pipeline = CheckpointPipeline([_InMemorySource(), _WeightMarkingLoader(), modifier])

    result = pipeline.execute(CheckpointContext(model=_TwoChildModel()))

    assert all(not p.requires_grad for p in result.model.parameters())
    assert {"type": "custom_recording"} in result.metadata["modifiers_applied"]
    assert CheckpointPipeline._stage_role(modifier) == "modifier"


def test_transfer_learning_then_freezing_compose_in_one_pipeline(tmp_path: Path) -> None:
    """A transfer-learning load followed by freezing keeps both layers' metadata and effects."""
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

    result = pipeline.execute(CheckpointContext(model=_MismatchTargetModel()))

    assert all(not p.requires_grad for p in result.model.encoder.parameters())
    assert result.metadata["loading_strategy"] == "transfer_learning"
    assert "decoder.weight" in result.metadata["transferred_params"]
    assert "Shape mismatch" in result.metadata["skipped_params"]["encoder.weight"]
    freezing_records = [entry for entry in result.metadata["modifiers_applied"] if entry["type"] == "freezing"]
    assert freezing_records
    assert freezing_records[0]["frozen_modules"][0]["name"] == "encoder"
