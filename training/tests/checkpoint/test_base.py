# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint base classes."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from anemoi.training.checkpoint import CheckpointContext
from anemoi.training.checkpoint import PipelineStage


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointContext:
    """Test CheckpointContext dataclass."""

    def test_context_initialization_empty(self):
        """Test context initialization with no arguments."""
        context = CheckpointContext()

        assert context.checkpoint_path is None
        assert context.checkpoint_data is None
        assert context.model is None
        assert context.optimizer is None
        assert context.scheduler is None
        assert context.metadata == {}
        assert context.config is None

    def test_context_initialization_with_values(self):
        """Test context initialization with values."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = DictConfig({"key": "value"})

        context = CheckpointContext(
            checkpoint_path=Path("/tmp/checkpoint.pt"),
            checkpoint_data={"epoch": 10},
            model=model,
            optimizer=optimizer,
            metadata={"training": True},
            config=config,
        )

        assert context.checkpoint_path == Path("/tmp/checkpoint.pt")
        assert context.checkpoint_data == {"epoch": 10}
        assert context.model == model
        assert context.optimizer == optimizer
        assert context.metadata == {"training": True}
        assert context.config == config

    def test_context_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        context = CheckpointContext(checkpoint_path="/tmp/checkpoint.pt")

        assert isinstance(context.checkpoint_path, Path)
        assert context.checkpoint_path == Path("/tmp/checkpoint.pt")

    def test_update_metadata(self):
        """Test updating metadata."""
        context = CheckpointContext()

        context.update_metadata(epoch=5, loss=0.1)
        assert context.metadata == {"epoch": 5, "loss": 0.1}

        context.update_metadata(epoch=10)  # Update existing
        assert context.metadata == {"epoch": 10, "loss": 0.1}

        context.update_metadata(accuracy=0.95)  # Add new
        assert context.metadata == {"epoch": 10, "loss": 0.1, "accuracy": 0.95}

    def test_get_metadata(self):
        """Test getting metadata with defaults."""
        context = CheckpointContext(metadata={"epoch": 5})

        assert context.get_metadata("epoch") == 5
        assert context.get_metadata("missing") is None
        assert context.get_metadata("missing", "default") == "default"

    def test_has_checkpoint_data(self):
        """Test checking for checkpoint data."""
        context = CheckpointContext()
        assert not context.has_checkpoint_data()

        context.checkpoint_data = {}
        assert not context.has_checkpoint_data()  # Empty dict

        context.checkpoint_data = {"state_dict": {}}
        assert context.has_checkpoint_data()

    def test_context_repr(self):
        """Test context string representation."""
        context = CheckpointContext(
            checkpoint_path=Path("/tmp/checkpoint.pt"), model=SimpleModel(), metadata={"epoch": 5, "loss": 0.1},
        )

        repr_str = repr(context)

        assert "CheckpointContext" in repr_str
        assert "path=checkpoint.pt" in repr_str
        assert "model=SimpleModel" in repr_str
        assert "metadata_keys=['epoch', 'loss']" in repr_str

    def test_context_fields_mutable(self):
        """Test that context fields are mutable."""
        context = CheckpointContext()

        # Test mutability
        model = SimpleModel()
        context.model = model
        assert context.model == model

        context.checkpoint_path = Path("/new/path.pt")
        assert context.checkpoint_path == Path("/new/path.pt")

        context.metadata["new_key"] = "new_value"
        assert context.metadata["new_key"] == "new_value"


class TestPipelineStage:
    """Test PipelineStage abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            PipelineStage()

        assert "Can't instantiate abstract class" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concrete_implementation(self):
        """Test concrete implementation of PipelineStage."""

        class ConcreteStage(PipelineStage):
            async def process(self, context: CheckpointContext) -> CheckpointContext:
                context.update_metadata(processed=True)
                return context

        stage = ConcreteStage()
        context = CheckpointContext()

        result = await stage.process(context)

        assert result.metadata["processed"] is True

    @pytest.mark.asyncio
    async def test_stage_callable(self):
        """Test that stage can be called directly."""

        class CallableStage(PipelineStage):
            async def process(self, context: CheckpointContext) -> CheckpointContext:
                context.update_metadata(called=True)
                return context

        stage = CallableStage()
        context = CheckpointContext()

        # Call using __call__
        result = await stage(context)

        assert result.metadata["called"] is True

    def test_stage_repr(self):
        """Test stage string representation."""

        class MyCustomStage(PipelineStage):
            async def process(self, context: CheckpointContext) -> CheckpointContext:
                return context

        stage = MyCustomStage()
        repr_str = repr(stage)

        assert repr_str == "MyCustomStage()"

    @pytest.mark.asyncio
    async def test_stage_error_propagation(self):
        """Test that errors are propagated from process method."""

        class ErrorStage(PipelineStage):
            async def process(self, context: CheckpointContext) -> CheckpointContext:
                raise ValueError("Test error")

        stage = ErrorStage()
        context = CheckpointContext()

        with pytest.raises(ValueError) as exc_info:
            await stage.process(context)

        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stage_modifies_context(self):
        """Test that stages can modify context in place."""

        class ModifyingStage(PipelineStage):
            async def process(self, context: CheckpointContext) -> CheckpointContext:
                # Modify in place
                context.metadata["modified"] = True
                # Also add new fields
                context.checkpoint_path = Path("/modified/path.pt")
                return context

        stage = ModifyingStage()
        context = CheckpointContext()

        result = await stage.process(context)

        # Check both return value and original context
        assert result.metadata["modified"] is True
        assert result.checkpoint_path == Path("/modified/path.pt")
        assert context == result  # Should be same object
