# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint pipeline orchestrator."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from anemoi.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointPipeline,
    PipelineStage,
)


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
            raise CheckpointError(f"Stage {self.name} failed")
        
        # Add marker to metadata
        context.update_metadata(**{f"stage_{self.name}": "processed"})
        return context


class TestCheckpointPipeline:
    """Test CheckpointPipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = CheckpointPipeline(stages)
        
        assert len(pipeline) == 2
        assert pipeline.stages == stages
        assert pipeline.async_execution is True
        assert pipeline.continue_on_error is False
    
    def test_pipeline_initialization_with_options(self):
        """Test pipeline initialization with options."""
        stages = [MockStage("stage1")]
        pipeline = CheckpointPipeline(
            stages,
            async_execution=False,
            continue_on_error=True
        )
        
        assert pipeline.async_execution is False
        assert pipeline.continue_on_error is True
    
    @pytest.mark.asyncio
    async def test_pipeline_execution_async(self):
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
    
    def test_pipeline_execution_sync(self):
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
    async def test_pipeline_execution_with_error(self):
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
    async def test_pipeline_continue_on_error(self):
        """Test pipeline continues on error when configured."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2", should_fail=True)
        stage3 = MockStage("stage3")
        
        pipeline = CheckpointPipeline(
            [stage1, stage2, stage3],
            continue_on_error=True
        )
        
        context = CheckpointContext()
        result = await pipeline.execute_async(context)
        
        # All stages should be called
        assert stage1.process_called
        assert stage2.process_called
        assert stage3.process_called
        
        # Check metadata shows failure
        assert "stage_1_MockStage" in result.metadata
        assert "failed" in result.metadata["stage_1_MockStage"]
    
    def test_add_stage(self):
        """Test adding stage to pipeline."""
        pipeline = CheckpointPipeline([])
        assert len(pipeline) == 0
        
        stage = MockStage("new_stage")
        pipeline.add_stage(stage)
        
        assert len(pipeline) == 1
        assert pipeline.stages[0] == stage
    
    def test_remove_stage(self):
        """Test removing stage from pipeline."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")
        
        pipeline = CheckpointPipeline([stage1, stage2])
        assert len(pipeline) == 2
        
        pipeline.remove_stage(stage1)
        assert len(pipeline) == 1
        assert pipeline.stages[0] == stage2
    
    def test_remove_nonexistent_stage(self):
        """Test removing non-existent stage."""
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")
        
        pipeline = CheckpointPipeline([stage1])
        
        # Should not raise, just log warning
        pipeline.remove_stage(stage2)
        assert len(pipeline) == 1
    
    def test_clear_stages(self):
        """Test clearing all stages."""
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = CheckpointPipeline(stages)
        
        assert len(pipeline) == 2
        
        pipeline.clear_stages()
        assert len(pipeline) == 0
        assert pipeline.stages == []
    
    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = CheckpointPipeline(stages, async_execution=False)
        
        repr_str = repr(pipeline)
        assert "CheckpointPipeline" in repr_str
        assert "MockStage" in repr_str
        assert "async=False" in repr_str
    
    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Test executing empty pipeline."""
        pipeline = CheckpointPipeline([])
        context = CheckpointContext()
        
        result = await pipeline.execute_async(context)
        
        # Should return context unchanged
        assert result == context
    
    @pytest.mark.asyncio
    async def test_context_passing(self):
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
    
    def test_pipeline_from_config(self):
        """Test creating pipeline from Hydra config."""
        config = OmegaConf.create({
            'stages': [
                {
                    '_target_': 'tests.checkpoint.test_pipeline.MockStage',
                    'name': 'stage1',
                    'should_fail': False
                },
                {
                    '_target_': 'tests.checkpoint.test_pipeline.MockStage',
                    'name': 'stage2',
                    'should_fail': False
                }
            ],
            'async_execution': False,
            'continue_on_error': True
        })
        
        pipeline = CheckpointPipeline.from_config(config)
        
        assert len(pipeline) == 2
        assert pipeline.async_execution is False
        assert pipeline.continue_on_error is True
        assert all(isinstance(stage, MockStage) for stage in pipeline.stages)
    
    def test_pipeline_mixed_stages(self):
        """Test pipeline with mix of instantiated and config stages."""
        instantiated_stage = MockStage("instantiated")
        config_stage = {
            '_target_': 'tests.checkpoint.test_pipeline.MockStage',
            'name': 'from_config',
            'should_fail': False
        }
        
        pipeline = CheckpointPipeline([instantiated_stage, config_stage])
        
        assert len(pipeline) == 2
        assert pipeline.stages[0] == instantiated_stage
        assert isinstance(pipeline.stages[1], MockStage)
        assert pipeline.stages[1].name == "from_config"
    
    def test_add_stage_from_config(self):
        """Test adding stage from Hydra config."""
        pipeline = CheckpointPipeline([])
        
        stage_config = {
            '_target_': 'tests.checkpoint.test_pipeline.MockStage',
            'name': 'added_stage',
            'should_fail': False
        }
        
        pipeline.add_stage(stage_config)
        
        assert len(pipeline) == 1
        assert isinstance(pipeline.stages[0], MockStage)
        assert pipeline.stages[0].name == "added_stage"
    
    @pytest.mark.asyncio
    async def test_hydra_configured_pipeline_execution(self):
        """Test executing pipeline created from Hydra config."""
        config = OmegaConf.create({
            'stages': [
                {
                    '_target_': 'tests.checkpoint.test_pipeline.MockStage',
                    'name': 'config_stage1',
                    'should_fail': False
                },
                {
                    '_target_': 'tests.checkpoint.test_pipeline.MockStage',
                    'name': 'config_stage2',
                    'should_fail': False
                }
            ]
        })
        
        pipeline = CheckpointPipeline.from_config(config)
        context = CheckpointContext()
        
        result = await pipeline.execute(context)
        
        assert result.metadata["stage_config_stage1"] == "processed"
        assert result.metadata["stage_config_stage2"] == "processed"