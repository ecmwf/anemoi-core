# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pipeline orchestrator for checkpoint processing.

This module provides the CheckpointPipeline class that orchestrates
the execution of multiple pipeline stages in sequence. It handles:
- Stage execution order
- Error propagation and recovery
- Async/sync execution modes
- Metadata tracking through stages
- Hydra-based configuration and instantiation

The pipeline pattern allows for flexible composition of checkpoint
processing operations, making it easy to build complex workflows
from simple, reusable components.

Example
-------
>>> from anemoi.training.checkpoint import CheckpointPipeline, CheckpointContext
>>> from anemoi.training.checkpoint.sources import LocalSource
>>> from anemoi.training.checkpoint.loaders import WeightsOnlyLoader
>>>
>>> # Build a pipeline manually
>>> pipeline = CheckpointPipeline([
...     LocalSource(path='/tmp/checkpoint.pt'),
...     WeightsOnlyLoader(strict=False)
... ])
>>>
>>> # Or build from Hydra config
>>> from omegaconf import OmegaConf
>>> config = OmegaConf.create({
...     'stages': [
...         {'_target_': 'anemoi.training.checkpoint.sources.LocalSource',
...          'path': '/tmp/checkpoint.pt'},
...         {'_target_': 'anemoi.training.checkpoint.loaders.WeightsOnlyLoader',
...          'strict': False}
...     ]
... })
>>> pipeline = CheckpointPipeline.from_config(config)
>>>
>>> # Execute pipeline
>>> context = CheckpointContext(model=my_model)
>>> result = await pipeline.execute(context)
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

from hydra.utils import instantiate
from omegaconf import DictConfig

if TYPE_CHECKING:
    from .base import CheckpointContext
    from .base import PipelineStage

LOGGER = logging.getLogger(__name__)


class CheckpointPipeline:
    """Orchestrates checkpoint processing through stages.

    The pipeline executes a series of stages in order, passing a
    CheckpointContext through each stage. Each stage can modify
    the context before passing it to the next stage. This creates
    a processing chain where each stage builds upon the work of
    previous stages.

    The pipeline supports:
    - Sequential execution of stages
    - Error handling with optional continuation
    - Async and sync execution modes
    - Dynamic stage management (add/remove)
    - Metadata tracking for debugging

    Parameters
    ----------
    stages : list of PipelineStage, dict, or DictConfig, optional
        List of pipeline stages to execute in order. Each item can be:
        - An instantiated PipelineStage object
        - A dict/DictConfig with '_target_' for Hydra instantiation
    async_execution : bool, optional
        Whether to use async execution (default: True). Set to False
        for synchronous execution in non-async contexts.
    continue_on_error : bool, optional
        Whether to continue pipeline on stage errors (default: False).
        If True, failed stages will be logged but won't stop the pipeline.
    config : DictConfig, optional
        If provided, overrides other parameters with config values

    Attributes
    ----------
    stages : list
        Current list of pipeline stages
    async_execution : bool
        Whether async execution is enabled
    continue_on_error : bool
        Whether to continue on stage errors

    Examples
    --------
    >>> # Simple pipeline
    >>> pipeline = CheckpointPipeline([
    ...     FetchStage(),
    ...     ValidateStage(),
    ...     LoadStage()
    ... ])
    >>>
    >>> # Pipeline with error handling
    >>> pipeline = CheckpointPipeline(
    ...     stages=[Stage1(), Stage2()],
    ...     continue_on_error=True  # Don't stop on failures
    ... )
    """

    def __init__(
        self,
        stages: list[Union[PipelineStage, DictConfig, dict]] | None = None,
        async_execution: bool = True,
        continue_on_error: bool = False,
        config: DictConfig | None = None,
    ):
        """Initialize the checkpoint pipeline.

        Parameters
        ----------
        stages : list of PipelineStage or dict/DictConfig, optional
            List of pipeline stages to execute in order. Each item can be:
            - An instantiated PipelineStage object
            - A dict/DictConfig with '_target_' for Hydra instantiation
        async_execution : bool, optional
            Whether to use async execution (default: True)
        continue_on_error : bool, optional
            Whether to continue pipeline on stage errors (default: False)
        config : DictConfig, optional
            If provided, overrides other parameters with config values
        """
        # Extract settings from config if provided
        if config is not None:
            self.async_execution = config.get("async_execution", async_execution)
            self.continue_on_error = config.get("continue_on_error", continue_on_error)
            stages = config.get("stages", stages)
        else:
            self.async_execution = async_execution
            self.continue_on_error = continue_on_error

        # Instantiate stages
        self.stages = self._instantiate_stages(stages or [])

        LOGGER.info("Initialized pipeline with %d stages", len(self.stages))
        for i, stage in enumerate(self.stages):
            LOGGER.debug("  Stage %d: %s", i, stage)

    def _instantiate_stages(self, stages: list[Any]) -> list[PipelineStage]:
        """Instantiate stages from configs or pass through existing instances.

        Parameters
        ----------
        stages : list
            List of either PipelineStage instances or configs with '_target_'

        Returns
        -------
        list of PipelineStage
            List of instantiated pipeline stages
        """
        instantiated = []
        for i, stage in enumerate(stages):
            if isinstance(stage, (dict, DictConfig)):
                # Use Hydra to instantiate from config
                try:
                    instantiated_stage = instantiate(stage)
                    instantiated.append(instantiated_stage)
                    LOGGER.debug("Instantiated stage %d from config: %s", i, instantiated_stage)
                except Exception:
                    LOGGER.exception("Failed to instantiate stage %d from config", i)
                    raise
            else:
                # Already instantiated
                instantiated.append(stage)
        return instantiated

    @classmethod
    def from_config(cls, config: DictConfig) -> CheckpointPipeline:
        """Create a pipeline from Hydra configuration.

        This is a convenience method that creates a pipeline entirely
        from a Hydra configuration, using instantiate for all stages.

        Parameters
        ----------
        config : DictConfig
            Hydra configuration with pipeline settings. Should contain:
            - stages: List of stage configs with '_target_'
            - async_execution: Optional bool for async mode
            - continue_on_error: Optional bool for error handling

        Returns
        -------
        CheckpointPipeline
            Configured pipeline instance

        Examples
        --------
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({
        ...     'stages': [
        ...         {'_target_': 'path.to.SourceStage', 'param': 'value'},
        ...         {'_target_': 'path.to.LoaderStage', 'strict': False}
        ...     ],
        ...     'async_execution': True,
        ...     'continue_on_error': False
        ... })
        >>> pipeline = CheckpointPipeline.from_config(config)
        """
        return cls(config=config)

    async def execute_async(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Execute pipeline stages asynchronously.

        Executes each stage in sequence, passing the context from one
        stage to the next. Each stage's execution is tracked in metadata
        for debugging and monitoring.

        Parameters
        ----------
        initial_context : CheckpointContext
            Initial context to process. This should contain any initial
            state needed by the first stage (e.g., model, config).

        Returns
        -------
        CheckpointContext
            Final processed context containing the accumulated results
            from all stages.

        Raises
        ------
        CheckpointError
            If a stage fails and continue_on_error is False.
            The error will contain information about which stage failed.

        Notes
        -----
        Stage execution is tracked in the context metadata with keys like:
        - 'stage_0_StageName': 'completed' or 'failed: error message'

        This allows for debugging pipeline execution and understanding
        which stages were executed and their results.
        """
        context = initial_context

        for i, stage in enumerate(self.stages):
            stage_name = stage.__class__.__name__
            LOGGER.debug("Executing stage %d/%d: %s", i, len(self.stages), stage_name)

            try:
                context = await stage.process(context)
                LOGGER.debug("Stage %s completed successfully", stage_name)

                # Update metadata with stage execution
                context.update_metadata(**{f"stage_{i}_{stage_name}": "completed"})

            except Exception as e:
                LOGGER.exception("Stage %s failed", stage_name)
                context.update_metadata(**{f"stage_{i}_{stage_name}": f"failed: {e!s}"})

                if not self.continue_on_error:
                    raise

                LOGGER.warning("Continuing pipeline despite error in %s", stage_name)

        LOGGER.info("Pipeline execution completed")
        return context

    def execute_sync(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Execute pipeline stages synchronously.

        This is a convenience method for synchronous execution,
        wrapping the async execution in asyncio.run().

        Parameters
        ----------
        initial_context : CheckpointContext
            Initial context to process

        Returns
        -------
        CheckpointContext
            Final processed context
        """
        return asyncio.run(self.execute_async(initial_context))

    async def execute(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Execute the pipeline.

        Main entry point for pipeline execution. Uses async or sync
        execution based on the async_execution flag. This method can
        be called from both async and sync contexts.

        Parameters
        ----------
        initial_context : CheckpointContext
            Initial context to process. Should contain:
            - model: The PyTorch model to load checkpoint into
            - config: Optional configuration for stages
            - Any other initial state needed by stages

        Returns
        -------
        CheckpointContext
            Final processed context containing:
            - checkpoint_path: Path to downloaded checkpoint (if applicable)
            - checkpoint_data: Loaded checkpoint data
            - model: Modified model with loaded weights
            - metadata: Execution tracking and stage results

        Examples
        --------
        >>> import asyncio
        >>> context = CheckpointContext(model=my_model)
        >>>
        >>> # In async context:
        >>> result = await pipeline.execute(context)
        >>>
        >>> # In sync context:
        >>> result = asyncio.run(pipeline.execute(context))
        """
        if self.async_execution:
            return await self.execute_async(initial_context)
        # For sync execution in async context
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_sync, initial_context)

    def add_stage(self, stage: Union[PipelineStage, DictConfig, dict]) -> None:
        """Add a stage to the pipeline.

        Parameters
        ----------
        stage : PipelineStage or dict/DictConfig
            Stage to add to the pipeline. Can be:
            - An instantiated PipelineStage object
            - A dict/DictConfig with '_target_' for Hydra instantiation

        Examples
        --------
        >>> # Add instantiated stage
        >>> pipeline.add_stage(MyStage())
        >>>
        >>> # Add from config
        >>> pipeline.add_stage({
        ...     '_target_': 'path.to.MyStage',
        ...     'param': 'value'
        ... })
        """
        if isinstance(stage, (dict, DictConfig)):
            stage = instantiate(stage)
        self.stages.append(stage)
        LOGGER.debug("Added stage %s to pipeline", stage)

    def remove_stage(self, stage: PipelineStage) -> None:
        """Remove a stage from the pipeline.

        Parameters
        ----------
        stage : PipelineStage
            Stage to remove from the pipeline
        """
        if stage in self.stages:
            self.stages.remove(stage)
            LOGGER.debug("Removed stage %s from pipeline", stage)
        else:
            LOGGER.warning("Stage %s not found in pipeline", stage)

    def clear_stages(self) -> None:
        """Clear all stages from the pipeline."""
        self.stages.clear()
        LOGGER.debug("Cleared all stages from pipeline")

    def __len__(self) -> int:
        """Return the number of stages in the pipeline.

        Returns
        -------
        int
            Number of stages
        """
        return len(self.stages)

    def __repr__(self) -> str:
        """String representation of the pipeline.

        Provides a readable representation showing the stages and
        execution mode for debugging and logging.

        Returns
        -------
        str
            String representation showing stage names and settings

        Examples
        --------
        >>> pipeline = CheckpointPipeline([Stage1(), Stage2()])
        >>> print(pipeline)
        CheckpointPipeline(stages=['Stage1', 'Stage2'], async=True)
        """
        stage_names = [s.__class__.__name__ for s in self.stages]
        return f"CheckpointPipeline(stages={stage_names}, async={self.async_execution})"
