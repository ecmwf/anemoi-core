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

        # Smart warnings about pipeline configuration
        self._validate_pipeline_composition()

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
                except Exception as e:
                    from .exceptions import CheckpointConfigError

                    stage_config_preview = str(stage)[:200] + "..." if len(str(stage)) > 200 else str(stage)
                    error_msg = (
                        f"Failed to instantiate pipeline stage {i} from configuration.\n"
                        f"Stage config: {stage_config_preview}\n"
                        f"Original error: {e}\n"
                        "Suggestions:\n"
                        "  • Check that the '_target_' path is correct and importable\n"
                        "  • Verify all required parameters are provided\n"
                        "  • Ensure the target class is a valid PipelineStage subclass"
                    )
                    LOGGER.exception("Failed to instantiate stage %d from config", i)
                    raise CheckpointConfigError(error_msg, config_path=f"stages[{i}]") from e
            else:
                # Already instantiated
                instantiated.append(stage)
        return instantiated

    def _validate_pipeline_composition(self) -> None:
        """Validate pipeline composition and provide smart warnings."""
        if not self.stages:
            LOGGER.warning(
                "Pipeline has no stages configured. "
                "Consider adding stages for source acquisition, loading, or model modification.",
            )
            return

        stage_types = [stage.__class__.__name__ for stage in self.stages]

        # Check for common composition issues
        self._check_source_loading_order(stage_types)
        self._check_modifier_placement(stage_types)
        self._check_duplicate_stages(stage_types)
        self._suggest_missing_stages(stage_types)

    def _check_source_loading_order(self, stage_types: list[str]) -> None:
        """Check that source stages come before loading stages."""
        source_indices = [i for i, name in enumerate(stage_types) if "Source" in name]
        loader_indices = [i for i, name in enumerate(stage_types) if "Loader" in name or "Loading" in name]

        if source_indices and loader_indices:
            last_source = max(source_indices)
            first_loader = min(loader_indices)

            if last_source > first_loader:
                LOGGER.warning(
                    "Source stage at position %d comes after loader stage at position %d. "
                    "This may cause issues as loaders typically expect checkpoints to be already acquired. "
                    "Consider reordering: sources first, then loaders, then modifiers.",
                    last_source,
                    first_loader,
                )

    def _check_modifier_placement(self, stage_types: list[str]) -> None:
        """Check that modifier stages come after loading stages."""
        loader_indices = [i for i, name in enumerate(stage_types) if "Loader" in name or "Loading" in name]
        modifier_indices = [i for i, name in enumerate(stage_types) if "Modifier" in name]

        if loader_indices and modifier_indices:
            last_loader = max(loader_indices)
            first_modifier = min(modifier_indices)

            if first_modifier < last_loader:
                LOGGER.warning(
                    "Modifier stage at position %d comes before loader stage at position %d. "
                    "Model modifications typically work best after checkpoint loading. "
                    "Consider reordering: sources, loaders, then modifiers.",
                    first_modifier,
                    last_loader,
                )

    def _check_duplicate_stages(self, stage_types: list[str]) -> None:
        """Check for potentially conflicting duplicate stages."""
        from collections import Counter

        type_counts = Counter(stage_types)
        duplicates = {stage_type: count for stage_type, count in type_counts.items() if count > 1}

        if duplicates:
            for stage_type, count in duplicates.items():
                if "Loader" in stage_type or "Loading" in stage_type:
                    LOGGER.warning(
                        "Found %d instances of %s. Multiple loading strategies may conflict. "
                        "Consider using a single loading strategy appropriate for your use case.",
                        count,
                        stage_type,
                    )
                elif "Source" in stage_type and stage_type.endswith("Source"):
                    LOGGER.warning(
                        "Found %d instances of %s. Multiple checkpoint sources may be redundant. "
                        "The pipeline will process them in sequence.",
                        count,
                        stage_type,
                    )

    def _suggest_missing_stages(self, stage_types: list[str]) -> None:
        """Suggest potentially missing stages based on common patterns."""
        has_source = any("Source" in name for name in stage_types)
        has_loader = any("Loader" in name or "Loading" in name for name in stage_types)
        has_modifier = any("Modifier" in name for name in stage_types)

        suggestions = []

        if has_loader and not has_source:
            suggestions.append(
                "You have a loading stage but no source stage. "
                "Consider adding a source stage (LocalSource, S3Source, etc.) to specify where to load from.",
            )

        if has_source and not has_loader:
            suggestions.append(
                "You have a source stage but no loading strategy. "
                "Consider adding a loading stage (WeightsOnlyLoader, TransferLearningLoader, etc.) "
                "to specify how to apply the checkpoint.",
            )

        if has_modifier and not (has_source or has_loader):
            suggestions.append(
                "You have model modifiers but no checkpoint loading. "
                "Modifiers work best when applied after loading a checkpoint. "
                "Consider adding source and loading stages.",
            )

        if suggestions:
            LOGGER.info("Pipeline composition suggestions:\n  • %s", "\n  • ".join(suggestions))

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

        # Perform environment health check before pipeline execution
        self._perform_pre_execution_validation(context)

        for i, stage in enumerate(self.stages):
            stage_name = stage.__class__.__name__
            LOGGER.debug("Executing stage %d/%d: %s", i, len(self.stages), stage_name)

            try:
                context = await stage.process(context)
                LOGGER.debug("Stage %s completed successfully", stage_name)

                # Update metadata with stage execution
                context.update_metadata(**{f"stage_{i}_{stage_name}": "completed"})

            except Exception as e:
                from .exceptions import CheckpointError

                # Enhance error with pipeline context
                stage_info = f"Pipeline stage {i + 1}/{len(self.stages)} ({stage_name})"

                # If it's already a CheckpointError, add pipeline context to details
                if isinstance(e, CheckpointError):
                    if hasattr(e, "details") and e.details:
                        e.details["pipeline_stage"] = stage_info
                        e.details["pipeline_stage_index"] = i
                        e.details["pipeline_total_stages"] = len(self.stages)
                    LOGGER.exception("Pipeline stage %s failed", stage_info)
                    context.update_metadata(**{f"stage_{i}_{stage_name}": f"failed: {e!s}"})

                    if not self.continue_on_error:
                        raise
                else:
                    # Wrap non-CheckpointError exceptions with context
                    error_msg = f"{stage_info} failed: {e}"
                    context.update_metadata(**{f"stage_{i}_{stage_name}": f"failed: {e!s}"})
                    LOGGER.exception("Pipeline stage %s failed", stage_info)

                    if not self.continue_on_error:
                        wrapped_error = CheckpointError(
                            error_msg,
                            {
                                "pipeline_stage": stage_info,
                                "pipeline_stage_index": i,
                                "pipeline_total_stages": len(self.stages),
                                "original_error": str(e),
                                "original_error_type": type(e).__name__,
                            },
                        )
                        raise wrapped_error from e

                LOGGER.warning("Continuing pipeline despite error in %s", stage_info)

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
            try:
                stage = instantiate(stage)
            except Exception as e:
                from .exceptions import CheckpointConfigError

                stage_config_preview = str(stage)[:200] + "..." if len(str(stage)) > 200 else str(stage)
                error_msg = (
                    f"Failed to add stage to pipeline: cannot instantiate from configuration.\n"
                    f"Stage config: {stage_config_preview}\n"
                    f"Original error: {e}\n"
                    "Suggestions:\n"
                    "  • Check that the '_target_' path is correct and importable\n"
                    "  • Verify all required parameters are provided\n"
                    "  • Ensure the target class is a valid PipelineStage subclass"
                )
                raise CheckpointConfigError(error_msg, config_path="add_stage") from e

        # Validate that it's actually a PipelineStage
        if not hasattr(stage, "process") or not callable(stage.process):
            from .exceptions import CheckpointConfigError

            error_message = (
                f"Invalid stage type: {type(stage).__name__}. "
                "Stages must be PipelineStage instances with a 'process' method. "
                f"Got: {stage}"
            )
            raise CheckpointConfigError(error_message)

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

    def _perform_pre_execution_validation(self, context: CheckpointContext) -> None:
        """Perform pre-execution health checks and validations.

        This method runs lightweight validations before pipeline execution
        to catch common configuration and environment issues early.

        Parameters
        ----------
        context : CheckpointContext
            The context about to be processed
        """
        try:
            from .validation import CheckpointPipelineValidator

            # Perform environment validation
            env_results = CheckpointPipelineValidator.validate_environment_setup()

            # Log any issues found
            for issue in env_results.get("issues", []):
                LOGGER.error("Environment issue: %s", issue)

            for warning in env_results.get("warnings", []):
                LOGGER.warning("Environment warning: %s", warning)

            for info in env_results.get("info", []):
                LOGGER.debug("Environment info: %s", info)

            # Validate configuration if provided
            if context.config and hasattr(context.config, "training"):
                config_results = CheckpointPipelineValidator.validate_configuration(context.config)

                # Log configuration issues
                for issue in config_results.get("issues", []):
                    LOGGER.error("Configuration issue: %s", issue)

                for warning in config_results.get("warnings", []):
                    LOGGER.warning("Configuration warning: %s", warning)

                # Add validation results to context metadata
                context.update_metadata(
                    validation_environment_status=env_results.get("status", "unknown"),
                    validation_config_status=config_results.get("status", "unknown"),
                    validation_performed=True,
                )
            else:
                context.update_metadata(
                    validation_environment_status=env_results.get("status", "unknown"),
                    validation_performed=True,
                )

        except ImportError:
            # Validation module not available - this is ok, validation is optional
            LOGGER.debug("Validation module not available, skipping pre-execution validation")
            context.update_metadata(validation_performed=False, validation_skipped="module_unavailable")
        except (ValueError, TypeError, AttributeError) as e:
            # Don't fail pipeline execution due to validation errors
            LOGGER.warning("Pre-execution validation failed: %s", e)
            context.update_metadata(validation_performed=False, validation_error=str(e))
