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
>>> from anemoi.training.checkpoint.loading import WeightsOnlyLoader
>>>
>>> # Build a pipeline manually
>>> pipeline = CheckpointPipeline([
...     LocalSource(path='/tmp/checkpoint.pt'),
...     WeightsOnlyLoader(strict=False),
... ])
>>>
>>> # Or, in a training run, build it from the declarative config. The trainer
>>> # reads ``training.checkpoint.{source,loading,modifiers}`` and assembles the
>>> # stages in that fixed order:
>>> from anemoi.training.checkpoint import build_checkpoint_pipeline
>>> pipeline = build_checkpoint_pipeline(cfg)  # cfg.training.checkpoint.*
>>>
>>> # Execute pipeline
>>> context = CheckpointContext(model=my_model)
>>> result = await pipeline.execute(context)

Execution Patterns
------------------
The pipeline supports two execution patterns:

**Pattern 1: Standalone Execution (Recommended)**

Execute during model initialization, before ``trainer.fit()``. This is the
recommended approach as checkpoint loading happens once at startup::

    # In your training script or AnemoiTrainer.model property
    pipeline = build_checkpoint_pipeline(config)  # reads training.checkpoint.*
    context = CheckpointContext(model=model)

    # Async execution (recommended for remote sources)
    result = await pipeline.execute(context)
    # Or sync execution
    result = asyncio.run(pipeline.execute(context))

    model = result.model

**Pattern 2: PyTorch Lightning Callback Integration**

For use cases requiring Lightning callback lifecycle integration, the
pipeline can be wrapped in a callback. This is useful when checkpoint
loading needs to coordinate with other Lightning callbacks::

    # See anemoi.training.diagnostics.callbacks.checkpoint for examples
    # of integrating checkpoint operations with the Lightning lifecycle

The standalone pattern is preferred because:

- Checkpoint loading happens once at initialization
- No async complexity during training loop
- Clear separation of concerns between loading and training
- Easier to debug and test
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
        """
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
            if isinstance(stage, dict | DictConfig):
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

    @staticmethod
    def _stage_role(stage: Any) -> str | None:
        """Classify a stage by its checkpoint-pipeline layer.

        Returns ``"source"``, ``"loader"``, ``"modifier"``, or ``None`` for a
        stage that belongs to no known layer. Classification is by ``isinstance``
        against the layer base classes — the robust contract — rather than by
        class-name substring, so a correctly-subclassed stage is recognised
        regardless of what it is named (e.g. a ``Restorer(LoadingStrategy)``).
        The base-class imports are local so importing this module does not pull
        in the layer subpackages.
        """
        from anemoi.training.checkpoint.loading.base import LoadingStrategy
        from anemoi.training.checkpoint.modifiers.base import ModelModifier
        from anemoi.training.checkpoint.sources.base import CheckpointSource

        if isinstance(stage, CheckpointSource):
            return "source"
        if isinstance(stage, LoadingStrategy):
            return "loader"
        if isinstance(stage, ModelModifier):
            return "modifier"
        return None

    def _validate_pipeline_composition(self) -> None:
        """Validate pipeline composition: raise on ordering/conflict violations, warn/suggest otherwise.

        Stage ordering (source before loader, modifier after loader) and a single
        loading strategy are structural invariants: a violation is a hard
        ``CheckpointConfigError`` at construction. Multiple sources and missing
        stages are advisory (warn / info) — the builder always emits stages in the
        canonical order, so only hand-built or misconfigured pipelines are bitten.
        """
        if not self.stages:
            LOGGER.warning(
                "Pipeline has no stages configured. "
                "Consider adding stages for source acquisition, loading, or model modification.",
            )
            return

        roles = [self._stage_role(stage) for stage in self.stages]
        stage_names = [stage.__class__.__name__ for stage in self.stages]

        violations: list[str] = []
        self._check_source_loading_order(roles, violations)
        self._check_modifier_placement(roles, violations)
        self._check_duplicate_stages(roles, stage_names, violations)

        if violations:
            from .exceptions import CheckpointConfigError

            message = "Invalid checkpoint pipeline composition:\n  - " + "\n  - ".join(violations)
            raise CheckpointConfigError(message, config_path="training.checkpoint")

        # Non-fatal guidance.
        self._warn_multiple_sources(roles, stage_names)
        self._suggest_missing_stages(roles)

    def _check_source_loading_order(self, roles: list[str | None], violations: list[str]) -> None:
        """A source stage must come before every loading stage."""
        source_indices = [i for i, role in enumerate(roles) if role == "source"]
        loader_indices = [i for i, role in enumerate(roles) if role == "loader"]

        if source_indices and loader_indices and max(source_indices) > min(loader_indices):
            violations.append(
                f"a source stage at position {max(source_indices)} comes after a loader stage at position "
                f"{min(loader_indices)}; loaders expect the checkpoint already acquired "
                "(order stages source -> loader -> modifiers).",
            )

    def _check_modifier_placement(self, roles: list[str | None], violations: list[str]) -> None:
        """Modifier stages must come after every loading stage."""
        loader_indices = [i for i, role in enumerate(roles) if role == "loader"]
        modifier_indices = [i for i, role in enumerate(roles) if role == "modifier"]

        if loader_indices and modifier_indices and min(modifier_indices) < max(loader_indices):
            violations.append(
                f"a modifier stage at position {min(modifier_indices)} comes before a loader stage at position "
                f"{max(loader_indices)}; model modifications must run after checkpoint loading.",
            )

    def _check_duplicate_stages(self, roles: list[str | None], stage_names: list[str], violations: list[str]) -> None:
        """Multiple loading strategies conflict; flag as a violation."""
        loaders = [stage_names[i] for i, role in enumerate(roles) if role == "loader"]
        if len(loaders) > 1:
            violations.append(
                f"found {len(loaders)} loading strategies ({', '.join(loaders)}); "
                "use a single loading strategy appropriate for your use case.",
            )

    def _warn_multiple_sources(self, roles: list[str | None], stage_names: list[str]) -> None:
        """Multiple sources are redundant but not fatal (the pipeline runs them in sequence)."""
        sources = [stage_names[i] for i, role in enumerate(roles) if role == "source"]
        if len(sources) > 1:
            LOGGER.warning(
                "Found %d checkpoint sources (%s). Multiple checkpoint sources may be redundant. "
                "The pipeline will process them in sequence.",
                len(sources),
                ", ".join(sources),
            )

    def _suggest_missing_stages(self, roles: list[str | None]) -> None:
        """Suggest potentially missing stages based on common patterns."""
        has_source = "source" in roles
        has_loader = "loader" in roles
        has_modifier = "modifier" in roles

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
                # Always re-raise MemoryError — never swallow it
                if isinstance(e, MemoryError):
                    raise

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

        # Safety check: if a source stage was configured but the model still
        # has random weights (no loading strategy actually applied them),
        # raise rather than silently returning an uninitialised model.
        self._verify_weights_loaded(context)

        LOGGER.info("Pipeline execution completed")
        return context

    def _verify_weights_loaded(self, context: CheckpointContext) -> None:
        """Raise if a source was configured but weights were never loaded.

        This prevents silently proceeding with a randomly-initialised model
        when ``continue_on_error=True`` swallowed the loading failure.
        ``weights_initialized`` is advisory for other readers, but the pipeline
        treats it as a gate here: a configured source with no applied weights is
        a hard error, not a warning.
        """
        has_source = any(self._stage_role(stage) == "source" for stage in self.stages)
        if not has_source:
            return

        model = context.model
        if model is None:
            return

        if not getattr(model, "weights_initialized", False):
            from .exceptions import CheckpointLoadError

            msg = (
                "A checkpoint source stage was configured but the model's "
                "weights were never loaded (weights_initialized is False). "
                "This usually means the loading strategy stage failed or was "
                "missing. Refusing to proceed with random weights."
            )
            LOGGER.error(msg)
            raise CheckpointLoadError(context.checkpoint_path or "<unresolved checkpoint>", RuntimeError(msg))

    def execute_sync(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Execute pipeline stages synchronously.

        This is a convenience method for synchronous execution,
        wrapping the async execution in asyncio.run(). If an event
        loop is already running (e.g. in Jupyter), uses a thread pool.

        Parameters
        ----------
        initial_context : CheckpointContext
            Initial context to process

        Returns
        -------
        CheckpointContext
            Final processed context
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Inside existing event loop (e.g. Jupyter) — run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.execute_async(initial_context))
                return future.result()
        return asyncio.run(self.execute_async(initial_context))

    async def execute(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Execute the pipeline.

        Main entry point for pipeline execution. This method is always
        asynchronous and delegates to :meth:`execute_async`; await it from
        an async context, or wrap it with ``asyncio.run`` from a sync one.
        Use :meth:`execute_sync` if you need a blocking call that manages
        the event loop for you.

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
        # execute() is already async, so always use execute_async
        return await self.execute_async(initial_context)

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
        if isinstance(stage, dict | DictConfig):
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
