# (C) Copyright 2024-2026 Anemoi contributors.
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
>>> result = pipeline.execute(context)

Execution
---------
The pipeline runs once during model construction, before ``trainer.fit()``:
``AnemoiTrainer.model`` calls :func:`~anemoi.training.checkpoint.runner.run_checkpoint_pipeline`,
which builds the pipeline from ``training.checkpoint.*`` with
:func:`build_checkpoint_pipeline` and then runs :meth:`CheckpointPipeline.execute`,
so checkpoint loading happens exactly once at startup and never inside the
training loop. On a resume the built pipeline has no source stage; the runner
resolves the file through ``build_resume_source``, so drive a resume config
through ``run_checkpoint_pipeline``. Any other config can be driven directly::

    pipeline = build_checkpoint_pipeline(config)  # reads training.checkpoint.*
    context = CheckpointContext(model=model)

    result = pipeline.execute(context)

    model = result.model
"""

from __future__ import annotations

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
    - Dynamic stage management (add/remove)
    - Metadata tracking for debugging

    Parameters
    ----------
    stages : list of PipelineStage, dict, or DictConfig, optional
        List of pipeline stages to execute in order. Each item can be:
        - An instantiated PipelineStage object
        - A dict/DictConfig with '_target_' for Hydra instantiation
    continue_on_error : bool, optional
        Whether to continue pipeline on stage errors (default: False).
        If True, failed stages will be logged but won't stop the pipeline.

    Attributes
    ----------
    stages : list
        Current list of pipeline stages
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
        continue_on_error: bool = False,
    ):
        """Initialize the checkpoint pipeline.

        Parameters
        ----------
        stages : list of PipelineStage or dict/DictConfig, optional
            List of pipeline stages to execute in order. Each item can be:
            - An instantiated PipelineStage object
            - A dict/DictConfig with '_target_' for Hydra instantiation
        continue_on_error : bool, optional
            Whether to continue pipeline on stage errors (default: False)
        """
        self.continue_on_error = continue_on_error

        # Instantiate stages
        self.stages = self._instantiate_stages(stages or [])

        LOGGER.info("Initialized pipeline with %d stages", len(self.stages))
        for i, stage in enumerate(self.stages):
            LOGGER.debug("  Stage %d: %s", i, stage)

        self._check_stage_order()

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

    def _check_stage_order(self) -> None:
        """Raise if the stages are not ordered source -> one loader -> modifiers.

        Stage ordering (source before loader and modifiers, modifier after loader)
        and a single loading strategy are structural invariants: a violation is a hard
        ``CheckpointConfigError`` at construction. More than one source, or a
        missing layer, is allowed. Stages that belong to no known layer are not
        placed. The builder always emits stages in the canonical order, so only
        hand-built pipelines are bitten.

        Among the modifiers, a ``FreezingModifierStage`` must not follow a
        ``PeftModifierStage``: the PEFT stage sets the trainable parameters, and
        freezing after it could freeze the adapters.

        Raises
        ------
        CheckpointConfigError
            If a source follows a loader or a modifier, a modifier precedes a
            loader, there is more than one loading strategy, or freezing follows PEFT.
        """
        from .exceptions import CheckpointConfigError
        from .modifiers.freezing import FreezingModifierStage
        from .modifiers.peft import PeftModifierStage

        names = [type(stage).__name__ for stage in self.stages]
        rank = {"source": 0, "loader": 1, "modifier": 2}
        ranks = [rank[role] for role in map(self._stage_role, self.stages) if role is not None]
        if ranks != sorted(ranks) or ranks.count(rank["loader"]) > 1:
            msg = f"Invalid checkpoint pipeline {names}: use the order source -> one loader -> modifiers."
            raise CheckpointConfigError(msg, config_path="training.checkpoint")

        peft_at = [i for i, stage in enumerate(self.stages) if isinstance(stage, PeftModifierStage)]
        freezing_at = [i for i, stage in enumerate(self.stages) if isinstance(stage, FreezingModifierStage)]
        if peft_at and freezing_at and max(freezing_at) > min(peft_at):
            msg = (
                f"Invalid checkpoint pipeline {names}: FreezingModifierStage after PeftModifierStage could "
                "freeze the adapters. List freezing first, or use the PEFT stage's trainable_modules."
            )
            raise CheckpointConfigError(msg, config_path="training.checkpoint.modifiers")

    def execute(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Execute the pipeline stages in sequence.

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
                context = stage.process(context)
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

        A resume is the exception: the pipeline has no source stage then (the
        runner resolves the file for ``Trainer.fit(ckpt_path=)``), so the weights
        arrive at ``fit()`` and are not missing.
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
        CheckpointPipeline(stages=['Stage1', 'Stage2'], continue_on_error=False)
        """
        stage_names = [s.__class__.__name__ for s in self.stages]
        return f"CheckpointPipeline(stages={stage_names}, continue_on_error={self.continue_on_error})"
