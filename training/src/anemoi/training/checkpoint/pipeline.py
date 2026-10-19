# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Run checkpoint pipeline stages in sequence.

Example
-------
>>> from anemoi.training.checkpoint import CheckpointPipeline, CheckpointContext
>>> from anemoi.training.checkpoint.sources import LocalSource
>>> from anemoi.training.checkpoint.loading import WeightsOnlyLoader
>>>
>>> pipeline = CheckpointPipeline([
...     LocalSource(path='/tmp/checkpoint.pt'),
...     WeightsOnlyLoader(strict=False),
... ])
>>> result = pipeline.execute(CheckpointContext(model=my_model))

``AnemoiTrainer.model`` calls ``anemoi.training.checkpoint.runner.run_checkpoint_pipeline``
once, before ``trainer.fit()``. It builds the pipeline from ``training.checkpoint`` with
``build_checkpoint_pipeline`` and runs it.
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
    """Run checkpoint stages in sequence.

    Each stage gets the ``CheckpointContext`` from the stage before it.
    The order is: source, loader, modifiers.

    Parameters
    ----------
    stages : list of PipelineStage, dict, or DictConfig, optional
        The stages to run, in order. Each item is a ``PipelineStage`` or a
        config with ``_target_`` for Hydra instantiation.

    Raises
    ------
    CheckpointConfigError
        If a stage config cannot be instantiated, or the stage order is wrong.

    Examples
    --------
    >>> pipeline = CheckpointPipeline([LocalSource(path="model.ckpt"), WeightsOnlyLoader()])
    """

    def __init__(self, stages: list[Union[PipelineStage, DictConfig, dict]] | None = None):
        self.stages = self._instantiate_stages(stages or [])

        LOGGER.debug("Initialized pipeline with %d stages", len(self.stages))
        for i, stage in enumerate(self.stages):
            LOGGER.debug("  Stage %d: %s", i, stage)

        self._check_stage_order()

    def _instantiate_stages(self, stages: list[Any]) -> list[PipelineStage]:
        """Instantiate stage configs. Pass stage instances through."""
        instantiated = []
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict | DictConfig):
                instantiated.append(stage)
                continue
            try:
                instantiated.append(instantiate(stage))
            except Exception as e:
                from .exceptions import CheckpointConfigError

                preview = str(stage)[:200]
                msg = (
                    f"Failed to instantiate pipeline stage {i} from configuration.\n"
                    f"Stage config: {preview}\n"
                    f"Original error: {e}\n"
                    "Make sure that '_target_' is importable and all required parameters are set."
                )
                raise CheckpointConfigError(msg, config_path=f"stages[{i}]") from e
        return instantiated

    def _check_stage_order(self) -> None:
        """Raise if the order is not source -> loader -> modifiers, or if there is more than one loader."""
        from anemoi.training.checkpoint.loading.base import LoadingStrategy
        from anemoi.training.checkpoint.modifiers.base import ModelModifier
        from anemoi.training.checkpoint.sources.base import CheckpointSource

        kinds = (CheckpointSource, LoadingStrategy, ModelModifier)
        ranks = [i for stage in self.stages for i, kind in enumerate(kinds) if isinstance(stage, kind)]
        if ranks != sorted(ranks) or ranks.count(1) > 1:
            from .exceptions import CheckpointConfigError

            names = [type(stage).__name__ for stage in self.stages]
            msg = f"Invalid checkpoint pipeline {names}: use the order source -> one loader -> modifiers."
            raise CheckpointConfigError(msg, config_path="training.checkpoint")

    def execute(self, initial_context: CheckpointContext) -> CheckpointContext:
        """Run the stages in sequence.

        Parameters
        ----------
        initial_context : CheckpointContext
            The context for the first stage.

        Returns
        -------
        CheckpointContext
            The context after the last stage.

        Raises
        ------
        CheckpointError
            If a stage fails. The details tell which stage.
        """
        from .exceptions import CheckpointError

        context = initial_context
        for i, stage in enumerate(self.stages):
            stage_name = type(stage).__name__
            stage_info = f"Pipeline stage {i + 1}/{len(self.stages)} ({stage_name})"
            details = {
                "pipeline_stage": stage_info,
                "pipeline_stage_index": i,
                "pipeline_total_stages": len(self.stages),
            }
            LOGGER.debug("Executing %s", stage_info)
            try:
                context = stage.process(context)
            except MemoryError:
                raise
            except CheckpointError as e:
                e.details.update(details)
                LOGGER.exception("%s failed", stage_info)
                raise
            except Exception as e:
                LOGGER.exception("%s failed", stage_info)
                details.update(original_error=str(e), original_error_type=type(e).__name__)
                msg = f"{stage_info} failed: {e}"
                raise CheckpointError(msg, details) from e

        LOGGER.debug("Pipeline execution completed")
        return context

    def add_stage(self, stage: Union[PipelineStage, DictConfig, dict]) -> None:
        """Add a stage (or a stage config with ``_target_``) at the end of the pipeline."""
        (stage,) = self._instantiate_stages([stage])
        if not callable(getattr(stage, "process", None)):
            from .exceptions import CheckpointConfigError

            msg = f"Invalid stage type: {type(stage).__name__}. A stage must have a 'process' method."
            raise CheckpointConfigError(msg)
        self.stages.append(stage)
        LOGGER.debug("Added stage %s to pipeline", stage)

    def remove_stage(self, stage: PipelineStage) -> None:
        """Remove a stage from the pipeline."""
        if stage in self.stages:
            self.stages.remove(stage)
            LOGGER.debug("Removed stage %s from pipeline", stage)
        else:
            LOGGER.warning("Stage %s not found in pipeline", stage)

    def clear_stages(self) -> None:
        """Remove all stages from the pipeline."""
        self.stages.clear()

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return f"CheckpointPipeline(stages={[type(s).__name__ for s in self.stages]})"
