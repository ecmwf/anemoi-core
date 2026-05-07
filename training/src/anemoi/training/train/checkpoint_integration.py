# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Integration layer for the checkpoint pipeline in the training workflow.

This module bridges the new checkpoint pipeline infrastructure with
AnemoiTrainer. It provides helper functions that:

1. Build a CheckpointPipeline from Hydra config
2. Create the CheckpointContext with model and checkpoint path
3. Execute the pipeline synchronously (trainer context isn't async)
4. Return the modified model with loaded weights

Example
-------
>>> from anemoi.training.train.checkpoint_integration import load_checkpoint_with_pipeline
>>>
>>> # In AnemoiTrainer.model property:
>>> if hasattr(config.training, 'checkpoint_pipeline'):
...     model = load_checkpoint_with_pipeline(
...         config=config.training.checkpoint_pipeline,
...         model=model,
...         checkpoint_path=self.last_checkpoint,
...     )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def load_checkpoint_with_pipeline(
    config: DictConfig,
    model: nn.Module,
    checkpoint_path: Path | str | None = None,
    optimizer: object | None = None,
    scheduler: object | None = None,
) -> nn.Module:
    """Load a checkpoint into the model using the checkpoint pipeline.

    This is the main integration point between the training workflow and
    the new checkpoint pipeline infrastructure. It handles:

    - Building the pipeline from Hydra config
    - Creating a CheckpointContext with the model and path
    - Executing the pipeline synchronously
    - Returning the model with loaded weights

    Parameters
    ----------
    config : DictConfig
        Hydra configuration for the checkpoint pipeline. Should contain:
        - stages: List of stage configs with '_target_'
        - async_execution: Optional bool (default: False for sync execution)
        - continue_on_error: Optional bool (default: False)
    model : nn.Module
        The PyTorch model to load weights into
    checkpoint_path : Path | str | None, optional
        Path to the checkpoint file. Required if using LocalSource.
        Can be set in config stages instead if using remote sources.
    optimizer : object | None, optional
        Optimizer instance for WarmStartLoader (full state restoration)
    scheduler : object | None, optional
        LR scheduler instance for WarmStartLoader

    Returns
    -------
    nn.Module
        The model with checkpoint weights loaded

    Raises
    ------
    CheckpointError
        If any stage in the pipeline fails
    CheckpointNotFoundError
        If the checkpoint file doesn't exist (for LocalSource)
    CheckpointLoadError
        If weights cannot be loaded into the model

    Examples
    --------
    >>> # Simple transfer learning
    >>> model = load_checkpoint_with_pipeline(
    ...     config=config.training.checkpoint_pipeline,
    ...     model=model,
    ...     checkpoint_path="/path/to/checkpoint.ckpt",
    ... )

    >>> # Warm start with optimizer restoration
    >>> model = load_checkpoint_with_pipeline(
    ...     config=config.training.checkpoint_pipeline,
    ...     model=model,
    ...     checkpoint_path=last_checkpoint,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ... )
    """
    from anemoi.training.checkpoint import CheckpointContext
    from anemoi.training.checkpoint import CheckpointPipeline

    # Build pipeline from config
    LOGGER.info("Building checkpoint pipeline from config")
    pipeline = CheckpointPipeline.from_config(config)

    # Create context with model and checkpoint path
    context = CheckpointContext(
        model=model,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    # Execute pipeline synchronously (trainer context isn't async)
    LOGGER.info("Executing checkpoint pipeline (sync mode)")
    result = pipeline.execute_sync(context)

    # Log results
    if result.metadata:
        LOGGER.debug("Pipeline execution metadata: %s", result.metadata)

    LOGGER.info("Checkpoint pipeline completed successfully")

    return result.model


def has_checkpoint_pipeline_config(config: DictConfig) -> bool:
    """Check if config has a checkpoint_pipeline section.

    Parameters
    ----------
    config : DictConfig
        The training configuration

    Returns
    -------
    bool
        True if checkpoint_pipeline is configured
    """
    return (
        hasattr(config, "training")
        and hasattr(config.training, "checkpoint_pipeline")
        and config.training.checkpoint_pipeline is not None
        and hasattr(config.training.checkpoint_pipeline, "stages")
        and config.training.checkpoint_pipeline.stages is not None
        and len(config.training.checkpoint_pipeline.stages) > 0
    )
