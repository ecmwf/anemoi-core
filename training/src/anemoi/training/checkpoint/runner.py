# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Run the checkpoint pipeline for the trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from pytorch_lightning.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn
    from omegaconf import DictConfig

    from anemoi.training.checkpoint.base import CheckpointContext
    from anemoi.training.checkpoint.sources.base import CheckpointSource

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """The result of a checkpoint pipeline run.

    Attributes
    ----------
    model : nn.Module
        The model after all stages.
    checkpoint_path : Path, optional
        The checkpoint file that the source resolved. On a resume, Lightning loads this file.
    weights_loaded : bool
        ``True`` if a loading strategy loaded the weights. ``False`` on a resume.
    """

    model: nn.Module
    checkpoint_path: Path | None = None
    weights_loaded: bool = False


def run_checkpoint_pipeline(
    model: nn.Module,
    config: DictConfig,
    *,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
    load_checkpoint: bool = True,
) -> PipelineResult:
    """Build the pipeline from ``config`` and run it on ``model``.

    On a resume, only resolve the source. ``Trainer.fit(ckpt_path=)`` loads the checkpoint.

    Parameters
    ----------
    model : nn.Module
        The model to load into.
    config : DictConfig
        The training config.
    parent_run_server2server : str, optional
        Resume lineage ID from the logger.
    fork_run_server2server : str, optional
        Fork lineage ID from the logger.
    load_checkpoint : bool, default True
        If ``False`` (dry run), do not look for a checkpoint.

    Returns
    -------
    PipelineResult
        The model and the checkpoint data that the trainer needs.
    """
    from anemoi.training.checkpoint.base import CheckpointContext
    from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
    from anemoi.training.checkpoint.builder import resumes_via_lightning
    from anemoi.training.checkpoint.builder import source_config

    pipeline = build_checkpoint_pipeline(
        config,
        parent_run_server2server=parent_run_server2server,
        fork_run_server2server=fork_run_server2server,
        load_checkpoint=load_checkpoint,
    )
    context = CheckpointContext(model=model, config=config)

    source = source_config(config, parent_run_server2server, fork_run_server2server)
    if load_checkpoint and source is not None and resumes_via_lightning(config):
        _resolve_resume_checkpoint(instantiate(source), context)

    context = pipeline.execute(context)

    return PipelineResult(
        model=context.model,
        checkpoint_path=context.checkpoint_path,
        weights_loaded=bool(getattr(context.model, "weights_initialized", False)),
    )


def _resolve_resume_checkpoint(source: CheckpointSource, context: CheckpointContext) -> None:
    """Resolve the resume checkpoint file on ``context``. Do not load it: Lightning loads it.

    Raises
    ------
    CheckpointConfigError
        On rank 0, if the source resolves no file. Other ranks defer the error to rank 0.
    """
    path = source.resolve(context)
    if path is not None:
        LOGGER.info("Resume checkpoint resolved to %s; Trainer.fit(ckpt_path=) loads it", path)
        return
    if rank_zero_only.rank == 0:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        msg = (
            f"Resume configured, but {type(source).__name__} resolved no checkpoint file for "
            "Trainer.fit(ckpt_path=). Check training.checkpoint.source."
        )
        raise CheckpointConfigError(msg)
    LOGGER.warning("%s resolved no checkpoint on this rank; deferring the error to rank 0", type(source).__name__)
