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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn
    from omegaconf import DictConfig


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


def checkpoint_pipeline_configured(config: DictConfig) -> bool:
    """Return ``True`` if ``training.checkpoint`` is set."""
    return OmegaConf.select(config, "training.checkpoint", default=None) is not None


def run_checkpoint_pipeline(
    model: nn.Module,
    config: DictConfig,
    *,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
    load_checkpoint: bool = True,
) -> PipelineResult:
    """Build the pipeline from ``config`` and run it on ``model``.

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

    pipeline = build_checkpoint_pipeline(
        config,
        parent_run_server2server=parent_run_server2server,
        fork_run_server2server=fork_run_server2server,
        load_checkpoint=load_checkpoint,
    )
    context = pipeline.execute(CheckpointContext(model=model, config=config))

    return PipelineResult(
        model=context.model,
        checkpoint_path=context.checkpoint_path,
        weights_loaded=bool(getattr(context.model, "weights_initialized", False)),
    )
