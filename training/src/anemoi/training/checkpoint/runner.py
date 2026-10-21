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

from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

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
    ckpt_path : Path, optional
        What ``Trainer.fit(ckpt_path=)`` receives: ``checkpoint_path`` on a resume,
        ``None`` when a loading strategy loaded the weights in the pipeline. Decided
        from the config by :func:`~anemoi.training.checkpoint.builder.resumes_via_lightning`.
    weights_loaded : bool
        ``True`` if a loading strategy loaded the weights. ``False`` on a resume.
    """

    model: nn.Module
    checkpoint_path: Path | None = None
    ckpt_path: Path | None = None
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

    On a resume, the source is only resolved to a local file before the pipeline
    runs; ``Trainer.fit(ckpt_path=)`` loads it. Whether the run resumes is read
    from the config alone (:func:`~anemoi.training.checkpoint.builder.resumes_via_lightning`):
    no loading block, or a loader declaring ``restores_training_state``
    (``WarmStartLoader``). Every other strategy loads in the pipeline and
    ``ckpt_path`` is withheld. ``weights_initialized`` never routes the load; the
    pipeline's weights-loaded check only refuses a load that left it unset.

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

    Raises
    ------
    CheckpointLoadError
        If a loading strategy ran in the pipeline but left ``weights_initialized``
        unset (raised by the pipeline's weights-loaded check).
    """
    from anemoi.training.checkpoint.base import CheckpointContext
    from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
    from anemoi.training.checkpoint.builder import build_resume_source
    from anemoi.training.checkpoint.builder import resumes_via_lightning

    build_kwargs = {
        "parent_run_server2server": parent_run_server2server,
        "fork_run_server2server": fork_run_server2server,
        "load_checkpoint": load_checkpoint,
    }
    # Build first: it rejects a resume from a source that cannot give a local file.
    pipeline = build_checkpoint_pipeline(config, **build_kwargs)
    context = CheckpointContext(model=model, config=config)

    resume_source = build_resume_source(config, **build_kwargs)
    if resume_source is not None:
        _resolve_resume_checkpoint(resume_source, context)

    context = pipeline.execute(context)

    # One owner per load, decided from the config: Lightning on a resume, the
    # pipeline's loading stage otherwise. A pipeline load keeps ckpt_path from
    # Lightning, which would otherwise load the same file a second time.
    resume = resumes_via_lightning(config)
    return PipelineResult(
        model=context.model,
        checkpoint_path=context.checkpoint_path,
        ckpt_path=context.checkpoint_path if resume else None,
        weights_loaded=load_checkpoint and not resume,
    )


def _resolve_resume_checkpoint(source: CheckpointSource, context: CheckpointContext) -> None:
    """Resolve the resume checkpoint file onto ``context``; load nothing.

    ``Trainer.fit(ckpt_path=)`` performs the load, so only the source's
    ``resolve`` step runs: it publishes the file on ``context.checkpoint_path``.

    Parameters
    ----------
    source : CheckpointSource
        The configured source.
    context : CheckpointContext
        The pipeline context (mutated in place).

    Raises
    ------
    CheckpointConfigError
        On rank 0, if the source resolved no file: a resume with nothing to hand
        Lightning would otherwise start from scratch without a word. Other ranks
        defer, as the sources do, so the job fails once, on rank 0.
    """
    path = source.resolve(context)
    if path is not None:
        LOGGER.info("Resume checkpoint resolved to %s; Trainer.fit(ckpt_path=) performs the load", path)
        return

    if rank_zero_only.rank == 0:
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        msg = (
            f"Resume configured, but {type(source).__name__} resolved no checkpoint file to hand "
            "Trainer.fit(ckpt_path=). Check the training.checkpoint.source configuration (a RunIdSource "
            "needs run_id), or add a loading strategy if you meant to start fresh training state."
        )
        raise CheckpointConfigError(msg)
    LOGGER.warning("%s resolved no checkpoint on this rank; deferring the error to rank 0", type(source).__name__)
