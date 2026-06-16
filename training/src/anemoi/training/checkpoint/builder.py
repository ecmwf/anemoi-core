# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build a checkpoint pipeline from a training configuration.

This module turns the declarative ``training.checkpoint`` and
``training.model_modifier`` configuration into an executable
:class:`~anemoi.training.checkpoint.pipeline.CheckpointPipeline`. It is the single
place that knows the configuration namespace and the canonical stage order, so a
caller (the trainer, or a test) can obtain a ready-to-run pipeline from a config
object without hand-assembling stages.

Stage order is fixed: the acquisition source first, then the loading strategy,
then any model-modifier stages in the order they are listed. Absent blocks are
skipped; a configuration with no checkpoint section yields an empty (no-op)
pipeline, leaving any legacy checkpoint handling unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.checkpoint.pipeline import CheckpointPipeline

if TYPE_CHECKING:
    from typing import Any

LOGGER = logging.getLogger(__name__)

# Configuration namespace for the checkpoint pipeline, kept together so the field
# names can be changed in a single edit without touching the build logic. All three
# stages live under ``training.checkpoint``: ``source`` and ``loading`` are single
# ``_target_`` objects (exactly one source, exactly one loader); ``modifiers`` is an
# ordered list of ``_target_`` objects applied in list order after loading.
_TRAINING = "training"
_CHECKPOINT = "checkpoint"
_SOURCE = "source"
_LOADING = "loading"
_MODIFIERS = "modifiers"


def build_checkpoint_pipeline(cfg: DictConfig) -> CheckpointPipeline:
    """Assemble a :class:`CheckpointPipeline` from a training configuration.

    Parameters
    ----------
    cfg : DictConfig
        A training-run configuration. The builder reads:

        - ``cfg.training.checkpoint.source`` — a single ``_target_`` source stage
          (acquisition layer), e.g. ``LocalSource`` / ``S3Source`` / ``HTTPSource``.
        - ``cfg.training.checkpoint.loading`` — a single ``_target_`` loading
          strategy, e.g. ``WeightsOnlyLoader`` / ``TransferLearningLoader`` /
          ``WarmStartLoader`` / ``ColdStartLoader``.
        - ``cfg.training.checkpoint.modifiers`` — an ordered list of
          ``_target_`` modifier stages, applied in list order after loading.

        Any of these blocks may be absent; an absent block contributes no stage.

    Returns
    -------
    CheckpointPipeline
        A pipeline whose stages are ordered source → loader → modifiers. When
        nothing is configured the pipeline has zero stages and is a no-op, so the
        caller's existing (legacy) checkpoint handling is left untouched.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    stage_configs: list[Any] = []

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    if source is not None:
        stage_configs.append(source)

    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is not None:
        stage_configs.append(loading)

    modifiers = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_MODIFIERS}", default=None)
    if modifiers:
        # Preserve list order: modifiers execute in the order they are declared.
        stage_configs.extend(modifiers)

    LOGGER.debug("Building checkpoint pipeline from %d configured stage(s)", len(stage_configs))

    # CheckpointPipeline instantiates each ``_target_`` config in order via Hydra
    # and surfaces a CheckpointConfigError with context on failure.
    return CheckpointPipeline(stages=stage_configs)
