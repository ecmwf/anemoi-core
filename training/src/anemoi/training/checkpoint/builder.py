# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build a checkpoint pipeline from a training configuration.

This module turns the declarative ``training.checkpoint`` configuration into an
executable :class:`~anemoi.training.checkpoint.pipeline.CheckpointPipeline`. All
pipeline stages (source, loading, modifiers) live under that one namespace. It is the single
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


def _inject_run_lineage(
    source: Any,
    parent_run_server2server: str | None,
    fork_run_server2server: str | None,
) -> Any:
    """Merge runtime server-to-server lineage onto a ``RunIdSource`` config.

    The lineage ids are logger-derived at runtime and cannot be expressed in the
    static Hydra config, so the trainer passes them to the builder. Only a
    ``RunIdSource`` target accepts them, and only non-``None`` values are merged, so
    an explicitly-configured value is never clobbered and other source types are
    left untouched (which would otherwise fail instantiation with an unknown
    keyword argument).
    """
    target = OmegaConf.select(source, "_target_", default="") or ""
    if not target.endswith("RunIdSource"):
        return source
    overrides = {
        key: value
        for key, value in (
            ("parent_run_server2server", parent_run_server2server),
            ("fork_run_server2server", fork_run_server2server),
        )
        if value is not None
    }
    if not overrides:
        return source
    return OmegaConf.merge(source, overrides)


def reject_unsupported_warm_start(cfg: DictConfig) -> None:
    """Reject warm start configured with a source that has no local checkpoint.

    Warm start restores optimizer and epoch state through Lightning's ``ckpt_path``,
    which needs a checkpoint reachable as a local file. Only ``LocalSource`` (an explicit
    path) and ``RunIdSource`` (a resolved ``last.ckpt`` on a shared filesystem) provide one.
    With an ``S3Source`` / ``HTTPSource`` — or no source at all — the pipeline would still
    load the weights, but there is no local path for Lightning to resume the optimizer /
    epoch state, so it would silently start from step 0. Fail loudly here instead of
    degrading silently. Called at the start of :func:`build_checkpoint_pipeline` so the
    checkpoint module owns this composition rule.

    Raises
    ------
    CheckpointConfigError
        If ``training.checkpoint.loading`` is a ``WarmStartLoader`` but the source is not
        a Local/Run source.
    """
    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is None:
        return
    loading_target = OmegaConf.select(loading, "_target_", default="") or ""
    if not loading_target.endswith("WarmStartLoader"):
        return

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    source_target = (OmegaConf.select(source, "_target_", default="") or "") if source is not None else ""
    if source_target.endswith(("LocalSource", "RunIdSource")):
        return

    from anemoi.training.checkpoint.exceptions import CheckpointConfigError

    described = source_target.rsplit(".", 1)[-1] if source_target else "no source"
    msg = (
        "Warm start restores optimizer and epoch state via Lightning's ckpt_path, "
        "which requires a checkpoint reachable as a local file. The configured "
        f"training.checkpoint.source ({described}) does not provide one, so the optimizer "
        "and epoch state would be silently dropped. Use a LocalSource or RunIdSource for "
        "warm start, or switch training.checkpoint.loading to WeightsOnlyLoader / "
        "TransferLearningLoader if you only need the weights."
    )
    raise CheckpointConfigError(msg)


def build_checkpoint_pipeline(
    cfg: DictConfig,
    *,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
) -> CheckpointPipeline:
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
    parent_run_server2server : str, optional
        Runtime server-to-server resume lineage id. When set and the source is a
        ``RunIdSource``, it is merged into the source config before instantiation so
        a cross-server resume resolves the same path the trainer would. Ignored
        for other source types.
    fork_run_server2server : str, optional
        Runtime server-to-server fork lineage id, merged into a ``RunIdSource``
        source config as above for the fork path.

    Returns
    -------
    CheckpointPipeline
        A pipeline whose stages are ordered source → loader → modifiers. When
        nothing is configured the pipeline has zero stages and is a no-op, so the
        caller's existing (legacy) checkpoint handling is left untouched.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Composition rule: warm start needs a source that yields a local ckpt_path.
    reject_unsupported_warm_start(cfg)

    stage_configs: list[Any] = []

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    if source is not None:
        source = _inject_run_lineage(source, parent_run_server2server, fork_run_server2server)
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
