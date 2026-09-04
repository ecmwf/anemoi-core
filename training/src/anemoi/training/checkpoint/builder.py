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


def loader_restores_training_state(cfg: DictConfig) -> bool:
    """Whether the configured loading strategy declares ``restores_training_state``.

    Resolves ``training.checkpoint.loading._target_`` to its class and reads the
    declared
    :attr:`~anemoi.training.checkpoint.loading.base.LoadingStrategy.restores_training_state`
    rather than matching the class name: a subclass, or any third-party strategy
    that declares the attribute, is treated exactly like ``WarmStartLoader``.

    Parameters
    ----------
    cfg : DictConfig
        A training-run configuration.

    Returns
    -------
    bool
        ``True`` when a loading strategy is configured and declares the attribute.
        An absent or empty ``_target_`` yields ``False``; so does one that cannot be
        imported, which the pipeline build then reports as a ``CheckpointConfigError``
        naming the target.
    """
    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is None:
        return False
    target = OmegaConf.select(loading, "_target_", default="") or ""
    if not target:
        return False

    from hydra.utils import get_class

    try:
        loader_cls = get_class(target)
    except (ImportError, ValueError):
        return False
    return bool(getattr(loader_cls, "restores_training_state", False))


def resumes_via_lightning(cfg: DictConfig) -> bool:
    """Whether ``Trainer.fit(ckpt_path=)`` owns the checkpoint load for this config.

    ``True`` when no loading strategy is configured (a bare
    ``training.checkpoint.source`` resumes the run, the way the removed
    ``training.run_id`` did) or when the configured strategy declares
    ``restores_training_state`` (``WarmStartLoader``). Every decision about a
    resume keys on this: the builder emits a resolve-only source and no loading
    stage, and the trainer keeps ``ckpt_path`` so Lightning loads the weights,
    optimizer, scheduler and loop progress together in one pass.

    Parameters
    ----------
    cfg : DictConfig
        A training-run configuration.

    Returns
    -------
    bool
        ``True`` when Lightning owns the load.
    """
    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is None:
        return True
    return loader_restores_training_state(cfg)


def reject_unsupported_warm_start(cfg: DictConfig) -> None:
    """Reject a resume that has nothing to resume from.

    A loading strategy that declares ``restores_training_state`` needs a
    ``training.checkpoint.source``: Lightning's ``ckpt_path`` load is the whole
    restore, and the source stage is what produces the file it reads. Any source
    will do — a remote one is downloaded to a node-local file that the trainer hands
    to Lightning and deletes afterwards. Called at the start of
    :func:`build_checkpoint_pipeline` so the checkpoint module owns this composition
    rule.

    Raises
    ------
    CheckpointConfigError
        If ``training.checkpoint.loading`` declares ``restores_training_state`` but no
        source is configured.
    """
    if not loader_restores_training_state(cfg):
        return

    if OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None) is not None:
        return

    from anemoi.training.checkpoint.exceptions import CheckpointConfigError

    msg = (
        "Warm start resumes through Trainer.fit(ckpt_path=), which needs a checkpoint to "
        "read, but no training.checkpoint.source is configured. Add a source "
        "(RunIdSource, LocalSource, S3Source or HTTPSource), or switch "
        "training.checkpoint.loading to WeightsOnlyLoader / TransferLearningLoader / "
        "ColdStartLoader to start fresh training state from a checkpoint."
    )
    raise CheckpointConfigError(msg)


def _resolve_only(source: Any) -> Any:
    """Wrap a source config so only its ``resolve`` step runs (see ``ResolveOnlySource``)."""
    return OmegaConf.create(
        {"_target_": "anemoi.training.checkpoint.sources.base.ResolveOnlySource", "source": source},
    )


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
        When the run resumes through Lightning (no ``loading`` block, or a loader
        that declares ``restores_training_state`` — see :func:`resumes_via_lightning`)
        the source is wrapped so only its ``resolve`` step runs, no loading stage is
        emitted, and modifiers still apply: ``Trainer.fit(ckpt_path=)`` performs the
        one and only load.
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

    # Composition rule: a resume needs a source to hand Lightning.
    reject_unsupported_warm_start(cfg)

    # Resume has one owner. Trainer.fit(ckpt_path=) reads the file and runs
    # on_load_checkpoint on the dict it loads, so the pipeline only resolves the
    # file (a download is kept on disk) and never loads it here: a pipeline load
    # would read the file a second time and apply the corrections to a copy that
    # Lightning then discards.
    resume = resumes_via_lightning(cfg)

    stage_configs: list[Any] = []

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    if source is not None:
        source = _inject_run_lineage(source, parent_run_server2server, fork_run_server2server)
        stage_configs.append(_resolve_only(source) if resume else source)

    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is not None and not resume:
        stage_configs.append(loading)

    modifiers = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_MODIFIERS}", default=None)
    if modifiers:
        # Preserve list order: modifiers execute in the order they are declared.
        stage_configs.extend(modifiers)

    LOGGER.debug("Building checkpoint pipeline from %d configured stage(s)", len(stage_configs))

    # CheckpointPipeline instantiates each ``_target_`` config in order via Hydra
    # and surfaces a CheckpointConfigError with context on failure.
    return CheckpointPipeline(stages=stage_configs)
