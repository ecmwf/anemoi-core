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
pipeline and the run starts from freshly initialised weights.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.checkpoint.pipeline import CheckpointPipeline

if TYPE_CHECKING:
    from typing import Any

    from anemoi.training.checkpoint.sources.base import CheckpointSource

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
    """Merge runtime server-to-server lineage onto a source config that accepts it.

    The lineage ids are logger-derived at runtime and cannot be expressed in the
    static Hydra config, so the trainer passes them to the builder. Which sources
    accept them is the source class's business
    (:meth:`~anemoi.training.checkpoint.sources.base.CheckpointSource.with_run_lineage`):
    ``RunIdSource`` and its subclasses merge the non-``None`` values, every other
    source returns its config untouched, so instantiation never fails with an
    unknown keyword argument. An unresolvable ``_target_`` is left for the pipeline
    build, or on a resume :func:`build_resume_source`, to report.
    """
    from anemoi.training.checkpoint.sources.base import source_class_from_config

    source_cls = source_class_from_config(source)
    if source_cls is None:
        return source
    return source_cls.with_run_lineage(source, parent_run_server2server, fork_run_server2server)


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
    resume keys on this: the builder emits no source or loading stage (the runner
    resolves the source through :func:`build_resume_source`), and the trainer keeps
    ``ckpt_path`` so Lightning loads the weights,
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
    restore, and the source is what produces the file it reads. That file has to
    be local, so a resume (see :func:`resumes_via_lightning`) needs a source that
    implements ``resolve`` (``LocalSource``, ``RunIdSource``); ``S3Source`` and
    ``HTTPSource`` delete their download straight after loading and cannot resume.
    Called at the start of :func:`build_checkpoint_pipeline` so the checkpoint
    module owns this composition rule.

    Raises
    ------
    CheckpointConfigError
        If ``training.checkpoint.loading`` declares ``restores_training_state`` but no
        source is configured, or if a resume uses a source that does not support it.
    """
    from anemoi.training.checkpoint.exceptions import CheckpointConfigError
    from anemoi.training.checkpoint.sources.base import source_class_from_config

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)

    if source is None:
        if loader_restores_training_state(cfg):
            msg = (
                "Warm start resumes through Trainer.fit(ckpt_path=), which needs a checkpoint to "
                "read, but no training.checkpoint.source is configured. Add a source "
                "(RunIdSource or LocalSource), or switch training.checkpoint.loading to "
                "WeightsOnlyLoader / TransferLearningLoader / ColdStartLoader to start fresh "
                "training state from a checkpoint."
            )
            raise CheckpointConfigError(msg)
        return

    if not resumes_via_lightning(cfg):
        return

    source_cls = source_class_from_config(source)
    if source_cls is not None and not source_cls.supports_resume():
        msg = (
            f"{source_cls.__name__} cannot resume a run: Trainer.fit(ckpt_path=) needs a local file. "
            "Download the checkpoint and use LocalSource, or add a loading strategy "
            "(WeightsOnlyLoader / TransferLearningLoader / ColdStartLoader) to load only the weights."
        )
        raise CheckpointConfigError(msg)


def _source_config(
    cfg: DictConfig,
    parent_run_server2server: str | None,
    fork_run_server2server: str | None,
) -> Any:
    """The ``training.checkpoint.source`` block with runtime lineage merged in, or ``None``."""
    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    if source is None:
        return None
    return _inject_run_lineage(source, parent_run_server2server, fork_run_server2server)


def build_resume_source(
    cfg: DictConfig,
    *,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
    load_checkpoint: bool = True,
) -> CheckpointSource | None:
    """Instantiate the source whose ``resolve`` step gives Lightning its ``ckpt_path``.

    On a resume (see :func:`resumes_via_lightning`) the pipeline has no source
    stage: ``Trainer.fit(ckpt_path=)`` performs the one load, so the runner only
    calls this source's ``resolve`` to publish the local file. Call
    :func:`build_checkpoint_pipeline` first: it rejects a source that cannot
    resume.

    Parameters
    ----------
    cfg : DictConfig
        A training-run configuration.
    parent_run_server2server : str, optional
        Runtime server-to-server resume lineage id, as for
        :func:`build_checkpoint_pipeline`.
    fork_run_server2server : str, optional
        Runtime server-to-server fork lineage id, as for
        :func:`build_checkpoint_pipeline`.
    load_checkpoint : bool, optional
        If ``False`` (the MLflow dry-run gate), nothing is resolved.

    Returns
    -------
    CheckpointSource or None
        The instantiated source, or ``None`` when the run does not resume through
        Lightning, has no source, or must not acquire a checkpoint.

    Raises
    ------
    CheckpointConfigError
        If the source configuration cannot be instantiated, or its ``_target_``
        is not a :class:`CheckpointSource`.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    if not load_checkpoint or not resumes_via_lightning(cfg):
        return None

    source = _source_config(cfg, parent_run_server2server, fork_run_server2server)
    if source is None:
        return None

    from hydra.utils import instantiate

    from anemoi.training.checkpoint.exceptions import CheckpointConfigError
    from anemoi.training.checkpoint.sources.base import CheckpointSource

    target = OmegaConf.select(source, "_target_", default="<unknown>")
    config_path = f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}"
    try:
        instance = instantiate(source)
    except Exception as e:
        msg = f"Failed to instantiate the resume checkpoint source '{target}': {e}"
        raise CheckpointConfigError(msg, config_path=config_path) from e
    if not isinstance(instance, CheckpointSource):
        msg = f"The resume checkpoint source '{target}' is not a CheckpointSource (got {type(instance).__name__})"
        raise CheckpointConfigError(msg, config_path=config_path)
    return instance


def build_checkpoint_pipeline(
    cfg: DictConfig,
    *,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
    load_checkpoint: bool = True,
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
        no source or loading stage is emitted (the runner resolves the source
        through :func:`build_resume_source`), and modifiers still apply:
        ``Trainer.fit(ckpt_path=)`` performs the one and only load.
    parent_run_server2server : str, optional
        Runtime server-to-server resume lineage id. When set and the source is a
        ``RunIdSource``, it is merged into the source config before instantiation so
        a cross-server resume resolves the same path the trainer would. Ignored
        for other source types.
    fork_run_server2server : str, optional
        Runtime server-to-server fork lineage id, merged into a ``RunIdSource``
        source config as above for the fork path.
    load_checkpoint : bool, optional
        Whether a checkpoint should be acquired and applied at all (default
        ``True``). Pass ``False`` when the run must start from scratch despite a
        configured source — the MLflow dry-run gate, where the parent run was
        minted by ``anemoi-training mlflow prepare`` and has no checkpoint yet.
        Only the source and loading stages are suppressed; modifier stages still
        apply, so a freezing-only configuration is unaffected.

    Returns
    -------
    CheckpointPipeline
        A pipeline whose stages are ordered source → loader → modifiers. When
        nothing is configured the pipeline has zero stages and is a no-op, so the
        model is returned as instantiated.

    Raises
    ------
    CheckpointConfigError
        If the configuration describes a resume that Lightning cannot load (see
        :func:`reject_unsupported_warm_start`).
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Composition rule: a resume needs a source that can hand Lightning a local file.
    reject_unsupported_warm_start(cfg)

    # Resume has one owner. Trainer.fit(ckpt_path=) reads the file and runs
    # on_load_checkpoint on the dict it loads, so the runner only resolves the
    # file and the pipeline never loads it: a pipeline load would read the file a
    # second time and apply the corrections to a copy that Lightning then discards.
    resume = resumes_via_lightning(cfg)

    # A second, independent reason not to acquire anything: the caller says the run
    # starts from scratch (the MLflow dry-run gate). Modifier stages run either way:
    # they change the model, not the checkpoint.
    if not load_checkpoint:
        LOGGER.info("Checkpoint acquisition and loading suppressed by the caller; modifier stages still apply")

    stage_configs: list[Any] = []

    source = _source_config(cfg, parent_run_server2server, fork_run_server2server)
    if source is not None and not resume and load_checkpoint:
        stage_configs.append(source)

    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is not None and not resume and load_checkpoint:
        stage_configs.append(loading)

    modifiers = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_MODIFIERS}", default=None)
    if modifiers:
        # Preserve list order: modifiers execute in the order they are declared.
        stage_configs.extend(modifiers)

    LOGGER.debug("Building checkpoint pipeline from %d configured stage(s)", len(stage_configs))

    # CheckpointPipeline instantiates each ``_target_`` config in order via Hydra
    # and surfaces a CheckpointConfigError with context on failure.
    pipeline = CheckpointPipeline(stages=stage_configs)
    _reject_lora_dropout_with_model_sharding(pipeline, cfg)
    return pipeline


def _reject_lora_dropout_with_model_sharding(pipeline: CheckpointPipeline, cfg: DictConfig) -> None:
    """Refuse LoRA dropout when the model is sharded across GPUs.

    Processors refuse dropout under model sharding, but they only check their own
    ``dropout_p``: dropout inside a LoRA adapter would pass that check unseen. The
    stage cannot see the hardware config, so the builder checks it here, before
    training starts.

    Parameters
    ----------
    pipeline : CheckpointPipeline
        The built pipeline.
    cfg : DictConfig
        The training-run configuration.

    Raises
    ------
    CheckpointConfigError
        If a ``PeftModifierStage`` has ``lora_dropout > 0`` and
        ``system.hardware.num_gpus_per_model > 1``.
    """
    num_gpus_per_model = OmegaConf.select(cfg, "system.hardware.num_gpus_per_model", default=1) or 1
    if num_gpus_per_model <= 1:
        return

    from anemoi.training.checkpoint.exceptions import CheckpointConfigError
    from anemoi.training.checkpoint.modifiers.peft import PeftModifierStage

    for stage in pipeline.stages:
        if isinstance(stage, PeftModifierStage) and stage.lora_config.lora_dropout > 0:
            msg = (
                f"lora_dropout={stage.lora_config.lora_dropout} is not supported when the model is sharded "
                f"(system.hardware.num_gpus_per_model={num_gpus_per_model}). Set lora_dropout to 0."
            )
            raise CheckpointConfigError(msg, config_path=f"{_TRAINING}.{_CHECKPOINT}.{_MODIFIERS}")
