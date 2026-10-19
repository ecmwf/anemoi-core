# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build a checkpoint pipeline from the ``training.checkpoint`` config.

The stage order is: source, loading, modifiers. The builder skips absent blocks.
If there is no checkpoint config, the pipeline is empty.
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

# Config keys under ``training.checkpoint``. ``modifiers`` is a list, applied in order.
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
    """Add the run lineage IDs to the source config if the source accepts them.

    The logger sets the lineage IDs at runtime, thus they are not in the Hydra config.
    Sources that do not accept lineage return the config unchanged.
    """
    from anemoi.training.checkpoint.sources.base import source_class_from_config

    source_cls = source_class_from_config(source)
    if source_cls is None:
        return source
    return source_cls.with_run_lineage(source, parent_run_server2server, fork_run_server2server)


def loader_restores_training_state(cfg: DictConfig) -> bool:
    """Return whether the configured loader declares ``restores_training_state``.

    The check reads the class attribute, not the class name, so subclasses also match.

    Parameters
    ----------
    cfg : DictConfig
        The training config.

    Returns
    -------
    bool
        ``False`` if the ``_target_`` is absent, empty or cannot be imported.
        The pipeline build reports an import failure.
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
    """Return whether ``Trainer.fit(ckpt_path=)`` loads the checkpoint.

    This is ``True`` if there is no loader, or if the loader declares
    ``restores_training_state``. Then Lightning loads the full training state.

    Parameters
    ----------
    cfg : DictConfig
        The training config.

    Returns
    -------
    bool
        ``True`` if Lightning loads the checkpoint.
    """
    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is None:
        return True
    return loader_restores_training_state(cfg)


def reject_unsupported_warm_start(cfg: DictConfig) -> None:
    """Reject a resume that Lightning cannot load.

    Lightning needs a local file for ``ckpt_path``. Only a source that implements
    ``resolve`` (``LocalSource``, ``RunIdSource``) gives that file.

    Raises
    ------
    CheckpointConfigError
        If the loader declares ``restores_training_state`` and there is no source,
        or if a resume uses a source that does not support it.
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


def _resolve_only(source: Any) -> Any:
    """Wrap a source config so that only its ``resolve`` step runs."""
    return OmegaConf.create(
        {"_target_": "anemoi.training.checkpoint.sources.base.ResolveOnlySource", "source": source},
    )


def build_checkpoint_pipeline(
    cfg: DictConfig,
    *,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
    load_checkpoint: bool = True,
) -> CheckpointPipeline:
    """Build a ``CheckpointPipeline`` from a training config.

    Parameters
    ----------
    cfg : DictConfig
        The training config. The builder reads ``source``, ``loading`` and
        ``modifiers`` under ``training.checkpoint``. All are optional.
        If Lightning loads the checkpoint (see ``resumes_via_lightning``),
        the source only resolves and there is no loading stage.
    parent_run_server2server : str, optional
        Server-to-server resume lineage ID. Only ``RunIdSource`` uses it.
    fork_run_server2server : str, optional
        Server-to-server fork lineage ID. Only ``RunIdSource`` uses it.
    load_checkpoint : bool, optional
        If ``False``, skip the source and loading stages. Modifiers still apply.
        The MLflow dry run uses this.

    Returns
    -------
    CheckpointPipeline
        Stages in the order source, loader, modifiers. Empty if nothing is configured.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    reject_unsupported_warm_start(cfg)

    # On resume, Lightning loads the file. The pipeline only resolves it,
    # because a second load here would be discarded.
    resume = resumes_via_lightning(cfg)

    if not load_checkpoint:
        LOGGER.info("Checkpoint acquisition and loading suppressed by the caller; modifier stages still apply")

    stage_configs: list[Any] = []

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    if source is not None and load_checkpoint:
        source = _inject_run_lineage(source, parent_run_server2server, fork_run_server2server)
        stage_configs.append(_resolve_only(source) if resume else source)

    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    if loading is not None and not resume and load_checkpoint:
        stage_configs.append(loading)

    modifiers = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_MODIFIERS}", default=None)
    if modifiers:
        stage_configs.extend(modifiers)

    LOGGER.debug("Building checkpoint pipeline from %d configured stage(s)", len(stage_configs))

    return CheckpointPipeline(stages=stage_configs)
