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
On a resume, there are only modifiers, because Lightning loads the checkpoint.
If there is no checkpoint config, the pipeline is empty.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.checkpoint.pipeline import CheckpointPipeline
from anemoi.training.checkpoint.utils import target_class

if TYPE_CHECKING:
    from typing import Any

LOGGER = logging.getLogger(__name__)

# Config keys under ``training.checkpoint``. ``modifiers`` is a list, applied in order.
_TRAINING = "training"
_CHECKPOINT = "checkpoint"
_SOURCE = "source"
_LOADING = "loading"
_MODIFIERS = "modifiers"


def source_config(
    cfg: DictConfig,
    parent_run_server2server: str | None = None,
    fork_run_server2server: str | None = None,
) -> Any:
    """Return the source config, with the run lineage IDs added for a ``RunIdSource``.

    The logger sets the lineage IDs at runtime, thus they are not in the Hydra config.

    Returns
    -------
    Any
        The source config, or ``None`` if there is no source.
    """
    from anemoi.training.checkpoint.sources.run import RunIdSource

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)
    source_cls = target_class(source)
    if source_cls is None or not issubclass(source_cls, RunIdSource):
        return source
    lineage = {
        "parent_run_server2server": parent_run_server2server,
        "fork_run_server2server": fork_run_server2server,
    }
    overrides = {key: value for key, value in lineage.items() if value is not None}
    return OmegaConf.merge(source, overrides) if overrides else source


def _loader_restores_training_state(cfg: DictConfig) -> bool:
    """Return whether the loader class declares ``restores_training_state``.

    ``False`` if there is no loader or its ``_target_`` cannot be imported.
    The pipeline build reports an import failure.
    """
    loading = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}", default=None)
    return bool(getattr(target_class(loading), "restores_training_state", False))


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
    return loading is None or _loader_restores_training_state(cfg)


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
    from anemoi.training.checkpoint.sources.base import CheckpointSource

    source = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_SOURCE}", default=None)

    if source is None:
        if _loader_restores_training_state(cfg):
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

    source_cls = target_class(source)
    if source_cls is not None and issubclass(source_cls, CheckpointSource) and not source_cls.supports_resume():
        msg = (
            f"{source_cls.__name__} cannot resume a run: Trainer.fit(ckpt_path=) needs a local file. "
            "Download the checkpoint and use LocalSource, or add a loading strategy "
            "(WeightsOnlyLoader / TransferLearningLoader / ColdStartLoader) to load only the weights."
        )
        raise CheckpointConfigError(msg)


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
        there is no source or loading stage. ``run_checkpoint_pipeline`` resolves the source.
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

    Raises
    ------
    CheckpointConfigError
        If the config gives a resume that Lightning cannot load.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    reject_unsupported_warm_start(cfg)

    if not load_checkpoint:
        LOGGER.info("Checkpoint acquisition and loading suppressed by the caller; modifier stages still apply")

    stage_configs: list[Any] = []

    # On resume, Lightning loads the file. A load here would be discarded.
    if load_checkpoint and not resumes_via_lightning(cfg):
        source = source_config(cfg, parent_run_server2server, fork_run_server2server)
        if source is not None:
            stage_configs.append(source)
        stage_configs.append(OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_LOADING}"))

    modifiers = OmegaConf.select(cfg, f"{_TRAINING}.{_CHECKPOINT}.{_MODIFIERS}", default=None)
    if modifiers:
        stage_configs.extend(modifiers)

    LOGGER.debug("Building checkpoint pipeline from %d configured stage(s)", len(stage_configs))

    return CheckpointPipeline(stages=stage_configs)
