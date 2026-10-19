# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint source that finds a checkpoint from a run ID.

The path is ``<checkpoints.root>.parent/<id>/last.ckpt``. ``LocalSource`` loads it.
``fork=False`` resumes the run. ``fork=True`` starts a new run from its weights.

Example
-------
>>> source = RunIdSource(run_id="abc123")        # resume
>>> context = CheckpointContext(config=cfg)
>>> result = source.process(context)
>>> result.checkpoint_path  # <checkpoints.root.parent>/abc123/last.ckpt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class RunIdSource(CheckpointSource):
    """Acquire a checkpoint from a run's lineage directory.

    Parameters
    ----------
    run_id : str, optional
        The run ID to resume or fork from.
    fork : bool, default False
        If ``True``, fork a new run. If ``False``, resume the run.
    parent_run_server2server : str, optional
        Resume lineage ID. If set, it replaces ``run_id`` in the resume path.
    fork_run_server2server : str, optional
        Fork lineage ID. If set, it replaces ``run_id`` in the fork path.
    """

    def __init__(
        self,
        run_id: str | None = None,
        fork: bool = False,
        parent_run_server2server: str | None = None,
        fork_run_server2server: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.fork = fork
        self.parent_run_server2server = parent_run_server2server
        self.fork_run_server2server = fork_run_server2server

    @staticmethod
    def resolve_path(
        config: DictConfig,
        run_id: str,
        fork: bool,
        parent_run_server2server: str | None = None,
        fork_run_server2server: str | None = None,
    ) -> Path:
        """Build the path ``<checkpoints.root.parent>/<id>/last.ckpt``.

        ``.parent`` removes the run ID that ``_update_paths`` adds to ``checkpoints.root``.

        Raises
        ------
        CheckpointConfigError
            If ``system.output.checkpoints.root`` is not configured.
        """
        from omegaconf import OmegaConf

        root = OmegaConf.select(config, "system.output.checkpoints.root", default=None)
        if root is None:
            from anemoi.training.checkpoint.exceptions import CheckpointConfigError

            msg = (
                "RunIdSource requires system.output.checkpoints.root to resolve a run "
                "checkpoint path, but it is not set in the config."
            )
            raise CheckpointConfigError(msg)

        lineage_id = (fork_run_server2server or run_id) if fork else (parent_run_server2server or run_id)
        return Path(Path(root).parent, lineage_id) / "last.ckpt"

    def resolve(self, context: CheckpointContext) -> Path | None:
        """Resolve the run checkpoint path and set it on the context.

        If the file is missing, rank 0 raises ``CheckpointNotFoundError``.
        Other ranks set ``source_deferred`` on the context and return ``None``.

        Raises
        ------
        CheckpointConfigError
            If ``run_id`` is not set.
        CheckpointNotFoundError
            On rank 0, if the checkpoint file does not exist.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError

        if self.run_id is None:
            from anemoi.training.checkpoint.exceptions import CheckpointConfigError

            msg = (
                "RunIdSource requires a run_id: training.checkpoint.source.run_id is unset. "
                "The config group ships it as null, so pass it explicitly, e.g. "
                "training/checkpoint/source=run +training.checkpoint.source.run_id=<id>."
            )
            raise CheckpointConfigError(msg, config_path="training.checkpoint.source.run_id")

        path = self.resolve_path(
            context.config,
            self.run_id,
            self.fork,
            self.parent_run_server2server,
            self.fork_run_server2server,
        )
        try:
            path = LocalSource(path).resolve(context)
        except CheckpointNotFoundError:
            if rank_zero_only.rank == 0:
                raise
            LOGGER.warning("RunIdSource: checkpoint not found at %s; deferring the error to rank 0.", path)
            context.checkpoint_path = None
            context.update_metadata(source_deferred=True, source_deferred_reason=f"checkpoint not found: {path}")
            return None

        LOGGER.info("RunIdSource: resolved checkpoint path (%s): %s", "fork" if self.fork else "resume", path)
        return path

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Resolve the run checkpoint path and load it.

        If the rank defers the error to rank 0, return the context unchanged.
        """
        path = self.resolve(context)
        if path is not None:
            self._load_from_path(context, path)
        return context


def run_identity_from_config(config: DictConfig) -> tuple[str | None, str | None]:
    """Return ``(run_id, fork_run_id)`` from ``training.checkpoint.source``.

    Sources other than ``RunIdSource`` and its subclasses give ``(None, None)``.

    Parameters
    ----------
    config : DictConfig
        The training config.

    Returns
    -------
    tuple[str | None, str | None]
        ``(run_id, fork_run_id)``; at most one is set.
    """
    from omegaconf import OmegaConf

    from anemoi.training.checkpoint.utils import target_class

    source = OmegaConf.select(config, "training.checkpoint.source", default=None)
    source_cls = target_class(source)
    if source_cls is None or not issubclass(source_cls, RunIdSource):
        return None, None
    run_id = OmegaConf.select(source, "run_id", default=None)
    if run_id is None:
        return None, None
    return (None, run_id) if OmegaConf.select(source, "fork", default=False) else (run_id, None)
