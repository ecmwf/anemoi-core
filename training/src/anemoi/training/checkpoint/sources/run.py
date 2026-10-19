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
>>> result = await source.process(context)
>>> result.checkpoint_path  # <checkpoints.root.parent>/abc123/last.ckpt
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.base import is_rank_zero
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

    @classmethod
    def run_identity(cls, source_cfg: Any) -> tuple[str | None, str | None]:
        """Return ``(run_id, fork_run_id)`` from the source config.

        Parameters
        ----------
        source_cfg : Any
            The source config.

        Returns
        -------
        tuple[str | None, str | None]
            ``(run_id, fork_run_id)``; at most one is set.
        """
        from omegaconf import OmegaConf

        run_id = OmegaConf.select(source_cfg, "run_id", default=None)
        if run_id is None:
            return None, None
        if bool(OmegaConf.select(source_cfg, "fork", default=False)):
            return None, run_id
        return run_id, None

    @classmethod
    def with_run_lineage(
        cls,
        source_cfg: Any,
        parent_run_server2server: str | None,
        fork_run_server2server: str | None,
    ) -> Any:
        """Merge the runtime lineage IDs into the source config.

        Parameters
        ----------
        source_cfg : Any
            The source config.
        parent_run_server2server : str or None
            Resume lineage ID.
        fork_run_server2server : str or None
            Fork lineage ID.

        Returns
        -------
        Any
            The config with the non-``None`` IDs merged in.
        """
        from omegaconf import OmegaConf

        overrides = {
            key: value
            for key, value in (
                ("parent_run_server2server", parent_run_server2server),
                ("fork_run_server2server", fork_run_server2server),
            )
            if value is not None
        }
        if not overrides:
            return source_cfg
        return OmegaConf.merge(source_cfg, overrides)

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

    async def resolve(self, context: CheckpointContext) -> Path | None:
        """Resolve the run checkpoint path and set it on the context.

        If the file is missing or not readable, rank 0 raises ``RuntimeError``.
        Other ranks set ``source_deferred`` on the context and return ``None``.

        Raises
        ------
        CheckpointConfigError
            If ``run_id`` is not set.
        """
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

        if not path.exists():
            if is_rank_zero():
                msg = f"Could not find checkpoint for run '{self.run_id}': {path}"
                raise RuntimeError(msg)
            LOGGER.warning("RunIdSource: checkpoint not found at %s; deferring the error to rank 0.", path)
            context.update_metadata(source_deferred=True, source_deferred_reason=f"checkpoint not found: {path}")
            return None

        if not os.access(path, os.R_OK):
            if is_rank_zero():
                msg = f"Checkpoint for run '{self.run_id}' is not readable: {path}"
                raise RuntimeError(msg)
            LOGGER.warning("RunIdSource: checkpoint not readable at %s; deferring the error to rank 0.", path)
            context.update_metadata(source_deferred=True, source_deferred_reason=f"checkpoint not readable: {path}")
            return None

        resolution = "fork" if self.fork else "resume"
        LOGGER.info("RunIdSource: resolved checkpoint path (%s): %s", resolution, path)
        context.checkpoint_path = path
        context.update_metadata(resolved_checkpoint_path=str(path), lineage_resolution=resolution)
        return await LocalSource().resolve(context)

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Resolve the run checkpoint path and load it with ``LocalSource``.

        If the rank defers the error to rank 0, return the context unchanged.
        """
        if await self.resolve(context) is None:
            return context

        return await LocalSource().process(context)


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

    from anemoi.training.checkpoint.sources.base import source_class_from_config

    source = OmegaConf.select(config, "training.checkpoint.source", default=None)
    source_cls = source_class_from_config(source)
    if source_cls is None:
        return None, None
    return source_cls.run_identity(source)
