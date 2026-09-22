# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Run-lineage checkpoint source.

``RunIdSource`` is the acquisition-layer source for *resume* and *fork* by run id.
It resolves the checkpoint path from a run id —
``<system.output.checkpoints.root>.parent/<id>/last.ckpt`` — and delegates the
actual load to :class:`~anemoi.training.checkpoint.sources.local.LocalSource`,
so load semantics are byte-identical to a local checkpoint.

``fork=False`` resumes the run (same MLflow run); ``fork=True`` forks from it
(a new MLflow run started from its weights). The trainer maps the source's
``run_id``/``fork`` onto the run-identity that drives MLflow and the output
paths, so this stage only has to find and load the checkpoint.

Server-to-server lineage overrides (``parent_run_server2server`` /
``fork_run_server2server``) are logger-derived at runtime, so the trainer injects
them as constructor inputs — the acquisition layer must not import the
trainer/MLflow logger. The path formula and the rank-0 missing-checkpoint
behaviour match the legacy run-lineage resolution this source replaces.

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
        The run id to resume from (``fork=False``) or fork from (``fork=True``).
        When ``None`` the source is a no-op pass-through.
    fork : bool, default False
        ``False`` resumes ``run_id``; ``True`` forks a new run from ``run_id``.
    parent_run_server2server : str, optional
        Server-to-server resume lineage id; when set it takes precedence over
        ``run_id`` for the resume path (injected by the trainer).
    fork_run_server2server : str, optional
        Server-to-server fork lineage id; when set it takes precedence over
        ``run_id`` for the fork path (injected by the trainer).
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
        """``fork=False`` resumes ``run_id``; ``fork=True`` forks from it (``run_id`` stays ``None``).

        A fork lets a fresh experiment-tracker id be minted; a resume reuses the
        run's own.

        Parameters
        ----------
        source_cfg : Any
            The source configuration block.

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
        """Merge the runtime lineage ids; a value already in the config is never clobbered.

        Parameters
        ----------
        source_cfg : Any
            The source configuration block.
        parent_run_server2server : str or None
            Runtime server-to-server resume lineage id.
        fork_run_server2server : str or None
            Runtime server-to-server fork lineage id.

        Returns
        -------
        Any
            The configuration with the non-``None`` ids merged in.
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
        """Build the run checkpoint path ``<checkpoints.root.parent>/<id>/last.ckpt``.

        The ``.parent`` mirrors the legacy undo of the ``_update_paths``
        lineage-append to ``checkpoints.root`` (the caller owns that mutation).
        Shared by :meth:`process` and the trainer's resume-path resolution so the
        two cannot drift.

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
        """Resolve the run checkpoint path and publish it on the context.

        Returns ``None`` (context untouched) on a rank that defers. On a missing or
        unreadable checkpoint, raises ``RuntimeError`` on rank 0 and defers (warns,
        returns ``None``) on other ranks — mirroring the legacy resolver. A deferring
        rank records ``source_deferred`` on the context so a loading stage can say
        "rank 0 is reporting the real error" instead of failing with a message about
        a corrupted file. The path is canonicalised through
        :meth:`LocalSource.resolve` so it is the same file an explicit local
        checkpoint would resolve to.

        Raises
        ------
        CheckpointConfigError
            If ``run_id`` is unset. The shipped preset ships ``run_id: null``, so
            selecting ``training/checkpoint/source=run`` and forgetting the id is
            the most likely first mistake; passing through left the loader to
            report a corrupted checkpoint for a file that was never opened.
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

        # An unreadable checkpoint (e.g. wrong permissions) is handled the same way as a
        # missing one: only rank 0 raises, other ranks defer, so a distributed run fails
        # cleanly on rank 0 instead of every rank raising out of the shared torch.load.
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
        """Resolve the run checkpoint path and load it via :class:`LocalSource`.

        Returns the context unchanged when the error was deferred to rank 0 (see
        :meth:`resolve`); an unset ``run_id`` raises there.
        """
        if await self.resolve(context) is None:
            return context

        # Delegate the actual torch.load + format detection to LocalSource so the
        # load path is identical to an explicit local checkpoint.
        return await LocalSource().process(context)


def run_identity_from_config(config: DictConfig) -> tuple[str | None, str | None]:
    """Resolve ``(run_id, fork_run_id)`` from a configured ``training.checkpoint.source``.

    The source class decides: ``RunIdSource`` (and any subclass of it) expresses a
    resume or a fork through :meth:`RunIdSource.run_identity`; every other source,
    or no source, carries no run identity and yields ``(None, None)``. The class is
    resolved from ``_target_``, not matched by name, so a subclass is treated like
    its parent. This replaces the trainer re-deriving the identity: the trainer
    reads it here and never mutates the config to communicate it to the logger or
    the output paths.

    Parameters
    ----------
    config : DictConfig
        The training config; ``training.checkpoint.source`` is inspected.

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
