# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Run-lineage checkpoint source.

``RunSource`` is the acquisition-layer source for *resume* and *fork* by run id.
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
>>> source = RunSource(run_id="abc123")        # resume
>>> context = CheckpointContext(config=cfg)
>>> result = await source.process(context)
>>> result.checkpoint_path  # <checkpoints.root.parent>/abc123/last.ckpt
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)

# Launcher rank variables, in the exact priority order of
# ``lightning_fabric.utilities.rank_zero._get_rank()`` — matching this set keeps
# the rank-0 missing-checkpoint gate at parity with the legacy guard.
_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


class RunSource(CheckpointSource):
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
                "RunSource requires system.output.checkpoints.root to resolve a run "
                "checkpoint path, but it is not set in the config."
            )
            raise CheckpointConfigError(msg)

        lineage_id = (fork_run_server2server or run_id) if fork else (parent_run_server2server or run_id)
        return Path(Path(root).parent, lineage_id) / "last.ckpt"

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Resolve the run checkpoint path and load it via :class:`LocalSource`.

        Returns the context unchanged when ``run_id`` is unset. On a missing
        checkpoint, raises ``RuntimeError`` on rank 0 and defers (warns,
        pass-through) on other ranks — mirroring the legacy resolver.
        """
        if self.run_id is None:
            LOGGER.debug("RunSource: no run_id set; pass-through.")
            return context

        path = self.resolve_path(
            context.config,
            self.run_id,
            self.fork,
            self.parent_run_server2server,
            self.fork_run_server2server,
        )

        if not path.exists():
            if _is_rank_zero():
                msg = f"Could not find checkpoint for run '{self.run_id}': {path}"
                raise RuntimeError(msg)
            LOGGER.warning("RunSource: checkpoint not found at %s; deferring the error to rank 0.", path)
            return context

        resolution = "fork" if self.fork else "resume"
        LOGGER.info("RunSource: resolved checkpoint path (%s): %s", resolution, path)
        context.checkpoint_path = path
        context.update_metadata(resolved_checkpoint_path=str(path), lineage_resolution=resolution)

        # Delegate the actual torch.load + format detection to LocalSource so the
        # load path is identical to an explicit local checkpoint.
        return await LocalSource().process(context)


def _is_rank_zero() -> bool:
    """Best-effort rank-0 detection without coupling to Lightning.

    Reads the launcher rank variables in ``lightning_fabric``'s priority order
    (``_RANK_ENV_VARS``). A process with no rank variable set (single-process /
    unit test) is treated as rank 0, and a malformed value — non-integer or
    negative — is treated conservatively as rank 0 so the missing-checkpoint
    error is never silently swallowed.
    """
    for var in _RANK_ENV_VARS:
        value = os.environ.get(var)
        if value is not None and value.strip():
            try:
                parsed = int(value)
            except ValueError:
                return True
            return parsed <= 0
    return True
