# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Run-lineage checkpoint path resolver for the acquisition layer.

``LineageResolver`` is an acquisition-layer pipeline stage that populates
``context.checkpoint_path`` ahead of the source stage, replicating the legacy
``last_checkpoint`` resolution (``train/train.py`` @ ``origin/main 6dcc870e0``).

Precedence (highest first): explicit warm-start path > fork id > lineage run
id. The resolved lineage path has the shape
``<checkpoints.root.parent>/<fork id or lineage run>/last.ckpt``.

The resolver is a no-op when none of ``training.run_id``,
``training.fork_run_id`` or ``system.input.warm_start`` is set (mirroring the
legacy ``start_from_checkpoint`` gate, ``train.py:102-106``).

Coupling and divergences from the legacy load path are recorded in
``checkpoint-sprint/DECISIONS.md`` D18; the short version:

- Server-to-server lineage (``parent_run_server2server`` /
  ``fork_run_server2server``) is logger-derived at runtime, so it is consumed
  as an **explicit constructor input** rather than recomputed here — the
  acquisition layer must not import the trainer/MLflow logger (rule 8).
- Rank-0 detection reads the distributed-launcher environment instead of
  ``pytorch_lightning.utilities.rank_zero`` so the layer stays Lightning-free.
- The legacy rank-0 missing-checkpoint message is a never-formatted tuple
  (``"...: %s", checkpoint`` — ``train.py:446-447``); this resolver raises a
  formatted ``RuntimeError`` whose ``str()`` contains the path.

The stage is inert until the trainer delegates to it (p3-g5): nothing in the
pipeline constructs it yet, and the ``checkpoints.root.parent`` formula assumes
the caller has already applied the lineage append to ``checkpoints.root``
(``_update_paths``, parity row 15 / p3-g5).

Example
-------
>>> resolver = LineageResolver()
>>> context = CheckpointContext(config=cfg)  # cfg.training.run_id set
>>> result = await resolver.process(context)
>>> result.checkpoint_path  # <checkpoints.root.parent>/<run_id>/last.ckpt
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)

# Launcher rank variables, in the exact priority order of
# ``lightning_fabric.utilities.rank_zero._get_rank()`` — the value behind the
# legacy ``rank_zero_only.rank`` guard at the point ``last_checkpoint`` is
# evaluated (before the Lightning strategy overwrites it). Matching this set and
# order keeps the resolver's rank-0 gate at parity with the legacy guard it
# replaces; diverging from it (e.g. dropping LOCAL_RANK) misclassifies workers
# on some launchers and can defer/raise on the wrong ranks.
_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


class LineageResolver(PipelineStage):
    """Resolve ``context.checkpoint_path`` from run-lineage configuration.

    Replicates the legacy ``last_checkpoint`` resolution as a standalone
    acquisition-layer stage. Reads the statically derivable lineage keys from
    ``context.config`` and takes the runtime, logger-derived server-to-server
    lineage as explicit constructor inputs (see module docstring / D18).

    Parameters
    ----------
    parent_run_server2server : str, optional
        Server-to-server parent run id. When set it takes precedence over the
        resolved ``training.run_id`` as the lineage run, mirroring legacy
        ``lineage_run = parent_run_server2server or run_id`` (``train.py:608``).
    fork_run_server2server : str, optional
        Server-to-server fork run id. When set it takes precedence over
        ``training.fork_run_id`` as the fork id, mirroring legacy
        ``fork_id = fork_run_server2server or fork_run_id`` (``train.py:439``).
    """

    def __init__(
        self,
        parent_run_server2server: str | None = None,
        fork_run_server2server: str | None = None,
    ) -> None:
        self.parent_run_server2server = parent_run_server2server
        self.fork_run_server2server = fork_run_server2server

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Populate ``context.checkpoint_path`` from run-lineage configuration.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context. ``context.config`` supplies the lineage keys.

        Returns
        -------
        CheckpointContext
            The same context, with ``checkpoint_path`` set when a checkpoint is
            resolved, or unchanged when no resume/fork/warm-start key is set.

        Raises
        ------
        FileNotFoundError
            If ``system.input.warm_start`` is set but is not an existing file
            (legacy ``_get_warm_start_checkpoint`` — ``train.py:416-427``).
        RuntimeError
            If the resolved lineage checkpoint does not exist, on rank 0
            (legacy ``last_checkpoint`` — ``train.py:444-448``).
        CheckpointConfigError
            If a lineage path is required but ``system.output.checkpoints.root``
            is not configured.
        """
        from omegaconf import OmegaConf

        config = context.config
        if config is None:
            LOGGER.debug("LineageResolver: no config on context; nothing to resolve.")
            return context

        run_id = OmegaConf.select(config, "training.run_id", default=None)
        fork_run_id = OmegaConf.select(config, "training.fork_run_id", default=None)
        warm_start = OmegaConf.select(config, "system.input.warm_start", default=None)

        # Legacy ``start_from_checkpoint`` gate (train.py:102-106): without a
        # resume, fork or warm-start key there is nothing to resolve.
        if not (run_id or fork_run_id or warm_start):
            LOGGER.debug("LineageResolver: no run_id/fork_run_id/warm_start set; pass-through.")
            return context

        # Precedence: explicit warm-start path > fork id > lineage run id
        # (legacy: ``warm_start or _get_checkpoint_directory(fork_id)`` — train.py:440).
        checkpoint = self._resolve_warm_start(warm_start)
        resolution = "warm_start"
        if checkpoint is None:
            checkpoint = self._resolve_lineage_directory(config, fork_run_id, run_id)
            resolution = "lineage"

        if checkpoint.exists():
            LOGGER.info("LineageResolver: resolved checkpoint path (%s): %s", resolution, checkpoint)
            context.checkpoint_path = checkpoint
            context.update_metadata(
                resolved_checkpoint_path=str(checkpoint),
                lineage_resolution=resolution,
            )
            return context

        # Missing checkpoint: a formatted error containing the path on rank 0.
        # The legacy site builds ``"...: %s", checkpoint`` — a tuple that is
        # never formatted (train.py:446-447); that defect is not reproduced.
        if _is_rank_zero():
            msg = f"Could not find last checkpoint: {checkpoint}"
            raise RuntimeError(msg)

        LOGGER.warning(
            "LineageResolver: checkpoint not found at %s; deferring the error to rank 0.",
            checkpoint,
        )
        return context

    def _resolve_warm_start(self, warm_start: str | os.PathLike | None) -> Path | None:
        """Resolve an explicit warm-start checkpoint path.

        Returns ``None`` when no warm-start path is configured. Raises
        ``FileNotFoundError`` when a path is configured but does not point at an
        existing file (legacy ``_get_warm_start_checkpoint`` —
        ``train.py:416-427``).
        """
        if not warm_start:
            return None

        warm_start_path = Path(warm_start)
        if not warm_start_path.is_file():
            msg = f"Warm start checkpoint not found: {warm_start_path}"
            raise FileNotFoundError(msg)
        return warm_start_path

    def _resolve_lineage_directory(
        self,
        config: DictConfig,
        fork_run_id: str | None,
        run_id: str | None,
    ) -> Path:
        """Build the lineage checkpoint path.

        ``<checkpoints.root.parent>/<fork id or lineage run>/last.ckpt`` (legacy
        ``_get_checkpoint_directory`` — ``train.py:429-431``). The ``.parent``
        mirrors the legacy undo of the ``_update_paths`` lineage-append to
        ``checkpoints.root`` (parity row 15, p3-g5); the caller owns that
        mutation.
        """
        from omegaconf import OmegaConf

        root = OmegaConf.select(config, "system.output.checkpoints.root", default=None)
        if root is None:
            from anemoi.training.checkpoint.exceptions import CheckpointConfigError

            msg = (
                "LineageResolver requires system.output.checkpoints.root to resolve a "
                "lineage checkpoint path, but it is not set in the config."
            )
            raise CheckpointConfigError(msg)

        fork_id = self.fork_run_server2server or fork_run_id
        lineage_run = self.parent_run_server2server or run_id
        return Path(Path(root).parent, fork_id or lineage_run) / "last.ckpt"


def _is_rank_zero() -> bool:
    """Best-effort rank-0 detection without coupling to Lightning.

    The acquisition layer must not import the trainer/Lightning stack (rule 8);
    the legacy site used ``pytorch_lightning.utilities.rank_zero.rank_zero_only``,
    whose value comes from ``lightning_fabric``'s ``_get_rank()``. Rank is read
    from the same launcher environment variables, in the same priority order
    (``_RANK_ENV_VARS``), so the gate agrees with the legacy guard. A process with
    no rank variable set (single-process / unit test) is treated as rank 0, and a
    malformed value — non-integer or negative — is treated conservatively as rank
    0 so the missing-checkpoint error is never silently swallowed.
    """
    for var in _RANK_ENV_VARS:
        value = os.environ.get(var)
        if value is not None and value.strip():
            try:
                parsed = int(value)
            except ValueError:
                return True
            # A negative (malformed) global rank is untrustworthy; behave as
            # rank 0 so the error surfaces rather than being deferred forever.
            return parsed <= 0
    return True
