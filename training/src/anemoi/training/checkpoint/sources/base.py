# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract base class for checkpoint sources.

Checkpoint sources are responsible for acquiring checkpoint data from
various locations (local filesystem, S3, HTTP, etc.) and populating
the pipeline context with the loaded data and detected format.

Source selection is **explicit**: the user picks which source class
runs by listing its fully-qualified name under ``_target_`` in the
Hydra checkpoint config. There is no scheme-based dispatcher that
maps ``s3://`` URLs to ``S3Source`` automatically — that choice is
left to whoever writes the config, so a future caller could (for
instance) use ``LocalSource`` against a FUSE-mounted S3 path without
the pipeline trying to interpret the scheme. Each source's
``supports(...)`` classmethod exists for opt-in validation, not for
automatic dispatch.

Concretely, if a user configures ``LocalSource`` with an ``s3://``
URI (without a FUSE mount), the URI is interpreted as a literal
local path: ``Path("s3://bucket/key").resolve()`` does not exist on
disk, so ``LocalSource.process(...)`` raises
:class:`~anemoi.training.checkpoint.exceptions.CheckpointNotFoundError`.
Callers that want scheme-based routing should perform their own
dispatch (e.g., by inspecting ``urlparse(uri).scheme``) before
selecting a ``_target_``.

``process`` gets the checkpoint and loads it. A local source also implements ``resolve``,
which only gives the local file. A resume calls only ``resolve``, because Lightning loads the file.

Example
-------
>>> class LocalSource(CheckpointSource):
...     def resolve(self, context: CheckpointContext) -> Path:
...         context.checkpoint_path = Path(context.checkpoint_path).expanduser().resolve()
...         return context.checkpoint_path
...
...     def process(self, context: CheckpointContext) -> CheckpointContext:
...         path = self.resolve(context)
...         self._load_from_path(context, path)
...         return context
"""

from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    from pathlib import Path

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


def source_class_from_config(source_cfg: Any) -> type[CheckpointSource] | None:
    """Return the source class from the ``_target_`` of a source config.

    Parameters
    ----------
    source_cfg : Any
        The source config, or ``None``.

    Returns
    -------
    type[CheckpointSource] or None
        ``None`` if the target is absent, cannot be imported, or is not a ``CheckpointSource``.
    """
    if source_cfg is None:
        return None
    from omegaconf import OmegaConf

    target = (
        OmegaConf.select(source_cfg, "_target_", default="")
        if OmegaConf.is_config(source_cfg)
        else source_cfg.get("_target_", "")
    ) or ""
    if not target:
        return None

    from hydra.utils import get_class

    try:
        cls = get_class(target)
    except (ImportError, ValueError):
        return None
    return cls if isinstance(cls, type) and issubclass(cls, CheckpointSource) else None


class CheckpointSource(PipelineStage):
    """Abstract base class for all checkpoint sources.

    Checkpoint sources form the acquisition layer of the checkpoint
    pipeline. They are responsible for obtaining checkpoint data from
    a specific source type (local file, cloud storage, HTTP endpoint,
    etc.) and populating the context for downstream stages.

    Subclasses implement ``process``. Use ``_load_from_path`` to load the file.
    A source that can give a local file for a resume also implements ``resolve``.

    Parameters
    ----------
    None

    Examples
    --------
    >>> class S3Source(CheckpointSource):
    ...     def __init__(self, url: str):
    ...         self.url = url
    ...
    ...     def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         path = self._download_to_temp(self.url)
    ...         self._load_from_path(context, path)
    ...         path.unlink()
    ...         return context
    """

    @classmethod
    def run_identity(cls, source_cfg: Any) -> tuple[str | None, str | None]:
        """Return ``(run_id, fork_run_id)`` for this source config.

        Only ``RunIdSource`` has a run identity. Other sources return ``(None, None)``.

        Parameters
        ----------
        source_cfg : Any
            The source config.

        Returns
        -------
        tuple[str | None, str | None]
            ``(run_id, fork_run_id)``. At most one is set.
        """
        del source_cfg
        return None, None

    @classmethod
    def with_run_lineage(
        cls,
        source_cfg: Any,
        parent_run_server2server: str | None,
        fork_run_server2server: str | None,
    ) -> Any:
        """Add the runtime lineage IDs to the source config if the source accepts them.

        The default returns the config unchanged.

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
            The config to instantiate.
        """
        del parent_run_server2server, fork_run_server2server
        return source_cfg

    def resolve(self, context: CheckpointContext) -> Path | None:
        """Get the checkpoint as a local file and set ``context.checkpoint_path``. Do not load it.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context.

        Returns
        -------
        Path or None
            The local checkpoint file, or ``None`` if this rank defers the error to rank 0.

        Raises
        ------
        NotImplementedError
            If the source does not implement ``resolve``. A resume cannot use this source.
        """
        msg = (
            f"{type(self).__name__} does not implement resolve(); a resume needs a source that can "
            "hand Trainer.fit(ckpt_path=) a local checkpoint file"
        )
        raise NotImplementedError(msg)

    @classmethod
    def supports_resume(cls) -> bool:
        """Return ``True`` if the source implements ``resolve``, thus a resume can use it."""
        return cls.resolve is not CheckpointSource.resolve

    @abstractmethod
    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Acquire checkpoint data from source and populate context.

        Implementations should:
        1. Call ``resolve`` to obtain the local checkpoint file
        2. Call ``_load_from_path`` to load it onto the context
        3. Add source-specific metadata for tracking

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context. May contain ``checkpoint_path``
            or source configuration in ``config``.

        Returns
        -------
        CheckpointContext
            Context with ``checkpoint_data`` and ``checkpoint_format``
            populated.

        Raises
        ------
        CheckpointSourceError
            If the source cannot be reached or data cannot be fetched
        CheckpointNotFoundError
            If the checkpoint does not exist at the source location
        CheckpointLoadError
            If the fetched data cannot be parsed as a checkpoint
        """

    def _load_from_path(self, context: CheckpointContext, path: Path) -> None:
        """Load the checkpoint file at ``path`` onto the context.

        ``weights_only=False`` is necessary because checkpoints contain non-tensor metadata.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context (mutated in place)
        path : Path
            The local checkpoint file

        Raises
        ------
        CheckpointLoadError
            If the file cannot be loaded by PyTorch
        """
        from anemoi.training.checkpoint.exceptions import CheckpointLoadError

        try:
            raw_data = torch.load(path, weights_only=False, map_location="cpu")
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as e:
            raise CheckpointLoadError(path, e) from e

        self._load_and_populate(context, raw_data)

    def _load_and_populate(
        self,
        context: CheckpointContext,
        raw_data: dict[str, Any],
    ) -> None:
        """Populate context **in-place** with loaded checkpoint data and detected format.

        This convenience method standardises how checkpoint sources
        attach raw data to the context. It detects the checkpoint format
        from the loaded data and sets both ``checkpoint_data`` and
        ``checkpoint_format`` on the context. The context is mutated in
        place; nothing is returned.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context (mutated in place)
        raw_data : dict
            Raw loaded checkpoint data dictionary
        """
        context.checkpoint_data = raw_data

        from anemoi.training.checkpoint.formats import detect_format_from_data

        context.checkpoint_format = detect_format_from_data(raw_data)
        LOGGER.debug(
            "Detected checkpoint format '%s' from data keys",
            context.checkpoint_format,
        )


#: Metadata key. ``ResolveOnlySource`` sets it to show that Lightning loads the checkpoint.
CHECKPOINT_LOAD_OWNER = "checkpoint_load_owner"


class ResolveOnlySource(CheckpointSource):
    """Run only the ``resolve`` step of a source.

    The builder uses this for a resume, because Lightning loads the file.

    Parameters
    ----------
    source : CheckpointSource
        The source to resolve.
    """

    def __init__(self, source: CheckpointSource) -> None:
        self.source = source

    def resolve(self, context: CheckpointContext) -> Path | None:
        """Delegate to the wrapped source's resolve step."""
        return self.source.resolve(context)

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Resolve the checkpoint file for ``Trainer.fit(ckpt_path=)``. Do not load it.

        Raises
        ------
        CheckpointConfigError
            On rank 0, if the source resolves no file. Other ranks defer the error.
        """
        path = self.resolve(context)
        context.update_metadata(**{CHECKPOINT_LOAD_OWNER: "trainer"})
        if path is not None:
            LOGGER.info("Resume checkpoint resolved to %s; Trainer.fit(ckpt_path=) performs the load", path)
            return context

        if rank_zero_only.rank == 0:
            from anemoi.training.checkpoint.exceptions import CheckpointConfigError

            msg = (
                f"Resume configured, but {type(self.source).__name__} resolved no checkpoint file to hand "
                "Trainer.fit(ckpt_path=). Check the training.checkpoint.source configuration (a RunIdSource "
                "needs run_id), or add a loading strategy if you meant to start fresh training state."
            )
            raise CheckpointConfigError(msg)
        LOGGER.warning(
            "%s resolved no checkpoint on this rank; deferring the error to rank 0",
            type(self.source).__name__,
        )
        return context

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.source!r})"
