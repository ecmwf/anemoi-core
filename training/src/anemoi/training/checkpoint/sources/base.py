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
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import torch

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    from collections.abc import Callable

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class CheckpointSource(PipelineStage):
    """Abstract base class for all checkpoint sources.

    Checkpoint sources form the acquisition layer of the checkpoint
    pipeline. They are responsible for obtaining checkpoint data from
    a specific source type (local file, cloud storage, HTTP endpoint,
    etc.) and populating the context for downstream stages.

    Subclasses implement ``process``. Use ``_load_from_path`` to load a local file
    and ``_load_from_download`` to load a remote file.
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
    ...         self._load_from_download(context, lambda path: download(self.url, path))
    ...         return context
    """

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

    def _load_from_download(self, context: CheckpointContext, download: Callable[[Path], None]) -> None:
        """Download the checkpoint to a temporary file, load it, then delete the file.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context (mutated in place)
        download : Callable[[Path], None]
            Writes the checkpoint to the given path.
        """
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            download(tmp_path)
            self._load_from_path(context, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

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
