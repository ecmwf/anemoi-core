# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Local filesystem checkpoint source.

Example
-------
>>> source = LocalSource()
>>> context = CheckpointContext(checkpoint_path=Path("/models/epoch_50.ckpt"))
>>> result = source.process(context)
>>> assert result.checkpoint_data is not None
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from anemoi.training.checkpoint.sources.base import CheckpointSource

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class LocalSource(CheckpointSource):
    """Checkpoint source for local filesystem access.

    Loads checkpoints from local file paths. This is the simplest source
    and serves as the default when no remote protocol is detected.

    The checkpoint is loaded with ``weights_only=False`` (to support full
    Lightning checkpoints with metadata) and ``map_location="cpu"`` (to
    ensure compatibility across hardware configurations).

    Examples
    --------
    >>> source = LocalSource()
    >>> context = CheckpointContext(checkpoint_path=Path("/models/model.ckpt"))
    >>> result = source.process(context)
    >>> result.checkpoint_format
    'lightning'

    If ``path`` is set, it replaces ``context.checkpoint_path``:

    >>> source = LocalSource(path="/models/epoch_50.ckpt")
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = path

    def resolve(self, context: CheckpointContext) -> Path:
        """Resolve the checkpoint path to an absolute file and set it on the context.

        A relative path is relative to the launch directory. Use an absolute path.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context. Its ``checkpoint_path`` is used if ``path`` is not set.

        Returns
        -------
        Path
            The absolute checkpoint file path.

        Raises
        ------
        CheckpointConfigError
            If neither ``path`` nor ``context.checkpoint_path`` is set
        CheckpointNotFoundError
            If the checkpoint file does not exist
        """
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError
        from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError

        if self.path is not None:
            context.checkpoint_path = self.path

        if context.checkpoint_path is None:
            msg = "LocalSource requires an explicit 'path' or checkpoint_path to be set on the context"
            raise CheckpointConfigError(msg)

        path = Path(context.checkpoint_path).expanduser().resolve()

        if not path.exists():
            raise CheckpointNotFoundError(path)

        context.checkpoint_path = path
        return path

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Resolve and load checkpoint data from a local file.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_path`` set to a local
            file path (str or Path).

        Returns
        -------
        CheckpointContext
            Context with ``checkpoint_path``, ``checkpoint_data``,
            ``checkpoint_format``, and source metadata populated.

        Raises
        ------
        CheckpointNotFoundError
            If the checkpoint file does not exist
        CheckpointLoadError
            If the file cannot be loaded by PyTorch
        """
        path = self.resolve(context)

        LOGGER.info("Loading checkpoint from local path: %s", path)
        self._load_from_path(context, path)

        context.update_metadata(
            source_type="local",
            source_path=str(path),
        )

        return context

    @staticmethod
    def supports(source: str | Path) -> bool:
        """Check whether a source string refers to a local file.

        Returns ``True`` for :class:`~pathlib.Path` objects, existing
        file paths, and strings that do not contain a URL scheme.

        Parameters
        ----------
        source : str or Path
            Source identifier to check

        Returns
        -------
        bool
            True if the source should be handled by LocalSource
        """
        if isinstance(source, Path):
            return True
        path = Path(source)
        if path.exists():
            return True
        return not urlparse(str(source)).scheme
