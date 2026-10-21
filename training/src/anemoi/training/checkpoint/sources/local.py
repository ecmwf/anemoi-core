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

import errno
import logging
import os
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

    An explicit ``path`` may be configured directly (the canonical way to load a
    specific checkpoint file); when set it is used instead of
    ``context.checkpoint_path``:

    >>> source = LocalSource(path="/models/epoch_50.ckpt")
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = path

    def resolve(self, context: CheckpointContext) -> Path:
        """Resolve the configured or context path to an existing, canonical file.

        Applies ``expanduser().resolve()`` and publishes the result on
        ``context.checkpoint_path``, so what the trainer later hands to
        ``Trainer.fit(ckpt_path=)`` is the file that was checked. A relative path
        resolves against the process working directory, which is where the
        training command was launched (the shipped config leaves
        ``hydra.job.chdir`` unset); prefer an absolute path.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context; ``checkpoint_path`` is used when no explicit
            ``path`` was configured.

        Returns
        -------
        Path
            The canonical checkpoint file.

        Raises
        ------
        CheckpointConfigError
            If neither ``path`` nor ``context.checkpoint_path`` is set
        CheckpointNotFoundError
            If the checkpoint file does not exist
        CheckpointLoadError
            If the checkpoint file exists but is not readable by this process
        """
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError
        from anemoi.training.checkpoint.exceptions import CheckpointLoadError
        from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError

        raw_path = self.path if self.path is not None else context.checkpoint_path

        if raw_path is None:
            msg = "LocalSource requires an explicit 'path' or checkpoint_path to be set on the context"
            raise CheckpointConfigError(msg)

        path = Path(raw_path).expanduser().resolve()

        if not path.exists():
            raise CheckpointNotFoundError(path)

        # Checked here rather than left to torch.load so an unreadable file (e.g. wrong
        # permissions) is reported as such, and RunIdSource can defer it to rank 0.
        if not os.access(path, os.R_OK):
            raise CheckpointLoadError(path, PermissionError(errno.EACCES, "Checkpoint file is not readable", str(path)))

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
            If the file is not readable or cannot be loaded by PyTorch
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
