# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Local filesystem checkpoint source.

Loads checkpoint data from a local file path and populates the pipeline
context with the loaded data and detected format.

Example
-------
>>> source = LocalSource()
>>> context = CheckpointContext(checkpoint_path=Path("/models/epoch_50.ckpt"))
>>> result = await source.process(context)
>>> assert result.checkpoint_data is not None
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch

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
    >>> result = await source.process(context)
    >>> result.checkpoint_format
    'lightning'
    """

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load checkpoint data from a local file.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_path`` set to a local
            file path (str or Path).

        Returns
        -------
        CheckpointContext
            Context with ``checkpoint_data``, ``checkpoint_format``,
            and source metadata populated.

        Raises
        ------
        CheckpointNotFoundError
            If the checkpoint file does not exist
        CheckpointLoadError
            If the file cannot be loaded by PyTorch
        """
        from anemoi.training.checkpoint.exceptions import CheckpointLoadError
        from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError

        path = Path(context.checkpoint_path)

        if not path.exists():
            raise CheckpointNotFoundError(path)

        LOGGER.info("Loading checkpoint from local path: %s", path)

        try:
            raw_data = torch.load(path, weights_only=False, map_location="cpu")
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as e:
            raise CheckpointLoadError(path, e) from e

        self._load_and_populate(context, raw_data)

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
