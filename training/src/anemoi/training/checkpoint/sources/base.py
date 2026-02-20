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

Example
-------
>>> class LocalSource(CheckpointSource):
...     async def process(self, context: CheckpointContext) -> CheckpointContext:
...         raw_data = torch.load(context.checkpoint_path, weights_only=False, map_location="cpu")
...         self._load_and_populate(context, raw_data)
...         return context
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class CheckpointSource(PipelineStage):
    """Abstract base class for all checkpoint sources.

    Checkpoint sources form the acquisition layer of the checkpoint
    pipeline. They are responsible for obtaining checkpoint data from
    a specific source type (local file, cloud storage, HTTP endpoint,
    etc.) and populating the context for downstream stages.

    Subclasses must implement the ``process`` method to handle their
    specific source type. The ``_load_and_populate`` convenience method
    is provided to standardise how raw checkpoint data is attached to
    the context with format detection.

    Parameters
    ----------
    None

    Examples
    --------
    >>> class S3Source(CheckpointSource):
    ...     def __init__(self, bucket: str, key: str):
    ...         self.bucket = bucket
    ...         self.key = key
    ...
    ...     async def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         raw_data = await self._download_from_s3()
    ...         return self._load_and_populate(context, raw_data)
    """

    @abstractmethod
    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Acquire checkpoint data from source and populate context.

        Implementations should:
        1. Validate that required source information is available
        2. Fetch/load raw checkpoint data from the source
        3. Use ``_load_and_populate`` to attach data to context
        4. Add source-specific metadata for tracking

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
