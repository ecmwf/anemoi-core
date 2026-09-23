# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""HTTP/HTTPS checkpoint source.

Downloads checkpoint data from an HTTP or HTTPS URL using async
download with retry and exponential backoff, then loads it into
the pipeline context.

Example
-------
>>> source = HTTPSource(url="https://models.ecmwf.int/model.ckpt")
>>> context = CheckpointContext()
>>> result = await source.process(context)
>>> assert result.checkpoint_data is not None
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch

from anemoi.training.checkpoint.sources.base import CheckpointSource

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class HTTPSource(CheckpointSource):
    """Checkpoint source for HTTP/HTTPS URLs.

    Downloads a checkpoint file to a temporary location using
    :func:`~anemoi.training.checkpoint.utils.download_with_retry`,
    which provides async streaming with exponential-backoff retry.
    The downloaded file is loaded with PyTorch and the temporary
    file is cleaned up afterwards.

    The URL is provided at construction time (compatible with Hydra
    instantiation) rather than via ``context.checkpoint_path``,
    because ``CheckpointContext`` normalises paths through
    :class:`~pathlib.Path` which strips the ``//`` from URL schemes.

    Parameters
    ----------
    url : str
        HTTP or HTTPS URL to download the checkpoint from
    max_retries : int
        Maximum download retry attempts (default: 3)
    timeout : int
        Per-attempt timeout in seconds (default: 300)
    expected_checksum : str or None
        Expected SHA-256 checksum of the downloaded file. If provided,
        the download is verified before loading. If None, a warning is
        logged that integrity was not verified.

    Examples
    --------
    >>> source = HTTPSource(
    ...     url="https://models.ecmwf.int/anemoi.ckpt",
    ...     max_retries=5,
    ...     timeout=600,
    ... )
    >>> context = CheckpointContext()
    >>> result = await source.process(context)
    """

    def __init__(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 300,
        expected_checksum: str | None = None,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            from anemoi.training.checkpoint.exceptions import CheckpointConfigError

            msg = f"HTTPSource requires an HTTP or HTTPS URL, got scheme '{parsed.scheme}' in: {url}"
            raise CheckpointConfigError(msg)
        if not parsed.netloc:
            from anemoi.training.checkpoint.exceptions import CheckpointConfigError

            msg = f"HTTPSource URL has no host: {url}"
            raise CheckpointConfigError(msg)

        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.expected_checksum = expected_checksum

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Download and load a checkpoint from an HTTP/HTTPS URL.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context. The ``checkpoint_path`` field will be
            updated to point at the downloaded temporary file during
            format detection.

        Returns
        -------
        CheckpointContext
            Context with ``checkpoint_data``, ``checkpoint_format``,
            and source metadata populated.

        Raises
        ------
        CheckpointSourceError
            If the download fails after all retries
        CheckpointTimeoutError
            If the download times out
        CheckpointLoadError
            If the downloaded file cannot be loaded by PyTorch
        """
        from anemoi.training.checkpoint.exceptions import CheckpointLoadError
        from anemoi.training.checkpoint.utils import calculate_checksum
        from anemoi.training.checkpoint.utils import download_with_retry

        LOGGER.info("Downloading checkpoint from %s", self.url)

        # Create a named temp file that persists until we explicitly delete it
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_fd:
            tmp_path = Path(tmp_fd.name)

        try:
            await download_with_retry(
                url=self.url,
                dest=tmp_path,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )

            # Checksum verification
            if self.expected_checksum is not None:
                actual = calculate_checksum(tmp_path)
                if actual != self.expected_checksum:
                    from anemoi.training.checkpoint.exceptions import CheckpointValidationError

                    msg = (
                        f"Checksum mismatch for checkpoint downloaded from {self.url}. "
                        f"Expected {self.expected_checksum}, got {actual}."
                    )
                    raise CheckpointValidationError(msg)
                LOGGER.info("Checksum verified for %s", self.url)
            else:
                LOGGER.warning(
                    "No checksum provided for HTTP checkpoint download from %s. Integrity not verified.",
                    self.url,
                )

            try:
                # SECURITY: weights_only=False is required for Anemoi checkpoints that
                # contain non-tensor metadata (hyper_parameters, callbacks, etc.).
                raw_data = await asyncio.to_thread(torch.load, tmp_path, weights_only=False, map_location="cpu")
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as e:
                raise CheckpointLoadError(tmp_path, e) from e

            self._load_and_populate(context, raw_data)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                LOGGER.debug("Cleaned up temporary file %s", tmp_path)

        context.update_metadata(
            source_type="http",
            source_url=self.url,
        )

        return context

    @staticmethod
    def supports(source: str | Path) -> bool:
        """Check whether a source string is an HTTP or HTTPS URL.

        Parameters
        ----------
        source : str or Path
            Source identifier to check

        Returns
        -------
        bool
            True if the source is an HTTP/HTTPS URL
        """
        if isinstance(source, Path):
            return False
        parsed = urlparse(str(source))
        return parsed.scheme in {"http", "https"}
