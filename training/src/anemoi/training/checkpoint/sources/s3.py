# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""S3-compatible checkpoint source (AWS S3, MinIO, Ceph, EWC).

Example
-------
>>> source = S3Source(url="s3://my-bucket/checkpoints/model.ckpt")
>>> context = CheckpointContext()
>>> result = source.process(context)
>>> assert result.checkpoint_data is not None
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from anemoi.training.checkpoint.sources.base import CheckpointSource

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class S3Source(CheckpointSource):
    """Checkpoint source for S3-compatible storage.

    Downloads a checkpoint file from an S3 bucket to a node-local temporary
    location, loads it with PyTorch and deletes the file straight after. A
    resume cannot use this source, because ``Trainer.fit(ckpt_path=)`` needs a
    local file: download the checkpoint and use ``LocalSource``. Download is
    delegated to ``anemoi.utils.remote.s3.download_file`` (obstore-backed),
    which handles endpoint URLs, credentials, and per-bucket configuration
    from ``~/.config/anemoi/settings.toml``.

    Parameters
    ----------
    url : str or None
        S3 URL in ``s3://bucket/key`` format. If None, the URL is read
        from ``context.config["url"]`` at process time.

    Examples
    --------
    >>> source = S3Source(url="s3://models/anemoi/pretrained.ckpt")
    >>> context = CheckpointContext()
    >>> result = source.process(context)
    """

    def __init__(self, url: str | None = None) -> None:
        self.url = url

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Download and load a checkpoint from S3.

        The download goes to a temporary file that is deleted straight after
        loading, also when the download or the load fails.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context. If ``self.url`` is None, the URL is read
            from ``context.config["url"]``.

        Returns
        -------
        CheckpointContext
            Context with ``checkpoint_data``, ``checkpoint_format``,
            and source metadata populated.

        Raises
        ------
        CheckpointNotFoundError
            If the S3 object does not exist.
        CheckpointSourceError
            If download fails (credentials, network, missing optional dep).
        CheckpointLoadError
            If the downloaded file cannot be loaded by PyTorch.
        """
        url = self._resolve_url(context)
        bucket, key = self._parse_s3_url(url)

        LOGGER.info("Downloading checkpoint from %s", url)

        # Deleted in the ``finally`` below, once the checkpoint is loaded.
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_fd:
            tmp_path = Path(tmp_fd.name)

        try:
            self._download_from_s3(url, tmp_path)
            self._load_from_path(context, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        context.update_metadata(
            source_type="s3",
            source_url=url,
            s3_bucket=bucket,
            s3_key=key,
        )

        return context

    def _resolve_url(self, context: CheckpointContext) -> str:
        """Resolve S3 URL from constructor or context config."""
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        url = self.url
        if url is None:
            config = context.config
            if isinstance(config, dict):
                url = config.get("url")
            elif hasattr(config, "url"):
                url = config.url
            if url is None:
                msg = "S3Source requires a URL. Set url= in constructor or context.config['url']."
                raise CheckpointConfigError(msg)
        return url

    @staticmethod
    def _download_from_s3(url: str, tmp_path: Path) -> None:
        """Download an S3 object to ``tmp_path`` via anemoi-utils.

        Maps obstore/anemoi-utils errors to checkpoint exception types.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
        from anemoi.training.checkpoint.exceptions import CheckpointSourceError

        try:
            from anemoi.utils.remote.s3 import download_file
        except ImportError as e:
            msg = "S3Source requires anemoi-utils[s3]: pip install 'anemoi-utils[s3]'"
            raise CheckpointSourceError(msg, url) from e

        try:
            download_file(
                url,
                str(tmp_path),
                True,  # overwrite — tmp file already exists
                False,  # resume
                0,  # verbosity
            )
        except FileNotFoundError as e:
            raise CheckpointNotFoundError(url) from e
        except ImportError as e:
            # obstore is imported lazily inside anemoi-utils download_file
            msg = f"S3 download requires obstore (pip install 'anemoi-utils[s3]'): {e}"
            raise CheckpointSourceError(msg, url) from e
        except (OSError, ValueError, RuntimeError) as e:
            msg = f"S3 download failed for {url}: {e}"
            raise CheckpointSourceError(msg, url) from e

    @staticmethod
    def _parse_s3_url(url: str) -> tuple[str, str]:
        """Parse an S3 URL into bucket and key.

        Parameters
        ----------
        url : str
            S3 URL in ``s3://bucket/key`` format

        Returns
        -------
        tuple[str, str]
            (bucket, key) pair

        Raises
        ------
        CheckpointConfigError
            If the URL is not a valid S3 URL
        """
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        parsed = urlparse(url)
        if parsed.scheme != "s3":
            msg = f"Expected s3:// URL scheme, got: {parsed.scheme}://"
            raise CheckpointConfigError(msg)
        bucket = parsed.netloc
        if not bucket:
            msg = f"S3 URL missing bucket name: {url}"
            raise CheckpointConfigError(msg)
        key = parsed.path.lstrip("/")
        if not key:
            msg = f"S3 URL missing object key: {url}"
            raise CheckpointConfigError(msg)
        return bucket, key

    @staticmethod
    def supports(source: str | Path) -> bool:
        """Check whether a source string is an S3 URL.

        Parameters
        ----------
        source : str or Path
            Source identifier to check

        Returns
        -------
        bool
            True if the source is an s3:// URL
        """
        if isinstance(source, Path):
            return False
        return urlparse(str(source)).scheme == "s3"
