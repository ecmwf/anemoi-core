# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""S3-compatible checkpoint source.

Downloads checkpoint data from an S3 bucket, then loads it into the
pipeline context. Supports AWS S3 and S3-compatible stores (MinIO,
Ceph, EWC) via ``anemoi.utils.remote.s3.s3_client``, which handles
endpoint URLs, credentials, and per-bucket configuration from
``~/.config/anemoi/settings.toml``.

Example
-------
>>> source = S3Source(url="s3://my-bucket/checkpoints/model.ckpt")
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


class S3Source(CheckpointSource):
    """Checkpoint source for S3-compatible storage.

    Downloads a checkpoint file from an S3 bucket to a temporary location,
    loads it with PyTorch, and cleans up the temporary file. S3 client
    creation is delegated to ``anemoi.utils.remote.s3.s3_client``, which
    handles endpoint URLs, credentials, and per-bucket configuration from
    ``~/.config/anemoi/settings.toml``.

    Parameters
    ----------
    url : str or None
        S3 URL in ``s3://bucket/key`` format. If None, the URL is read
        from ``context.config["url"]`` at process time.
    region : str or None
        AWS region for the bucket. If None, anemoi-utils resolves the
        endpoint from its ``object-storage`` config section.

    Examples
    --------
    >>> source = S3Source(url="s3://models/anemoi/pretrained.ckpt")
    >>> context = CheckpointContext()
    >>> result = await source.process(context)
    """

    def __init__(
        self,
        url: str | None = None,
        region: str | None = None,
    ) -> None:
        self.url = url
        self.region = region

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Download and load a checkpoint from S3.

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
            If the S3 object does not exist (NoSuchKey/NoSuchBucket)
        CheckpointSourceError
            If download fails (credentials, network, etc.)
        CheckpointLoadError
            If the downloaded file cannot be loaded by PyTorch
        """
        from anemoi.training.checkpoint.exceptions import CheckpointLoadError

        url = self._resolve_url(context)
        bucket, key = self._parse_s3_url(url)

        LOGGER.info("Downloading checkpoint from s3://%s/%s", bucket, key)

        s3_client = self._create_s3_client(bucket, key)

        # Download to temp file, load, clean up
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_fd:
            tmp_path = Path(tmp_fd.name)

        try:
            await self._download_from_s3(s3_client, bucket, key, tmp_path)

            try:
                raw_data = await asyncio.to_thread(
                    torch.load,
                    tmp_path,
                    weights_only=False,
                    map_location="cpu",
                )
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as e:
                raise CheckpointLoadError(tmp_path, e) from e

            self._load_and_populate(context, raw_data)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                LOGGER.debug("Cleaned up temporary file %s", tmp_path)

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

    def _create_s3_client(self, bucket: str, key: str) -> object:
        """Create S3 client via anemoi-utils.

        Uses ``anemoi.utils.remote.s3.s3_client`` which provides:
        - Per-bucket endpoint/credential config from settings.toml
        - Thread-safe client caching
        - Anonymous access fallback when no credentials are found
        """
        from anemoi.training.checkpoint.exceptions import CheckpointSourceError

        try:
            from anemoi.utils.remote.s3 import s3_client
        except ImportError as e:
            msg = "anemoi-utils with S3 support is required. Install with: pip install anemoi-utils[remote]"
            raise CheckpointSourceError(msg, f"s3://{bucket}/{key}") from e

        return s3_client(bucket, region=self.region)

    async def _download_from_s3(self, s3_client: object, bucket: str, key: str, tmp_path: Path) -> None:
        """Download file from S3, translating boto errors to checkpoint exceptions."""
        from botocore.exceptions import ClientError
        from botocore.exceptions import EndpointConnectionError
        from botocore.exceptions import NoCredentialsError

        from anemoi.training.checkpoint.exceptions import CheckpointSourceError

        try:
            await asyncio.to_thread(s3_client.download_file, bucket, key, str(tmp_path))
        except NoCredentialsError as e:
            msg = f"S3 credentials not found for s3://{bucket}/{key}"
            s3_path = f"s3://{bucket}/{key}"
            raise CheckpointSourceError(msg, s3_path) from e
        except ClientError as e:
            self._handle_client_error(e, bucket, key)
        except EndpointConnectionError as e:
            msg = f"Could not connect to S3 endpoint for bucket '{bucket}'"
            s3_path = f"s3://{bucket}/{key}"
            raise CheckpointSourceError(msg, s3_path) from e

    @staticmethod
    def _handle_client_error(e: Exception, bucket: str, key: str) -> None:
        """Translate boto3 ClientError to checkpoint exceptions."""
        from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
        from anemoi.training.checkpoint.exceptions import CheckpointSourceError

        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in {"NoSuchBucket", "NoSuchKey", "404"}:
            s3_path = f"s3://{bucket}/{key}"
            raise CheckpointNotFoundError(s3_path) from e
        s3_path = f"s3://{bucket}/{key}"
        if error_code in {"AccessDenied", "403"}:
            msg = f"Access denied to {s3_path}"
            raise CheckpointSourceError(msg, s3_path) from e
        msg = f"S3 download failed for {s3_path}: {e}"
        raise CheckpointSourceError(msg, s3_path) from e

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
