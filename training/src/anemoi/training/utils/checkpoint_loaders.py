# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


class CheckpointLoader(ABC):
    """Abstract base class for loading model checkpoints."""

    @abstractmethod
    def load_checkpoint(self, source: str | Path) -> dict:
        """Load checkpoint from source.

        Parameters
        ----------
        source : str | Path
            Checkpoint source (path, URL, etc.)

        Returns
        -------
        dict
            Loaded checkpoint dictionary
        """
        ...

    @abstractmethod
    def supports_source(self, source: str | Path) -> bool:
        """Check if this loader supports the given source.

        Parameters
        ----------
        source : str | Path
            Checkpoint source to check

        Returns
        -------
        bool
            True if this loader can handle the source
        """
        ...


class LocalCheckpointLoader(CheckpointLoader):
    """Load checkpoints from local filesystem."""

    def supports_source(self, source: str | Path) -> bool:
        """Check if source is a local path."""
        if isinstance(source, Path):
            return True
        try:
            path = Path(source)
            return path.exists() or not urlparse(str(source)).scheme
        except (ValueError, OSError):
            return False

    def load_checkpoint(self, source: str | Path) -> dict:
        """Load checkpoint from local filesystem."""
        import torch

        path = Path(source)
        if not path.exists():
            msg = f"Checkpoint not found: {path}"
            raise FileNotFoundError(msg)

        LOGGER.info("Loading checkpoint from local path: %s", path)
        return torch.load(path, weights_only=False, map_location="cpu")


class RemoteCheckpointLoader(CheckpointLoader):
    """Load checkpoints from remote sources (S3, HTTP, etc.)."""

    def supports_source(self, source: str | Path) -> bool:
        """Check if source is a remote URL."""
        if isinstance(source, Path):
            return False
        try:
            parsed = urlparse(str(source))
            return bool(parsed.scheme and parsed.scheme in {"http", "https", "s3", "gs", "azure"})
        except (ValueError, OSError):
            return False

    def load_checkpoint(self, source: str | Path) -> dict:
        """Load checkpoint from remote source."""
        import tempfile
        from pathlib import Path

        import torch

        LOGGER.info("Loading checkpoint from remote source: %s", source)

        # Create temporary file to download checkpoint
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            self._download_checkpoint(str(source), tmp_path)
            return torch.load(tmp_path, weights_only=False, map_location="cpu")
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    def _download_checkpoint(self, source: str, dest_path: Path) -> None:
        """Download checkpoint from remote source."""
        parsed = urlparse(source)

        if parsed.scheme in {"http", "https"}:
            self._download_http(source, dest_path)
        elif parsed.scheme == "s3":
            self._download_s3(source, dest_path)
        elif parsed.scheme in {"gs", "gcs"}:
            self._download_gcs(source, dest_path)
        elif parsed.scheme in {"azure", "az"}:
            self._download_azure(source, dest_path)
        else:
            msg = f"Unsupported remote scheme: {parsed.scheme}"
            raise ValueError(msg)

    def _download_http(self, url: str, dest_path: Path) -> None:
        """Download from HTTP/HTTPS."""
        import urllib.request

        urllib.request.urlretrieve(url, dest_path)  # noqa: S310

    def _download_s3(self, s3_url: str, dest_path: Path) -> None:
        """Download from S3."""
        try:
            import boto3
        except ImportError as e:
            msg = "boto3 required for S3 downloads. Install with: pip install boto3"
            raise ImportError(msg) from e

        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3_client = boto3.client("s3")
        s3_client.download_file(bucket, key, str(dest_path))

    def _download_gcs(self, gcs_url: str, dest_path: Path) -> None:
        """Download from Google Cloud Storage."""
        try:
            from google.cloud import storage
        except ImportError as e:
            msg = "google-cloud-storage required for GCS downloads. Install with: pip install google-cloud-storage"
            raise ImportError(msg) from e

        parsed = urlparse(gcs_url)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(dest_path))

    def _download_azure(self, azure_url: str, dest_path: Path) -> None:
        """Download from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            msg = "azure-storage-blob required for Azure downloads. Install with: pip install azure-storage-blob"
            raise ImportError(msg) from e

        # Parse Azure URL format: azure://account.blob.core.windows.net/container/blob
        parsed = urlparse(azure_url)
        account_url = f"https://{parsed.netloc}"
        container_name = parsed.path.split("/")[1]
        blob_name = "/".join(parsed.path.split("/")[2:])

        blob_service_client = BlobServiceClient(account_url=account_url)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        dest_path.write_bytes(blob_client.download_blob().readall())


class CheckpointLoaderRegistry:
    """Registry for checkpoint loaders."""

    def __init__(self) -> None:
        self._loaders: list[CheckpointLoader] = []
        # Register default loaders
        self.register(RemoteCheckpointLoader())
        self.register(LocalCheckpointLoader())

    def register(self, loader: CheckpointLoader) -> None:
        """Register a checkpoint loader."""
        self._loaders.append(loader)

    def get_loader(self, source: str | Path) -> CheckpointLoader:
        """Get appropriate loader for source."""
        for loader in self._loaders:
            if loader.supports_source(source):
                return loader
        msg = f"No loader found for source: {source}"
        raise ValueError(msg)

    def load_checkpoint(self, source: str | Path) -> dict:
        """Load checkpoint using appropriate loader."""
        loader = self.get_loader(source)
        return loader.load_checkpoint(source)


# Global registry instance
checkpoint_registry = CheckpointLoaderRegistry()


def load_checkpoint_from_source(source: str | Path) -> dict:
    """Load checkpoint from any supported source.

    Parameters
    ----------
    source : str | Path
        Checkpoint source (local path, S3 URL, HTTP URL, etc.)

    Returns
    -------
    dict
        Loaded checkpoint dictionary
    """
    return checkpoint_registry.load_checkpoint(source)
