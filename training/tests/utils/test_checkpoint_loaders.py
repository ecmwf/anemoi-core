# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from anemoi.training.utils.checkpoint_loaders import CheckpointLoaderRegistry
from anemoi.training.utils.checkpoint_loaders import LocalCheckpointLoader
from anemoi.training.utils.checkpoint_loaders import RemoteCheckpointLoader
from anemoi.training.utils.checkpoint_loaders import load_checkpoint_from_source


class TestLocalCheckpointLoader:
    """Test LocalCheckpointLoader functionality."""

    def test_supports_local_path(self) -> None:
        """Test that local paths are supported."""
        loader = LocalCheckpointLoader()

        # Test Path object
        assert loader.supports_source(Path("/some/path"))

        # Test string path (should be supported even if doesn't exist)
        assert loader.supports_source("/some/local/path")

        # Test non-URL string
        assert loader.supports_source("model.ckpt")

    def test_does_not_support_urls(self) -> None:
        """Test that URLs are not supported by local loader."""
        loader = LocalCheckpointLoader()

        assert not loader.supports_source("http://example.com/model.ckpt")
        assert not loader.supports_source("https://example.com/model.ckpt")
        assert not loader.supports_source("s3://bucket/model.ckpt")

    def test_load_existing_checkpoint(self) -> None:
        """Test loading an existing checkpoint file."""
        loader = LocalCheckpointLoader()

        # Create a temporary checkpoint file
        test_data = {"model_state": {"layer.weight": torch.randn(3, 3)}}

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            torch.save(test_data, tmp_file.name)
            tmp_path = Path(tmp_file.name)

        try:
            loaded_data = loader.load_checkpoint(tmp_path)
            assert "model_state" in loaded_data
            assert "layer.weight" in loaded_data["model_state"]
        finally:
            tmp_path.unlink()

    def test_load_nonexistent_checkpoint(self) -> None:
        """Test error handling for non-existent checkpoint."""
        loader = LocalCheckpointLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_checkpoint("/nonexistent/path/model.ckpt")


class TestRemoteCheckpointLoader:
    """Test RemoteCheckpointLoader functionality."""

    def test_supports_remote_urls(self) -> None:
        """Test that remote URLs are supported."""
        loader = RemoteCheckpointLoader()

        assert loader.supports_source("http://example.com/model.ckpt")
        assert loader.supports_source("https://example.com/model.ckpt")
        assert loader.supports_source("s3://bucket/model.ckpt")
        assert loader.supports_source("gs://bucket/model.ckpt")
        assert loader.supports_source("azure://account.blob.core.windows.net/container/model.ckpt")

    def test_does_not_support_local_paths(self) -> None:
        """Test that local paths are not supported by remote loader."""
        loader = RemoteCheckpointLoader()

        assert not loader.supports_source(Path("/local/path"))
        assert not loader.supports_source("/local/path")
        assert not loader.supports_source("model.ckpt")

    def test_download_http(self) -> None:
        """Test HTTP download functionality."""
        loader = RemoteCheckpointLoader()

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                loader._download_http("http://example.com/model.ckpt", tmp_path)
                mock_retrieve.assert_called_once_with("http://example.com/model.ckpt", tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_download_s3(self) -> None:
        """Test S3 download functionality."""
        loader = RemoteCheckpointLoader()

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with patch("boto3.client") as mock_boto3:
                mock_client = Mock()
                mock_boto3.return_value = mock_client

                loader._download_s3("s3://my-bucket/path/to/model.ckpt", tmp_path)

                mock_boto3.assert_called_once_with("s3")
                mock_client.download_file.assert_called_once_with("my-bucket", "path/to/model.ckpt", str(tmp_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_download_s3_missing_dependency(self) -> None:
        """Test S3 download with missing boto3 dependency."""
        loader = RemoteCheckpointLoader()

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with (
                patch("boto3.client", side_effect=ImportError("No module named 'boto3'")),
                pytest.raises(ImportError, match="boto3 required for S3 downloads"),
            ):
                loader._download_s3("s3://bucket/model.ckpt", tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_unsupported_scheme(self) -> None:
        """Test error handling for unsupported URL schemes."""
        loader = RemoteCheckpointLoader()

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with pytest.raises(ValueError, match="Unsupported remote scheme"):
                loader._download_checkpoint("ftp://example.com/model.ckpt", tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestCheckpointLoaderRegistry:
    """Test CheckpointLoaderRegistry functionality."""

    def test_registry_has_default_loaders(self) -> None:
        """Test that registry comes with default loaders."""
        registry = CheckpointLoaderRegistry()

        # Should have at least the default loaders
        assert len(registry._loaders) >= 2

        # Test that it can find appropriate loaders
        local_loader = registry.get_loader("/local/path")
        assert isinstance(local_loader, LocalCheckpointLoader)

        remote_loader = registry.get_loader("http://example.com/model.ckpt")
        assert isinstance(remote_loader, RemoteCheckpointLoader)

    def test_register_custom_loader(self) -> None:
        """Test registering a custom loader."""
        registry = CheckpointLoaderRegistry()
        custom_loader = Mock()
        custom_loader.supports_source.return_value = False

        registry.register(custom_loader)
        assert custom_loader in registry._loaders

    def test_no_loader_found(self) -> None:
        """Test error handling when no loader supports the source."""
        registry = CheckpointLoaderRegistry()

        # Mock all loaders to return False for supports_source
        for loader in registry._loaders:
            loader.supports_source = Mock(return_value=False)

        with pytest.raises(ValueError, match="No loader found for source"):
            registry.get_loader("unsupported://source")

    def test_load_checkpoint_delegates_to_loader(self) -> None:
        """Test that load_checkpoint delegates to the appropriate loader."""
        registry = CheckpointLoaderRegistry()

        # Create a mock loader that supports our test source
        mock_loader = Mock()
        mock_loader.supports_source.return_value = True
        mock_loader.load_checkpoint.return_value = {"test": "data"}

        # Replace the loaders with our mock
        registry._loaders = [mock_loader]

        result = registry.load_checkpoint("test://source")

        mock_loader.supports_source.assert_called_once_with("test://source")
        mock_loader.load_checkpoint.assert_called_once_with("test://source")
        assert result == {"test": "data"}


class TestLoadCheckpointFromSource:
    """Test the convenience function."""

    def test_load_checkpoint_from_source_function(self) -> None:
        """Test the main convenience function."""
        test_data = {"model": "data"}

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
            torch.save(test_data, tmp_file.name)
            tmp_path = Path(tmp_file.name)

        try:
            loaded_data = load_checkpoint_from_source(tmp_path)
            assert loaded_data == test_data
        finally:
            tmp_path.unlink()
