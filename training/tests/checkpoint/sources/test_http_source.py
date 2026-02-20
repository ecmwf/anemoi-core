# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for HTTPSource checkpoint source."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointSourceError
from anemoi.training.checkpoint.exceptions import CheckpointTimeoutError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.sources.http import HTTPSource

if TYPE_CHECKING:
    from typing import Any

_TEST_URL = "https://example.com/model.ckpt"
_DOWNLOAD_TARGET = "anemoi.training.checkpoint.utils.download_with_retry"


def _make_download_side_effect(state_dict: dict) -> AsyncMock:
    """Create a side effect that writes a valid checkpoint to dest.

    The returned callable has the same signature as ``download_with_retry``
    and writes a torch checkpoint containing *state_dict* to the ``dest``
    path argument.
    """

    async def _side_effect(
        url: str,  # noqa: ARG001
        dest: Path,
        **kwargs: Any,  # noqa: ARG001
    ) -> Path:
        torch.save({"state_dict": state_dict}, dest)
        return dest

    return AsyncMock(side_effect=_side_effect)


@pytest.fixture
def simple_state_dict() -> dict:
    return {"layer.weight": torch.randn(4, 3), "layer.bias": torch.randn(4)}


@pytest.fixture
def source() -> HTTPSource:
    return HTTPSource(url=_TEST_URL, max_retries=3, timeout=60)


class TestHTTPSourceInit:
    """Tests for HTTPSource.__init__ URL validation."""

    def test_valid_https_url(self) -> None:
        source = HTTPSource(url="https://models.ecmwf.int/model.ckpt")
        assert source.url == "https://models.ecmwf.int/model.ckpt"

    def test_valid_http_url(self) -> None:
        source = HTTPSource(url="http://example.com/model.ckpt")
        assert source.url == "http://example.com/model.ckpt"

    def test_invalid_scheme_raises(self) -> None:
        with pytest.raises(CheckpointConfigError, match="HTTP or HTTPS"):
            HTTPSource(url="ftp://example.com/model.ckpt")

    def test_no_scheme_raises(self) -> None:
        with pytest.raises(CheckpointConfigError, match="HTTP or HTTPS"):
            HTTPSource(url="/local/path/model.ckpt")

    def test_empty_host_raises(self) -> None:
        with pytest.raises(CheckpointConfigError, match="no host"):
            HTTPSource(url="https:///no-host.ckpt")

    def test_expected_checksum_stored(self) -> None:
        source = HTTPSource(url=_TEST_URL, expected_checksum="abc123")
        assert source.expected_checksum == "abc123"


class TestHTTPSourceProcess:
    """Tests for HTTPSource.process()."""

    @patch(_DOWNLOAD_TARGET)
    async def test_successful_download_and_load(
        self,
        mock_download: AsyncMock,
        source: HTTPSource,
        simple_state_dict: dict,
    ) -> None:
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect

        context = CheckpointContext()
        result = await source.process(context)

        mock_download.assert_awaited_once()
        assert result.checkpoint_data is not None
        assert "state_dict" in result.checkpoint_data
        assert result.metadata["source_type"] == "http"
        assert result.metadata["source_url"] == _TEST_URL

    @patch(_DOWNLOAD_TARGET)
    async def test_download_passes_retry_and_timeout(
        self,
        mock_download: AsyncMock,
        simple_state_dict: dict,
    ) -> None:
        """Verify max_retries and timeout are forwarded to download_with_retry."""
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect

        source = HTTPSource(url=_TEST_URL, max_retries=7, timeout=120)
        context = CheckpointContext()
        await source.process(context)

        _, kwargs = mock_download.call_args
        assert kwargs["max_retries"] == 7
        assert kwargs["timeout"] == 120

    @patch(_DOWNLOAD_TARGET)
    async def test_source_error_propagates(
        self,
        mock_download: AsyncMock,
        source: HTTPSource,
    ) -> None:
        """CheckpointSourceError from download_with_retry should propagate."""
        mock_download.side_effect = CheckpointSourceError(
            "http",
            _TEST_URL,
            ConnectionError("refused"),
        )

        context = CheckpointContext()

        with pytest.raises(CheckpointSourceError):
            await source.process(context)

    @patch(_DOWNLOAD_TARGET)
    async def test_timeout_error_propagates(
        self,
        mock_download: AsyncMock,
        source: HTTPSource,
    ) -> None:
        mock_download.side_effect = CheckpointTimeoutError(
            f"Download of {_TEST_URL}",
            60,
        )

        context = CheckpointContext()

        with pytest.raises(CheckpointTimeoutError):
            await source.process(context)

    @patch(_DOWNLOAD_TARGET)
    async def test_corrupt_download_raises_load_error(
        self,
        mock_download: AsyncMock,
        source: HTTPSource,
    ) -> None:
        """If downloaded file isn't a valid checkpoint, raise CheckpointLoadError."""

        async def _write_garbage(url: str, dest: Path, **kwargs: Any) -> Path:  # noqa: ARG001
            dest.write_text("not a checkpoint")
            return dest

        mock_download.side_effect = _write_garbage

        context = CheckpointContext()

        with pytest.raises(CheckpointLoadError):
            await source.process(context)

    @patch(_DOWNLOAD_TARGET)
    async def test_temp_file_cleaned_up_on_success(
        self,
        mock_download: AsyncMock,
        source: HTTPSource,
        simple_state_dict: dict,
    ) -> None:
        """Temp file should not remain after successful processing."""
        created_paths: list[Path] = []

        async def _track_and_save(url: str, dest: Path, **kwargs: Any) -> Path:  # noqa: ARG001
            created_paths.append(dest)
            torch.save({"state_dict": simple_state_dict}, dest)
            return dest

        mock_download.side_effect = _track_and_save

        context = CheckpointContext()
        await source.process(context)

        assert len(created_paths) == 1
        assert not created_paths[0].exists(), "Temp file should be deleted after success"

    @patch(_DOWNLOAD_TARGET)
    async def test_temp_file_cleaned_up_on_failure(
        self,
        mock_download: AsyncMock,
        source: HTTPSource,
    ) -> None:
        """Temp file should not remain even when loading fails."""
        created_paths: list[Path] = []

        async def _track_and_corrupt(url: str, dest: Path, **kwargs: Any) -> Path:  # noqa: ARG001
            created_paths.append(dest)
            dest.write_text("corrupt")
            return dest

        mock_download.side_effect = _track_and_corrupt

        context = CheckpointContext()

        with pytest.raises(CheckpointLoadError):
            await source.process(context)

        assert len(created_paths) == 1
        assert not created_paths[0].exists(), "Temp file should be deleted after failure"

    @patch(_DOWNLOAD_TARGET)
    @patch("anemoi.training.checkpoint.utils.calculate_checksum")
    async def test_checksum_mismatch_raises(
        self,
        mock_checksum: AsyncMock,
        mock_download: AsyncMock,
        simple_state_dict: dict,
    ) -> None:
        """Checksum mismatch should raise CheckpointValidationError."""
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect
        mock_checksum.return_value = "wrong_checksum"

        source = HTTPSource(url=_TEST_URL, expected_checksum="correct_checksum")
        context = CheckpointContext()

        with pytest.raises(CheckpointValidationError, match="Checksum mismatch"):
            await source.process(context)

    @patch(_DOWNLOAD_TARGET)
    @patch("anemoi.training.checkpoint.utils.calculate_checksum")
    async def test_checksum_match_succeeds(
        self,
        mock_checksum: AsyncMock,
        mock_download: AsyncMock,
        simple_state_dict: dict,
    ) -> None:
        """Matching checksum should proceed normally."""
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect
        mock_checksum.return_value = "correct_checksum"

        source = HTTPSource(url=_TEST_URL, expected_checksum="correct_checksum")
        context = CheckpointContext()
        result = await source.process(context)

        assert result.checkpoint_data is not None


class TestHTTPSourceSupports:
    """Tests for HTTPSource.supports()."""

    def test_http_url_returns_true(self) -> None:
        assert HTTPSource.supports("http://example.com/model.ckpt") is True

    def test_https_url_returns_true(self) -> None:
        assert HTTPSource.supports("https://models.ecmwf.int/anemoi.ckpt") is True

    def test_s3_url_returns_false(self) -> None:
        assert HTTPSource.supports("s3://bucket/model.ckpt") is False

    def test_gcs_url_returns_false(self) -> None:
        assert HTTPSource.supports("gs://bucket/model.ckpt") is False

    def test_local_path_string_returns_false(self) -> None:
        assert HTTPSource.supports("/models/model.ckpt") is False

    def test_path_object_returns_false(self) -> None:
        assert HTTPSource.supports(Path("/models/model.ckpt")) is False
