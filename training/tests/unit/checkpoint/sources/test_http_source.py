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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointSourceError
from anemoi.training.checkpoint.exceptions import CheckpointTimeoutError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.http import HTTPSource

if TYPE_CHECKING:
    from typing import Any

_TEST_URL = "https://example.com/model.ckpt"
_DOWNLOAD_TARGET = "multiurl.download"


def _make_download_side_effect(state_dict: dict) -> MagicMock:
    """Create a ``multiurl.download`` side effect that writes a checkpoint with ``state_dict``."""

    def _side_effect(
        url: str,  # noqa: ARG001
        dest: str,
        **kwargs: Any,  # noqa: ARG001
    ) -> None:
        torch.save({"state_dict": state_dict}, dest)

    return MagicMock(side_effect=_side_effect)


@pytest.fixture
def simple_state_dict() -> dict:
    return {"layer.weight": torch.randn(4, 3), "layer.bias": torch.randn(4)}


@pytest.fixture
def source() -> HTTPSource:
    return HTTPSource(url=_TEST_URL, max_retries=3, timeout=60)


def test_http_source_extends_checkpoint_source() -> None:
    assert issubclass(HTTPSource, CheckpointSource)


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
    def test_successful_download_and_load(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
        simple_state_dict: dict,
    ) -> None:
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect

        context = CheckpointContext()
        result = source.process(context)

        mock_download.assert_called_once()
        assert result.checkpoint_data is not None
        assert "state_dict" in result.checkpoint_data
        assert result.metadata["source_type"] == "http"
        assert result.metadata["source_url"] == _TEST_URL

    @patch(_DOWNLOAD_TARGET)
    def test_download_passes_retry_and_timeout(
        self,
        mock_download: MagicMock,
        simple_state_dict: dict,
    ) -> None:
        """Verify max_retries and timeout are forwarded to multiurl."""
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect

        source = HTTPSource(url=_TEST_URL, max_retries=7, timeout=120)
        context = CheckpointContext()
        source.process(context)

        _, kwargs = mock_download.call_args
        assert kwargs["maximum_retries"] == 7
        assert kwargs["timeout"] == 120

    @patch(_DOWNLOAD_TARGET)
    def test_source_error_propagates(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
    ) -> None:
        """A request error becomes a CheckpointSourceError."""
        mock_download.side_effect = requests.ConnectionError("refused")

        context = CheckpointContext()

        with pytest.raises(CheckpointSourceError):
            source.process(context)

    @patch(_DOWNLOAD_TARGET)
    def test_timeout_error_propagates(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
    ) -> None:
        mock_download.side_effect = requests.ReadTimeout("slow")

        context = CheckpointContext()

        with pytest.raises(CheckpointTimeoutError):
            source.process(context)

    @patch(_DOWNLOAD_TARGET)
    def test_corrupt_download_raises_load_error(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
    ) -> None:
        """If downloaded file isn't a valid checkpoint, raise CheckpointLoadError."""

        def _write_garbage(url: str, dest: str, **kwargs: Any) -> None:  # noqa: ARG001
            Path(dest).write_text("not a checkpoint")

        mock_download.side_effect = _write_garbage

        context = CheckpointContext()

        with pytest.raises(CheckpointLoadError):
            source.process(context)

    @patch(_DOWNLOAD_TARGET)
    def test_download_is_deleted_after_loading(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
        simple_state_dict: dict,
    ) -> None:
        """The download is loaded and then deleted."""
        created_paths: list[Path] = []

        def _track_and_save(url: str, dest: str, **kwargs: Any) -> None:  # noqa: ARG001
            created_paths.append(Path(dest))
            torch.save({"state_dict": simple_state_dict}, dest)

        mock_download.side_effect = _track_and_save

        result = source.process(CheckpointContext())

        assert len(created_paths) == 1
        assert result.checkpoint_data is not None
        assert not created_paths[0].exists()

    @patch(_DOWNLOAD_TARGET)
    def test_download_is_deleted_on_load_failure(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
    ) -> None:
        """A download that fails to load is also deleted."""
        created_paths: list[Path] = []

        def _track_and_corrupt(url: str, dest: str, **kwargs: Any) -> None:  # noqa: ARG001
            created_paths.append(Path(dest))
            Path(dest).write_text("corrupt")

        mock_download.side_effect = _track_and_corrupt

        with pytest.raises(CheckpointLoadError):
            source.process(CheckpointContext())

        assert len(created_paths) == 1
        assert not created_paths[0].exists()

    @patch(_DOWNLOAD_TARGET)
    def test_failed_download_leaves_no_file_behind(
        self,
        mock_download: MagicMock,
        source: HTTPSource,
    ) -> None:
        """A failed download is removed."""
        created_paths: list[Path] = []

        def _track_and_fail(url: str, dest: str, **kwargs: Any) -> None:
            del url, kwargs
            created_paths.append(Path(dest))
            msg = "connection reset"
            raise requests.ConnectionError(msg)

        mock_download.side_effect = _track_and_fail

        context = CheckpointContext()
        with pytest.raises(CheckpointSourceError):
            source.process(context)

        assert len(created_paths) == 1
        assert not created_paths[0].exists()

    @patch(_DOWNLOAD_TARGET)
    @patch("anemoi.training.checkpoint.utils.calculate_checksum")
    def test_checksum_mismatch_raises(
        self,
        mock_checksum: MagicMock,
        mock_download: MagicMock,
        simple_state_dict: dict,
    ) -> None:
        """Checksum mismatch raises CheckpointValidationError and removes the file."""
        created_paths: list[Path] = []
        save = _make_download_side_effect(simple_state_dict).side_effect

        def _track_and_save(url: str, dest: str, **kwargs: Any) -> None:
            created_paths.append(Path(dest))
            save(url, dest, **kwargs)

        mock_download.side_effect = _track_and_save
        mock_checksum.return_value = "wrong_checksum"

        source = HTTPSource(url=_TEST_URL, expected_checksum="correct_checksum")
        context = CheckpointContext()

        with pytest.raises(CheckpointValidationError, match="Checksum mismatch"):
            source.process(context)

        assert len(created_paths) == 1
        assert not created_paths[0].exists()

    @patch(_DOWNLOAD_TARGET)
    @patch("anemoi.training.checkpoint.utils.calculate_checksum")
    def test_checksum_match_succeeds(
        self,
        mock_checksum: MagicMock,
        mock_download: MagicMock,
        simple_state_dict: dict,
    ) -> None:
        """Matching checksum should proceed normally."""
        mock_download.side_effect = _make_download_side_effect(simple_state_dict).side_effect
        mock_checksum.return_value = "correct_checksum"

        source = HTTPSource(url=_TEST_URL, expected_checksum="correct_checksum")
        context = CheckpointContext()
        result = source.process(context)

        assert result.checkpoint_data is not None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("http://example.com/model.ckpt", True),
        ("https://models.ecmwf.int/anemoi.ckpt", True),
        ("s3://bucket/model.ckpt", False),
        ("gs://bucket/model.ckpt", False),
        ("/models/model.ckpt", False),
        (Path("/models/model.ckpt"), False),
    ],
)
def test_http_source_supports(source: str | Path, expected: bool) -> None:
    assert HTTPSource.supports(source) is expected
