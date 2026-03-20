"""Tests for HTTPSource."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointSourceError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.http import HTTPSource


def test_http_source_extends_checkpoint_source() -> None:
    assert issubclass(HTTPSource, CheckpointSource)


@pytest.mark.asyncio
async def test_http_source_uses_download_with_retry() -> None:
    """MUST use Phase 1 download_with_retry(), NOT urllib.request.urlretrieve."""
    source = HTTPSource(url="https://example.com/model.ckpt")
    context = CheckpointContext()

    async def fake_download(url: str, dest: Path, **kwargs: object) -> None:  # noqa: ARG001
        torch.save({"state_dict": {}}, dest)

    with patch(
        "anemoi.training.checkpoint.utils.download_with_retry",
        side_effect=fake_download,
    ) as mock_dl:
        await source.process(context)
        mock_dl.assert_called_once()


@pytest.mark.asyncio
async def test_http_source_wraps_errors_as_checkpoint_source_error() -> None:
    """Network errors must surface as CheckpointSourceError."""
    source = HTTPSource(url="https://example.com/model.ckpt")
    context = CheckpointContext()

    async def failing_download(url: str, dest: Path, **kwargs: object) -> None:  # noqa: ARG001
        msg = "Download failed after retries"
        raise CheckpointSourceError(msg, url)

    with (
        patch(
            "anemoi.training.checkpoint.utils.download_with_retry",
            side_effect=failing_download,
        ),
        pytest.raises(CheckpointSourceError),
    ):
        await source.process(context)
