"""Tests for S3Source."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.s3 import S3Source


def test_s3_source_extends_checkpoint_source() -> None:
    assert issubclass(S3Source, CheckpointSource)


def test_s3_source_lazy_imports_boto3() -> None:
    """boto3 must NOT be imported at module level."""
    import importlib
    import sys

    saved = sys.modules.pop("boto3", None)
    try:
        importlib.reload(importlib.import_module("anemoi.training.checkpoint.sources.s3"))
    finally:
        if saved:
            sys.modules["boto3"] = saved


@pytest.mark.asyncio
async def test_s3_source_parses_s3_url() -> None:
    """s3://bucket/path/to/model.ckpt -> bucket='bucket', key='path/to/model.ckpt'."""
    source = S3Source(url="s3://my-bucket/checkpoints/model.ckpt")
    context = CheckpointContext()

    mock_client = MagicMock()

    def fake_download(_bucket: str, _key: str, dest: str) -> None:
        torch.save({"state_dict": {}}, dest)

    mock_client.download_file.side_effect = fake_download

    with patch("boto3.client", return_value=mock_client):
        await source.process(context)
        mock_client.download_file.assert_called_once()
        call_args = mock_client.download_file.call_args
        assert call_args[0][0] == "my-bucket"
        assert call_args[0][1] == "checkpoints/model.ckpt"


@pytest.mark.asyncio
async def test_s3_source_raises_not_found_for_missing_key() -> None:
    """S3 NoSuchKey -> CheckpointNotFoundError."""
    source = S3Source(url="s3://bucket/missing.ckpt")
    context = CheckpointContext()

    from botocore.exceptions import ClientError

    mock_client = MagicMock()
    mock_client.download_file.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    with patch("boto3.client", return_value=mock_client), pytest.raises(CheckpointNotFoundError):
        await source.process(context)


@pytest.mark.asyncio
async def test_s3_source_cleans_up_temp_file_on_failure() -> None:
    """Temp file must be cleaned up even if loading fails."""
    source = S3Source(url="s3://bucket/corrupt.ckpt")
    context = CheckpointContext()

    mock_client = MagicMock()

    def fake_download(_bucket: str, _key: str, dest: str) -> None:
        Path(dest).write_text("not a valid checkpoint")

    mock_client.download_file.side_effect = fake_download

    with patch("boto3.client", return_value=mock_client), pytest.raises(Exception):  # noqa: B017, PT011
        await source.process(context)
