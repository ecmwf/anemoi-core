"""Gate sources-g4: S3Source.

CANONICAL GATE TEST — DO NOT MODIFY.

S3Source downloads via ``anemoi.utils.remote.s3.download_file`` (obstore-backed).
boto3 is not a dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.exceptions import CheckpointSourceError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.s3 import S3Source


def _fake_anemoi_utils_s3(download_impl: object) -> ModuleType:
    """Stub ``anemoi.utils.remote.s3`` module exposing a custom ``download_file``."""
    module = ModuleType("anemoi.utils.remote.s3")
    module.download_file = download_impl  # type: ignore[attr-defined]
    return module


def test_s3_source_extends_checkpoint_source() -> None:
    assert issubclass(S3Source, CheckpointSource)


def test_s3_source_does_not_import_anemoi_utils_at_module_level() -> None:
    """anemoi.utils.remote.s3 must be imported lazily inside the method."""
    import importlib

    saved = sys.modules.pop("anemoi.utils.remote.s3", None)
    try:
        importlib.reload(importlib.import_module("anemoi.training.checkpoint.sources.s3"))
    finally:
        if saved is not None:
            sys.modules["anemoi.utils.remote.s3"] = saved


@pytest.mark.asyncio
async def test_s3_source_calls_anemoi_utils_download_file() -> None:
    """S3Source forwards the URL to anemoi.utils.remote.s3.download_file."""
    source = S3Source(url="s3://my-bucket/checkpoints/model.ckpt")
    context = CheckpointContext()

    captured: dict[str, object] = {}

    def fake_download(url: str, target: str, *_args: object, **_kwargs: object) -> None:
        captured["url"] = url
        captured["target"] = target
        torch.save({"state_dict": {}}, target)

    fake_module = _fake_anemoi_utils_s3(fake_download)
    with patch.dict(sys.modules, {"anemoi.utils.remote.s3": fake_module}):
        result = await source.process(context)

    assert captured["url"] == "s3://my-bucket/checkpoints/model.ckpt"
    assert result.checkpoint_data == {"state_dict": {}}
    assert result.metadata["source_type"] == "s3"
    assert result.metadata["s3_bucket"] == "my-bucket"
    assert result.metadata["s3_key"] == "checkpoints/model.ckpt"


@pytest.mark.asyncio
async def test_s3_source_raises_not_found_for_missing_object() -> None:
    """FileNotFoundError from anemoi-utils -> CheckpointNotFoundError."""
    source = S3Source(url="s3://bucket/missing.ckpt")
    context = CheckpointContext()

    def fake_download(*_args: object, **_kwargs: object) -> None:
        msg = "object does not exist"
        raise FileNotFoundError(msg)

    fake_module = _fake_anemoi_utils_s3(fake_download)
    with patch.dict(sys.modules, {"anemoi.utils.remote.s3": fake_module}), pytest.raises(CheckpointNotFoundError):
        await source.process(context)


@pytest.mark.asyncio
async def test_s3_source_raises_source_error_for_transport_failure() -> None:
    """OSError/RuntimeError from anemoi-utils -> CheckpointSourceError."""
    source = S3Source(url="s3://bucket/key.ckpt")
    context = CheckpointContext()

    def fake_download(*_args: object, **_kwargs: object) -> None:
        msg = "connection refused"
        raise OSError(msg)

    fake_module = _fake_anemoi_utils_s3(fake_download)
    with patch.dict(sys.modules, {"anemoi.utils.remote.s3": fake_module}), pytest.raises(CheckpointSourceError):
        await source.process(context)


@pytest.mark.asyncio
async def test_s3_source_raises_when_obstore_missing() -> None:
    """ImportError raised by anemoi-utils download_file (obstore missing) -> CheckpointSourceError."""
    source = S3Source(url="s3://bucket/key.ckpt")
    context = CheckpointContext()

    def fake_download(*_args: object, **_kwargs: object) -> None:
        msg = "No module named 'obstore'"
        raise ImportError(msg)

    fake_module = _fake_anemoi_utils_s3(fake_download)
    patched = patch.dict(sys.modules, {"anemoi.utils.remote.s3": fake_module})
    with patched, pytest.raises(CheckpointSourceError) as excinfo:
        await source.process(context)
    assert "obstore" in str(excinfo.value)
    assert "anemoi-utils[s3]" in str(excinfo.value)


@pytest.mark.asyncio
async def test_s3_source_cleans_up_temp_file_on_failure() -> None:
    """Temp file must be cleaned up even if loading fails."""
    source = S3Source(url="s3://bucket/corrupt.ckpt")
    context = CheckpointContext()

    def fake_download(_url: str, target: str, *_args: object, **_kwargs: object) -> None:
        Path(target).write_text("not a valid checkpoint")

    fake_module = _fake_anemoi_utils_s3(fake_download)
    with patch.dict(sys.modules, {"anemoi.utils.remote.s3": fake_module}), pytest.raises(CheckpointLoadError):
        await source.process(context)
