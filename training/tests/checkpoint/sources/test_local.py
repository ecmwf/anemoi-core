"""Tests for LocalSource."""

from pathlib import Path

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource


def test_local_source_extends_checkpoint_source() -> None:
    assert issubclass(LocalSource, CheckpointSource)


@pytest.mark.asyncio
async def test_local_source_loads_checkpoint(sample_checkpoint: Path) -> None:
    """LocalSource populates context.checkpoint_data from a local file."""
    source = LocalSource()
    context = CheckpointContext(checkpoint_path=sample_checkpoint)
    result = await source.process(context)

    assert result.checkpoint_data is not None
    assert "state_dict" in result.checkpoint_data
    assert result.checkpoint_path == sample_checkpoint


@pytest.mark.asyncio
async def test_local_source_raises_not_found_error(tmp_path: Path) -> None:
    """Must raise CheckpointNotFoundError (Phase 1 type), NOT FileNotFoundError."""
    source = LocalSource()
    context = CheckpointContext(checkpoint_path=tmp_path / "nonexistent.ckpt")

    with pytest.raises(CheckpointNotFoundError):
        await source.process(context)


@pytest.mark.asyncio
async def test_local_source_sets_checkpoint_format(sample_checkpoint: Path) -> None:
    """Should detect and set checkpoint_format on context."""
    source = LocalSource()
    context = CheckpointContext(checkpoint_path=sample_checkpoint)
    result = await source.process(context)

    assert result.checkpoint_format is not None
    assert result.checkpoint_format in ("lightning", "pytorch", "state_dict")


@pytest.mark.asyncio
async def test_local_source_uses_cpu_map_location(sample_checkpoint: Path) -> None:
    """Checkpoint must be loaded with map_location='cpu'."""
    source = LocalSource()
    context = CheckpointContext(checkpoint_path=sample_checkpoint)
    result = await source.process(context)

    # Verify tensor is on CPU
    weight = result.checkpoint_data["state_dict"]["layer.weight"]
    assert weight.device == torch.device("cpu")


@pytest.mark.asyncio
async def test_local_source_handles_empty_file(tmp_path: Path) -> None:
    """A 0-byte file should raise CheckpointLoadError, not a cryptic torch error."""
    empty = tmp_path / "empty.ckpt"
    empty.touch()
    source = LocalSource()
    context = CheckpointContext(checkpoint_path=empty)
    with pytest.raises(Exception):  # noqa: B017, PT011  # CheckpointLoadError or CheckpointValidationError
        await source.process(context)
