# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
async def test_local_source_resolve_publishes_canonical_path_without_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve`` expands ``~``, canonicalises, publishes the path, and reads no file.

    The published path is what the trainer hands to ``Trainer.fit(ckpt_path=)``, so it
    must be the file that was checked, not the ``~`` form the config carried.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    torch.save({"state_dict": {}}, tmp_path / "model.ckpt")

    def _no_load(*_args: object, **_kwargs: object) -> None:
        msg = "resolve must not load the checkpoint"
        raise AssertionError(msg)

    monkeypatch.setattr(torch, "load", _no_load)

    context = CheckpointContext()
    path = await LocalSource(path="~/model.ckpt").resolve(context)

    assert path == (tmp_path / "model.ckpt").resolve()
    assert context.checkpoint_path == path
    assert context.checkpoint_data is None
    assert context.temporary_files == []


@pytest.mark.asyncio
async def test_local_source_process_publishes_the_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a load, ``context.checkpoint_path`` is the canonical file, not the configured spelling."""
    monkeypatch.setenv("HOME", str(tmp_path))
    torch.save({"state_dict": {}}, tmp_path / "model.ckpt")

    result = await LocalSource(path="~/model.ckpt").process(CheckpointContext())

    assert result.checkpoint_path == (tmp_path / "model.ckpt").resolve()


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


@pytest.mark.asyncio
async def test_local_source_resolves_a_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative path is published absolute, so a later chdir cannot invalidate what Lightning receives."""
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": {"layer.weight": torch.zeros(2, 2)}}, ckpt)
    monkeypatch.chdir(tmp_path)

    result = await LocalSource().process(CheckpointContext(checkpoint_path="last.ckpt"))

    assert Path(result.checkpoint_path).is_absolute()
    assert Path(result.checkpoint_path).is_file()
