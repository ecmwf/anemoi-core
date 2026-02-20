# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for LocalSource checkpoint source."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path
    from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _save_checkpoint(path: Path, data: dict) -> Path:
    """Save checkpoint data to a file and return the path."""
    torch.save(data, path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source() -> LocalSource:
    return LocalSource()


@pytest.fixture
def simple_state_dict() -> dict:
    return {"layer.weight": torch.randn(4, 3), "layer.bias": torch.randn(4)}


@pytest.fixture
def ckpt_file(tmp_path: Path, simple_state_dict: dict) -> Path:
    """A minimal .ckpt file with a raw state dict under 'state_dict' key."""
    return _save_checkpoint(
        tmp_path / "model.ckpt",
        {"state_dict": simple_state_dict, "epoch": 5},
    )


@pytest.fixture
def lightning_ckpt_file(tmp_path: Path, simple_state_dict: dict) -> Path:
    """A Lightning-format .ckpt file."""
    return _save_checkpoint(
        tmp_path / "lightning.ckpt",
        {
            "state_dict": simple_state_dict,
            "pytorch-lightning_version": "2.2.0",
            "epoch": 10,
            "global_step": 500,
            "hyper_parameters": {"lr": 1e-4},
        },
    )


class TestLocalSourceProcess:
    """Tests for LocalSource.process()."""

    def test_load_valid_ckpt(self, source: LocalSource, ckpt_file: Path) -> None:
        context = CheckpointContext(checkpoint_path=ckpt_file)
        result = _run(source.process(context))

        assert result.checkpoint_data is not None
        assert "state_dict" in result.checkpoint_data
        assert result.checkpoint_data["epoch"] == 5
        assert result.metadata["source_type"] == "local"
        assert result.metadata["source_path"] == str(ckpt_file)

    def test_load_lightning_checkpoint(
        self,
        source: LocalSource,
        lightning_ckpt_file: Path,
    ) -> None:
        context = CheckpointContext(checkpoint_path=lightning_ckpt_file)
        result = _run(source.process(context))

        assert result.checkpoint_data is not None
        assert result.checkpoint_format == "lightning"
        assert result.checkpoint_data["pytorch-lightning_version"] == "2.2.0"
        assert result.checkpoint_data["epoch"] == 10
        assert result.metadata["source_type"] == "local"

    def test_file_not_found_raises(self, source: LocalSource, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.ckpt"
        context = CheckpointContext(checkpoint_path=missing)

        with pytest.raises(CheckpointNotFoundError) as exc_info:
            _run(source.process(context))

        assert missing.name in str(exc_info.value)

    def test_invalid_file_raises(self, source: LocalSource, tmp_path: Path) -> None:
        bad_file = tmp_path / "garbage.ckpt"
        bad_file.write_text("this is not a valid checkpoint")

        context = CheckpointContext(checkpoint_path=bad_file)

        with pytest.raises(CheckpointLoadError) as exc_info:
            _run(source.process(context))

        assert str(bad_file) in str(exc_info.value)

    def test_string_path_is_normalised(
        self,
        source: LocalSource,
        ckpt_file: Path,
    ) -> None:
        """Passing a string path should work the same as a Path object."""
        context = CheckpointContext(checkpoint_path=str(ckpt_file))
        result = _run(source.process(context))

        assert result.checkpoint_data is not None
        assert result.metadata["source_type"] == "local"


class TestLocalSourceSupports:
    """Tests for LocalSource.supports()."""

    def test_path_object_returns_true(self, tmp_path: Path) -> None:
        assert LocalSource.supports(tmp_path / "model.ckpt") is True

    def test_existing_file_returns_true(self, ckpt_file: Path) -> None:
        assert LocalSource.supports(str(ckpt_file)) is True

    def test_local_string_without_scheme_returns_true(self) -> None:
        assert LocalSource.supports("/models/epoch_50.ckpt") is True
        assert LocalSource.supports("relative/path/model.pt") is True

    def test_s3_url_returns_false(self) -> None:
        assert LocalSource.supports("s3://bucket/model.ckpt") is False

    def test_http_url_returns_false(self) -> None:
        assert LocalSource.supports("http://example.com/model.ckpt") is False
        assert LocalSource.supports("https://example.com/model.ckpt") is False

    def test_gcs_url_returns_false(self) -> None:
        assert LocalSource.supports("gs://bucket/model.ckpt") is False
