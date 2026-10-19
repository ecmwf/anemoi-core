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

import argparse
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(path: Path, data: dict) -> Path:
    """Save checkpoint data to a file and return the path."""
    torch.save(data, path)
    return path


def _no_load(*_args: object, **_kwargs: object) -> None:
    msg = "resolve must not load the checkpoint"
    raise AssertionError(msg)


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


def test_local_source_extends_checkpoint_source() -> None:
    assert issubclass(LocalSource, CheckpointSource)


class TestLocalSourceProcess:
    """Tests for LocalSource.process()."""

    def test_load_valid_ckpt(self, source: LocalSource, ckpt_file: Path) -> None:
        context = CheckpointContext(checkpoint_path=ckpt_file)
        result = source.process(context)

        assert result.checkpoint_data is not None
        assert "state_dict" in result.checkpoint_data
        assert result.checkpoint_data["epoch"] == 5
        assert result.checkpoint_data["state_dict"]["layer.weight"].device == torch.device("cpu")
        assert result.checkpoint_path == ckpt_file
        assert result.metadata["source_type"] == "local"
        assert result.metadata["source_path"] == str(ckpt_file)

    def test_load_lightning_checkpoint_with_non_tensor_metadata(
        self,
        source: LocalSource,
        simple_state_dict: dict,
        tmp_path: Path,
    ) -> None:
        """LocalSource loads Lightning metadata that ``weights_only=True`` rejects."""
        path = _save_checkpoint(
            tmp_path / "lightning.ckpt",
            {
                "state_dict": simple_state_dict,
                "pytorch-lightning_version": "2.2.0",
                "epoch": 10,
                "hyper_parameters": {"config": argparse.Namespace(lr=1e-4)},
            },
        )
        with pytest.raises(pickle.UnpicklingError):
            torch.load(path, weights_only=True, map_location="cpu")

        result = source.process(CheckpointContext(checkpoint_path=path))

        assert result.checkpoint_format == "lightning"
        assert result.checkpoint_data["pytorch-lightning_version"] == "2.2.0"
        assert result.checkpoint_data["epoch"] == 10
        assert result.checkpoint_data["hyper_parameters"]["config"].lr == 1e-4

    def test_file_not_found_raises(self, source: LocalSource, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.ckpt"
        context = CheckpointContext(checkpoint_path=missing)

        with pytest.raises(CheckpointNotFoundError) as exc_info:
            source.process(context)

        assert missing.name in str(exc_info.value)

    @pytest.mark.parametrize(
        ("relative_path", "hint"),
        [
            ("ckpts/missing.ckpt", "Found similar files in"),
            ("no_such_dir/model.ckpt", "Directory does not exist"),
        ],
    )
    def test_file_not_found_gives_a_hint(
        self,
        source: LocalSource,
        tmp_path: Path,
        relative_path: str,
        hint: str,
    ) -> None:
        """A missing file lists other checkpoints in its directory, or reports a missing directory."""
        (tmp_path / "ckpts").mkdir()
        (tmp_path / "ckpts" / "epoch_1.ckpt").write_bytes(b"x")

        with pytest.raises(CheckpointNotFoundError, match=hint):
            source.process(CheckpointContext(checkpoint_path=tmp_path / relative_path))

    @pytest.mark.parametrize(
        "make_bad_file",
        [
            lambda path: path.write_bytes(b""),
            lambda path: path.write_text("this is not a valid checkpoint"),
            lambda path: path.mkdir(),
        ],
        ids=["empty", "garbage", "directory"],
    )
    def test_invalid_file_raises(
        self,
        source: LocalSource,
        tmp_path: Path,
        make_bad_file: Callable[[Path], object],
    ) -> None:
        bad_file = tmp_path / "bad.ckpt"
        make_bad_file(bad_file)

        with pytest.raises(CheckpointLoadError) as exc_info:
            source.process(CheckpointContext(checkpoint_path=bad_file))

        assert str(bad_file) in str(exc_info.value)

    def test_string_path_is_normalised(
        self,
        source: LocalSource,
        ckpt_file: Path,
    ) -> None:
        """Passing a string path should work the same as a Path object."""
        context = CheckpointContext(checkpoint_path=str(ckpt_file))
        result = source.process(context)

        assert result.checkpoint_data is not None
        assert result.metadata["source_type"] == "local"

    def test_relative_path_becomes_absolute(
        self,
        source: LocalSource,
        ckpt_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(ckpt_file.parent)

        result = source.process(CheckpointContext(checkpoint_path=ckpt_file.name))

        assert Path(result.checkpoint_path).is_absolute()
        assert Path(result.checkpoint_path).is_file()

    def test_none_path_raises_config_error(self, source: LocalSource) -> None:
        """Passing None checkpoint_path should raise CheckpointConfigError."""
        context = CheckpointContext(checkpoint_path=None)

        with pytest.raises(CheckpointConfigError, match="checkpoint_path"):
            source.process(context)

    def test_explicit_path_overrides_context_path(self, ckpt_file: Path, tmp_path: Path) -> None:
        """``LocalSource(path=...)`` takes precedence over ``context.checkpoint_path``."""
        other = _save_checkpoint(tmp_path / "other.ckpt", {"state_dict": {}, "epoch": 99})
        source = LocalSource(path=str(ckpt_file))
        context = CheckpointContext(checkpoint_path=other)

        result = source.process(context)

        assert result.metadata["source_path"] == str(ckpt_file)
        assert result.checkpoint_data["epoch"] == 5

    def test_tilde_path_is_expanded(
        self,
        simple_state_dict: dict,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``~`` path is expanded before loading and published as the canonical path."""
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = _save_checkpoint(tmp_path / "home_model.ckpt", {"state_dict": simple_state_dict}).resolve()

        result = LocalSource(path="~/home_model.ckpt").process(CheckpointContext())

        assert result.checkpoint_data is not None
        assert result.checkpoint_path == expected
        assert result.metadata["source_path"] == str(expected)

    def test_symlink_path_is_resolved(self, source: LocalSource, ckpt_file: Path, tmp_path: Path) -> None:
        """A symlink path is resolved to its target in metadata."""
        link = tmp_path / "link.ckpt"
        link.symlink_to(ckpt_file)
        context = CheckpointContext(checkpoint_path=link)

        result = source.process(context)

        assert result.metadata["source_path"] == str(ckpt_file.resolve())

    @pytest.mark.parametrize("exc", [RuntimeError, EOFError, ValueError, pickle.UnpicklingError])
    def test_torch_load_errors_wrap_as_load_error(
        self,
        source: LocalSource,
        ckpt_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        exc: type[Exception],
    ) -> None:
        """Each torch.load error is wrapped as CheckpointLoadError."""
        msg = "simulated torch.load failure"

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise exc(msg)

        monkeypatch.setattr(torch, "load", _raise)
        context = CheckpointContext(checkpoint_path=ckpt_file)

        with pytest.raises(CheckpointLoadError) as exc_info:
            source.process(context)

        assert isinstance(exc_info.value.original_error, exc)


def test_resolve_publishes_canonical_path_without_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve`` sets the expanded, canonical path and loads nothing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    torch.save({"state_dict": {}}, tmp_path / "model.ckpt")
    monkeypatch.setattr(torch, "load", _no_load)

    context = CheckpointContext()
    path = LocalSource(path="~/model.ckpt").resolve(context)

    assert path == (tmp_path / "model.ckpt").resolve()
    assert context.checkpoint_path == path
    assert context.checkpoint_data is None


class TestLocalSourceSupports:
    """Tests for LocalSource.supports()."""

    def test_path_object_returns_true(self, tmp_path: Path) -> None:
        assert LocalSource.supports(tmp_path / "model.ckpt") is True

    def test_existing_file_returns_true(self, ckpt_file: Path) -> None:
        assert LocalSource.supports(str(ckpt_file)) is True

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("/models/epoch_50.ckpt", True),
            ("relative/path/model.pt", True),
            ("s3://bucket/model.ckpt", False),
            ("http://example.com/model.ckpt", False),
            ("https://example.com/model.ckpt", False),
            ("gs://bucket/model.ckpt", False),
        ],
    )
    def test_string_without_scheme_is_local(self, source: str, expected: bool) -> None:
        assert LocalSource.supports(source) is expected
