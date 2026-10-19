# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for RunIdSource and LocalSource with an explicit path."""

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource
from anemoi.training.checkpoint.sources.run import RunIdSource


def _config(root: Path) -> OmegaConf:
    return OmegaConf.create({"system": {"output": {"checkpoints": {"root": str(root)}}}})


def _write_ckpt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"layer.weight": torch.zeros(2, 2)}}, path)
    return path


def test_run_source_extends_checkpoint_source() -> None:
    assert issubclass(RunIdSource, CheckpointSource)


def test_resolve_path_resume(tmp_path: Path) -> None:
    """Resume path is <checkpoints.root>.parent/<run_id>/last.ckpt."""
    root = tmp_path / "job" / "checkpoints"
    path = RunIdSource.resolve_path(_config(root), "run_A", fork=False)
    assert path == tmp_path / "job" / "run_A" / "last.ckpt"


def test_resolve_path_fork_uses_same_formula(tmp_path: Path) -> None:
    """Fork uses the same path formula."""
    root = tmp_path / "job" / "checkpoints"
    path = RunIdSource.resolve_path(_config(root), "base999", fork=True)
    assert path == tmp_path / "job" / "base999" / "last.ckpt"


def test_resolve_path_server2server_overrides(tmp_path: Path) -> None:
    """Server-to-server lineage IDs override run_id."""
    root = tmp_path / "job" / "checkpoints"
    resume = RunIdSource.resolve_path(_config(root), "local_id", fork=False, parent_run_server2server="remote_parent")
    assert resume == tmp_path / "job" / "remote_parent" / "last.ckpt"
    fork = RunIdSource.resolve_path(_config(root), "local_id", fork=True, fork_run_server2server="remote_fork")
    assert fork == tmp_path / "job" / "remote_fork" / "last.ckpt"


def test_resolve_path_missing_root_raises() -> None:
    with pytest.raises(CheckpointConfigError):
        RunIdSource.resolve_path(OmegaConf.create({"system": {"output": {"checkpoints": {}}}}), "run_A", fork=False)


def test_run_source_resume_loads_checkpoint(tmp_path: Path) -> None:
    """Resume finds the run path and loads it."""
    root = tmp_path / "job" / "checkpoints"
    _write_ckpt(tmp_path / "job" / "run_A" / "last.ckpt")

    context = CheckpointContext(config=_config(root))
    result = RunIdSource(run_id="run_A").process(context)

    assert result.checkpoint_path == tmp_path / "job" / "run_A" / "last.ckpt"
    assert result.checkpoint_data is not None
    assert "state_dict" in result.checkpoint_data
    assert result.metadata["lineage_resolution"] == "resume"


def test_run_source_fork_loads_parent_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "job" / "checkpoints"
    _write_ckpt(tmp_path / "job" / "base999" / "last.ckpt")

    context = CheckpointContext(config=_config(root))
    result = RunIdSource(run_id="base999", fork=True).process(context)

    assert result.checkpoint_path == tmp_path / "job" / "base999" / "last.ckpt"
    assert result.metadata["lineage_resolution"] == "fork"


def test_run_source_no_run_id_raises_naming_the_key(tmp_path: Path) -> None:
    """A missing run_id raises an error that names the key.

    The preset has ``run_id: null``, so a missing ID is a likely user error.
    """
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))

    with pytest.raises(CheckpointConfigError, match="requires a run_id"):
        RunIdSource().process(context)


def test_run_source_missing_checkpoint_raises_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing run checkpoint raises RuntimeError on rank 0."""
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
    with pytest.raises(RuntimeError, match="run_missing"):
        RunIdSource(run_id="run_missing").process(context)


def test_run_source_missing_checkpoint_defers_on_nonzero_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero rank defers and does not raise."""
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
    result = RunIdSource(run_id="run_missing").process(context)
    assert result.checkpoint_path is None


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permission bits, so a chmod-000 file stays readable",
)
def test_run_source_unreadable_checkpoint_raises_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable checkpoint raises on rank 0."""
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    ckpt = _write_ckpt(tmp_path / "job" / "run_locked" / "last.ckpt")
    ckpt.chmod(0o000)
    try:
        context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
        with pytest.raises(RuntimeError, match="not readable"):
            RunIdSource(run_id="run_locked").process(context)
    finally:
        ckpt.chmod(0o644)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permission bits, so a chmod-000 file stays readable",
)
def test_run_source_unreadable_checkpoint_defers_on_nonzero_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero rank defers on an unreadable checkpoint."""
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    ckpt = _write_ckpt(tmp_path / "job" / "run_locked" / "last.ckpt")
    ckpt.chmod(0o000)
    try:
        context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
        result = RunIdSource(run_id="run_locked").process(context)
        assert result.checkpoint_path is None
    finally:
        ckpt.chmod(0o644)


def test_local_source_explicit_path_loads(tmp_path: Path) -> None:
    """LocalSource(path=...) loads the file without a context path."""
    ckpt = _write_ckpt(tmp_path / "explicit" / "model.ckpt")
    context = CheckpointContext()
    result = LocalSource(path=ckpt).process(context)
    assert result.checkpoint_path == ckpt
    assert "state_dict" in result.checkpoint_data


def test_run_source_resolve_publishes_path_without_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve`` sets the canonical run checkpoint path and loads nothing."""
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    root = tmp_path / "job" / "checkpoints"
    ckpt = _write_ckpt(tmp_path / "job" / "run_A" / "last.ckpt")

    def _no_load(*_args: object, **_kwargs: object) -> None:
        msg = "resolve must not load the checkpoint"
        raise AssertionError(msg)

    monkeypatch.setattr(torch, "load", _no_load)

    context = CheckpointContext(config=_config(root))
    path = RunIdSource(run_id="run_A").resolve(context)

    assert path == ckpt.resolve()
    assert context.checkpoint_path == path
    assert context.checkpoint_data is None
    assert context.metadata["lineage_resolution"] == "resume"


def test_run_source_resolve_defers_on_nonzero_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing checkpoint on a non-zero rank resolves to None."""
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))

    assert RunIdSource(run_id="run_missing").resolve(context) is None
    assert context.checkpoint_path is None


@pytest.mark.parametrize("unreadable", [False, True], ids=["missing", "unreadable"])
def test_deferring_rank_reports_rank_zero_not_a_corrupted_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unreadable: bool,
) -> None:
    """A rank that defers to rank 0 does not report a corrupted checkpoint."""
    from anemoi.training.checkpoint.exceptions import CheckpointLoadError
    from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

    if unreadable and hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses file permission bits, so a chmod-000 file stays readable")

    root = tmp_path / "job" / "checkpoints"
    context = CheckpointContext(config=_config(root), model=torch.nn.Linear(2, 2))
    monkeypatch.setattr(rank_zero_only, "rank", 1)

    if unreadable:
        run_dir = root.parent / "run-x"
        run_dir.mkdir(parents=True)
        ckpt = run_dir / "last.ckpt"
        ckpt.write_bytes(b"weights")
        ckpt.chmod(0o000)

    try:
        result = RunIdSource(run_id="run-x").process(context)
        assert result.metadata["source_deferred"] is True

        with pytest.raises(CheckpointLoadError, match="rank 0"):
            WeightsOnlyLoader().process(result)
    finally:
        if unreadable:
            ckpt.chmod(0o600)
