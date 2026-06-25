"""Tests for RunSource (resume / fork by run id) and LocalSource explicit path."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.local import LocalSource
from anemoi.training.checkpoint.sources.run import RunSource


def _config(root: Path) -> OmegaConf:
    return OmegaConf.create({"system": {"output": {"checkpoints": {"root": str(root)}}}})


def _write_ckpt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"layer.weight": torch.zeros(2, 2)}}, path)
    return path


def test_run_source_extends_checkpoint_source() -> None:
    assert issubclass(RunSource, CheckpointSource)


def test_resolve_path_resume(tmp_path: Path) -> None:
    """Resume path is <checkpoints.root>.parent/<run_id>/last.ckpt."""
    root = tmp_path / "job" / "checkpoints"
    path = RunSource.resolve_path(_config(root), "run_A", fork=False)
    assert path == tmp_path / "job" / "run_A" / "last.ckpt"


def test_resolve_path_fork_uses_same_formula(tmp_path: Path) -> None:
    """Fork resolves the parent run's checkpoint by the same formula."""
    root = tmp_path / "job" / "checkpoints"
    path = RunSource.resolve_path(_config(root), "base999", fork=True)
    assert path == tmp_path / "job" / "base999" / "last.ckpt"


def test_resolve_path_server2server_overrides(tmp_path: Path) -> None:
    """Server-to-server lineage ids take precedence over run_id."""
    root = tmp_path / "job" / "checkpoints"
    resume = RunSource.resolve_path(_config(root), "local_id", fork=False, parent_run_server2server="remote_parent")
    assert resume == tmp_path / "job" / "remote_parent" / "last.ckpt"
    fork = RunSource.resolve_path(_config(root), "local_id", fork=True, fork_run_server2server="remote_fork")
    assert fork == tmp_path / "job" / "remote_fork" / "last.ckpt"


def test_resolve_path_missing_root_raises() -> None:
    with pytest.raises(CheckpointConfigError):
        RunSource.resolve_path(OmegaConf.create({"system": {"output": {"checkpoints": {}}}}), "run_A", fork=False)


@pytest.mark.asyncio
async def test_run_source_resume_loads_checkpoint(tmp_path: Path) -> None:
    """Resume resolves the run path and loads it via LocalSource semantics."""
    root = tmp_path / "job" / "checkpoints"
    _write_ckpt(tmp_path / "job" / "run_A" / "last.ckpt")

    context = CheckpointContext(config=_config(root))
    result = await RunSource(run_id="run_A").process(context)

    assert result.checkpoint_path == tmp_path / "job" / "run_A" / "last.ckpt"
    assert result.checkpoint_data is not None
    assert "state_dict" in result.checkpoint_data
    assert result.metadata["lineage_resolution"] == "resume"


@pytest.mark.asyncio
async def test_run_source_fork_loads_parent_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "job" / "checkpoints"
    _write_ckpt(tmp_path / "job" / "base999" / "last.ckpt")

    context = CheckpointContext(config=_config(root))
    result = await RunSource(run_id="base999", fork=True).process(context)

    assert result.checkpoint_path == tmp_path / "job" / "base999" / "last.ckpt"
    assert result.metadata["lineage_resolution"] == "fork"


@pytest.mark.asyncio
async def test_run_source_no_run_id_is_passthrough(tmp_path: Path) -> None:
    """No run_id -> no-op pass-through (nothing resolved)."""
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
    result = await RunSource().process(context)
    assert result.checkpoint_path is None
    assert result.checkpoint_data is None


@pytest.mark.asyncio
async def test_run_source_missing_checkpoint_raises_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing run checkpoint raises RuntimeError on rank 0 (default, no rank env)."""
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(var, raising=False)
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
    with pytest.raises(RuntimeError, match="run_missing"):
        await RunSource(run_id="run_missing").process(context)


@pytest.mark.asyncio
async def test_run_source_missing_checkpoint_defers_on_nonzero_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-rank-0 defers (warns, pass-through) rather than raising."""
    monkeypatch.setenv("RANK", "1")
    context = CheckpointContext(config=_config(tmp_path / "job" / "checkpoints"))
    result = await RunSource(run_id="run_missing").process(context)
    assert result.checkpoint_path is None


@pytest.mark.asyncio
async def test_local_source_explicit_path_loads(tmp_path: Path) -> None:
    """LocalSource(path=...) loads the explicit file without a context path set."""
    ckpt = _write_ckpt(tmp_path / "explicit" / "model.ckpt")
    context = CheckpointContext()
    result = await LocalSource(path=ckpt).process(context)
    assert result.checkpoint_path == ckpt
    assert "state_dict" in result.checkpoint_data
