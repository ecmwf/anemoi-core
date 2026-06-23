"""Tests for the run-lineage checkpoint path resolver (LineageResolver).

Verifies parity with the legacy ``last_checkpoint`` resolution
(``train/train.py`` @ ``origin/main 6dcc870e0``): precedence
(warm-start > fork > lineage), the ``<root.parent>/<id>/last.ckpt`` path shape,
the formatted missing-path error (not the legacy tuple defect), and the no-op
pass-through when no resume/fork/warm-start key is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.sources import LineageResolver

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

# Mirrors lightning_fabric._get_rank() — the env set the legacy rank_zero_only.rank read.
RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


@pytest.fixture
def rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a deterministic rank-0 environment (no launcher rank vars set)."""
    for var in RANK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_config(
    *,
    root: Path,
    run_id: str | None = None,
    fork_run_id: str | None = None,
    warm_start: str | None = None,
) -> DictConfig:
    return OmegaConf.create(
        {
            "training": {"run_id": run_id, "fork_run_id": fork_run_id},
            "system": {
                "input": {"warm_start": warm_start},
                "output": {"checkpoints": {"root": str(root)}},
            },
        },
    )


def _touch_ckpt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ckpt")
    return path


def test_lineage_resolver_is_pipeline_stage() -> None:
    assert issubclass(LineageResolver, PipelineStage)


async def test_resolves_lineage_run_to_path(tmp_path: Path) -> None:
    """run_id (no fork, no warm-start) => <root.parent>/<run_id>/last.ckpt."""
    root = tmp_path / "run_A"  # post-_update_paths root (base/lineage_run)
    ckpt = _touch_ckpt(tmp_path / "run_A" / "last.ckpt")
    context = CheckpointContext(config=_make_config(root=root, run_id="run_A"))

    result = await LineageResolver().process(context)

    assert result.checkpoint_path == ckpt
    assert result.metadata["lineage_resolution"] == "lineage"
    assert result.metadata["resolved_checkpoint_path"] == str(ckpt)


async def test_precedence_warm_start_over_fork_and_lineage(tmp_path: Path) -> None:
    """Explicit warm-start path wins over both fork id and lineage run."""
    warm = _touch_ckpt(tmp_path / "warm" / "explicit.ckpt")
    # A fork dir and a lineage dir also exist; both must be ignored.
    _touch_ckpt(tmp_path / "fork_B" / "last.ckpt")
    _touch_ckpt(tmp_path / "run_A" / "last.ckpt")
    context = CheckpointContext(
        config=_make_config(root=tmp_path / "run_A", run_id="run_A", fork_run_id="fork_B", warm_start=str(warm)),
    )

    result = await LineageResolver().process(context)

    assert result.checkpoint_path == warm
    assert result.metadata["lineage_resolution"] == "warm_start"


async def test_precedence_fork_over_lineage(tmp_path: Path) -> None:
    """With no warm-start, the fork id wins over the lineage run id."""
    fork_ckpt = _touch_ckpt(tmp_path / "fork_B" / "last.ckpt")
    _touch_ckpt(tmp_path / "run_A" / "last.ckpt")  # lineage dir also present
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A", fork_run_id="fork_B"))

    result = await LineageResolver().process(context)

    assert result.checkpoint_path == fork_ckpt


async def test_server2server_fork_overrides_config_fork(tmp_path: Path) -> None:
    """fork_run_server2server (explicit input) takes precedence over config fork_run_id."""
    s2s_ckpt = _touch_ckpt(tmp_path / "fork_S2S" / "last.ckpt")
    _touch_ckpt(tmp_path / "fork_B" / "last.ckpt")  # config fork dir present but lower precedence
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A", fork_run_id="fork_B"))

    result = await LineageResolver(fork_run_server2server="fork_S2S").process(context)

    assert result.checkpoint_path == s2s_ckpt


async def test_server2server_parent_overrides_lineage_run(tmp_path: Path) -> None:
    """parent_run_server2server (explicit input) takes precedence over run_id as lineage."""
    s2s_ckpt = _touch_ckpt(tmp_path / "parent_S2S" / "last.ckpt")
    _touch_ckpt(tmp_path / "run_A" / "last.ckpt")  # config run dir present but lower precedence
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A"))

    result = await LineageResolver(parent_run_server2server="parent_S2S").process(context)

    assert result.checkpoint_path == s2s_ckpt


@pytest.mark.usefixtures("rank_zero")
async def test_missing_checkpoint_raises_formatted_runtimeerror(tmp_path: Path) -> None:
    """Missing resolved checkpoint => RuntimeError whose str() contains the path.

    The legacy tuple defect would render the message as
    ``"('Could not find last checkpoint: %s', PosixPath(...))"``; the resolver
    must produce a plain formatted string instead.
    """
    expected = tmp_path / "run_A" / "last.ckpt"  # never created
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A"))

    with pytest.raises(RuntimeError) as excinfo:
        await LineageResolver().process(context)

    message = str(excinfo.value)
    assert str(expected) in message
    assert "%s" not in message
    assert not message.startswith("(")


async def test_fork_beats_parent_server2server_lineage(tmp_path: Path) -> None:
    """With a fork id set, the fork wins even when the lineage run is a server2server value.

    Locks the single-segment ``fork_id or lineage_run`` precedence: a regression
    that let the lineage component win when a fork is present must fail here.
    """
    fork_ckpt = _touch_ckpt(tmp_path / "fork_B" / "last.ckpt")
    _touch_ckpt(tmp_path / "parent_S2S" / "last.ckpt")  # lineage (s2s) dir also present
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A", fork_run_id="fork_B"))

    result = await LineageResolver(parent_run_server2server="parent_S2S").process(context)

    assert result.checkpoint_path == fork_ckpt


async def test_missing_checkpoints_root_raises_config_error() -> None:
    """A lineage path is required but system.output.checkpoints.root is absent => CheckpointConfigError."""
    from anemoi.training.checkpoint.exceptions import CheckpointConfigError

    config = OmegaConf.create(
        {"training": {"run_id": "run_A", "fork_run_id": None}, "system": {"input": {"warm_start": None}}},
    )
    context = CheckpointContext(config=config)

    with pytest.raises(CheckpointConfigError):
        await LineageResolver().process(context)


async def test_warm_start_set_but_missing_raises_filenotfound(tmp_path: Path) -> None:
    """A configured-but-missing warm-start path raises FileNotFoundError (legacy parity)."""
    missing = tmp_path / "warm" / "nope.ckpt"
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", warm_start=str(missing)))

    with pytest.raises(FileNotFoundError):
        await LineageResolver().process(context)


async def test_no_op_when_no_key_set(tmp_path: Path) -> None:
    """No run_id/fork_run_id/warm_start => pass-through; checkpoint_path untouched."""
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A"))

    result = await LineageResolver().process(context)

    assert result is context
    assert result.checkpoint_path is None
    assert "lineage_resolution" not in result.metadata


async def test_no_op_when_config_absent() -> None:
    """No config on context => pass-through (nothing to resolve)."""
    context = CheckpointContext()

    result = await LineageResolver().process(context)

    assert result.checkpoint_path is None


async def test_non_rank_zero_defers_on_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero global rank does not raise on a missing checkpoint; it defers."""
    monkeypatch.setenv("RANK", "1")
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A"))

    result = await LineageResolver().process(context)  # must NOT raise

    assert result.checkpoint_path is None
    # The deferral path resolves nothing, so it must not record resolution metadata.
    assert "lineage_resolution" not in result.metadata
    assert "resolved_checkpoint_path" not in result.metadata


async def test_non_integer_rank_treated_as_rank_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer rank value is treated conservatively as rank 0 and raises on a missing checkpoint."""
    monkeypatch.setenv("RANK", "not-a-number")
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A"))

    with pytest.raises(RuntimeError):
        await LineageResolver().process(context)


async def test_local_rank_nonzero_defers_on_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker with only LOCAL_RANK set (non-zero) defers, matching lightning_fabric._get_rank().

    Regression guard: LOCAL_RANK must be honoured (it is in the legacy rank_zero_only.rank
    fallback). If it were dropped, such a worker would mis-detect as rank 0 and raise on every node.
    """
    for var in ("RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")
    context = CheckpointContext(config=_make_config(root=tmp_path / "run_A", run_id="run_A"))

    result = await LineageResolver().process(context)  # must NOT raise

    assert result.checkpoint_path is None
