# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity coverage for run-lineage, MLflow, determinism and guard behaviour.

These tests drive the trainer run-identity wiring (``_update_paths``,
``_check_dry_run``, ``run_id`` fall-through, ``_skip_lightning_restore``), the
MLflow lineage tagging on resume/fork, the acquisition-layer rank-0 gate, the
checkpoint builder error surface, and the seed-scaling rule. Each test exercises
the real classes/functions on CPU with no network access; remote downloads are
replaced with a synthetic writer, but the code under test always runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.sources.http import HTTPSource
from anemoi.training.checkpoint.sources.run import RunSource
from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.seeding import get_base_seed

_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


# ---------------------------------------------------------------------------
# _update_paths: lineage-append to the output paths (train.py:758-781)
# ---------------------------------------------------------------------------


def _paths_namespace(
    *,
    run_id: str | None,
    parent_run_server2server: str | None,
    fork_run_id: str | None,
    checkpoints_root: str,
    plots: str,
) -> SimpleNamespace:
    """Minimal ``self`` for ``AnemoiTrainer._update_paths`` (reads run_id + config only)."""
    training: dict = {}
    if fork_run_id is not None:
        training["fork_run_id"] = fork_run_id
    config = OmegaConf.create(
        {
            "system": {"output": {"checkpoints": {"root": checkpoints_root}, "plots": plots}},
            "training": training,
        },
    )
    return SimpleNamespace(
        run_id=run_id,
        parent_run_server2server=parent_run_server2server,
        config=config,
    )


def test_update_paths_appends_run_id_to_checkpoints_and_plots() -> None:
    ns = _paths_namespace(
        run_id="run-abc",
        parent_run_server2server=None,
        fork_run_id=None,
        checkpoints_root="/base/ckpts",
        plots="/base/plots",
    )

    AnemoiTrainer._update_paths(ns)

    assert ns.config.system.output.checkpoints.root == Path("/base/ckpts", "run-abc")
    assert ns.config.system.output.plots == Path("/base/plots", "run-abc")


def test_update_paths_uses_fork_run_id_when_run_id_absent() -> None:
    ns = _paths_namespace(
        run_id=None,
        parent_run_server2server=None,
        fork_run_id="fork-999",
        checkpoints_root="/base/ckpts",
        plots="/base/plots",
    )

    AnemoiTrainer._update_paths(ns)

    # The fork branch appends only to checkpoints.root; plots is left untouched.
    assert ns.config.system.output.checkpoints.root == Path("/base/ckpts", "fork-999")
    assert ns.config.system.output.plots == "/base/plots"


def test_update_paths_server2server_takes_precedence_over_run_id() -> None:
    ns = _paths_namespace(
        run_id="local-run",
        parent_run_server2server="remote-parent",
        fork_run_id=None,
        checkpoints_root="/base/ckpts",
        plots="/base/plots",
    )

    AnemoiTrainer._update_paths(ns)

    assert ns.config.system.output.checkpoints.root == Path("/base/ckpts", "remote-parent")
    assert ns.config.system.output.plots == Path("/base/plots", "remote-parent")


# ---------------------------------------------------------------------------
# _check_dry_run: rank-zero gate + parent-dry-run suppression (train.py:786-800)
# ---------------------------------------------------------------------------


def _dry_run_namespace(
    *,
    fork_run_id: str | None,
    parent_dry_run: bool,
    start_from_checkpoint: bool,
) -> SimpleNamespace:
    """Minimal ``self`` for ``AnemoiTrainer._check_dry_run`` with an mlflow logger."""
    training: dict = {}
    if fork_run_id is not None:
        training["fork_run_id"] = fork_run_id
    config = OmegaConf.create(
        {
            # A path that does not exist on disk -> ``.is_dir()`` is False.
            "system": {"output": {"checkpoints": {"root": "/nonexistent/checkpoint/dir"}}},
            "training": training,
        },
    )
    return SimpleNamespace(
        start_from_checkpoint=start_from_checkpoint,
        logger=SimpleNamespace(logger_name="mlflow"),
        mlflow_logger=SimpleNamespace(_parent_dry_run=parent_dry_run),
        config=config,
    )


def test_check_dry_run_is_skipped_on_non_zero_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    ns = _dry_run_namespace(fork_run_id=None, parent_dry_run=True, start_from_checkpoint=True)

    AnemoiTrainer._check_dry_run(ns)

    # The @rank_zero_only decorator skips the whole body on rank > 0.
    assert not hasattr(ns, "dry_run")
    assert ns.start_from_checkpoint is True


def test_check_dry_run_suppresses_start_from_checkpoint_for_dry_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    ns = _dry_run_namespace(fork_run_id=None, parent_dry_run=True, start_from_checkpoint=True)

    AnemoiTrainer._check_dry_run(ns)

    assert ns.dry_run is True
    assert ns.start_from_checkpoint is False


def test_check_dry_run_preserves_start_from_checkpoint_for_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    ns = _dry_run_namespace(fork_run_id="fork-1", parent_dry_run=True, start_from_checkpoint=True)

    AnemoiTrainer._check_dry_run(ns)

    assert ns.dry_run is True
    # A fork keeps start_from_checkpoint even when the parent is dry.
    assert ns.start_from_checkpoint is True


# ---------------------------------------------------------------------------
# _skip_lightning_restore: loader-class resolution (train.py:825-844)
# ---------------------------------------------------------------------------


def _skip_restore_namespace(loading: dict | None) -> SimpleNamespace:
    checkpoint = {"loading": loading} if loading is not None else {}
    return SimpleNamespace(config=OmegaConf.create({"training": {"checkpoint": checkpoint}}))


@pytest.mark.parametrize(
    ("loading", "expected"),
    [
        ({"_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"}, True),
        ({"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"}, False),
        ({"_target_": "nonexistent.LoaderClass"}, False),
    ],
)
def test_skip_lightning_restore_matches_loading_strategy(loading: dict, expected: bool) -> None:
    ns = _skip_restore_namespace(loading)

    # An unresolvable _target_ must be caught (ImportError/ValueError) and yield
    # False rather than propagating, so the resume path is not suppressed here.
    assert AnemoiTrainer._skip_lightning_restore(ns) is expected


# ---------------------------------------------------------------------------
# MLflow lineage tags on resume / fork (logger.py:_get_mlflow_run_params)
# ---------------------------------------------------------------------------


def _create_parent_run(
    save_dir: str,
    experiment_name: str,
    *,
    tags: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> str:
    """Create a finished MLflow run in an offline file store and return its id."""
    mlflow.set_tracking_uri(f"file://{save_dir}")
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags=tags or {}):
        for key, value in (params or {}).items():
            mlflow.log_param(key, value)
        return mlflow.active_run().info.run_id


def _offline_logger(save_dir: str, experiment_name: str, **kwargs: object) -> AnemoiMLflowLogger:
    return AnemoiMLflowLogger(
        experiment_name=experiment_name,
        offline=True,
        tracking_uri=None,
        authentication=False,
        save_dir=save_dir,
        **kwargs,
    )


def test_mlflow_resume_sets_resumed_run_tag(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "mlruns")
    parent = _create_parent_run(save_dir, "exp")

    logger = _offline_logger(save_dir, "exp", run_id=parent, fork_run_id=None)
    child_id = logger.run_id  # forces the child run to be created

    client = mlflow.MlflowClient(f"file://{save_dir}")
    run = client.get_run(child_id)
    assert run.data.tags.get("resumedRun") == "True"


def test_mlflow_resume_child_links_parent_run_id(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "mlruns")
    parent = _create_parent_run(save_dir, "exp")

    logger = _offline_logger(save_dir, "exp", run_id=parent, fork_run_id=None)
    child_id = logger.run_id

    client = mlflow.MlflowClient(f"file://{save_dir}")
    run = client.get_run(child_id)
    assert run.data.tags.get("mlflow.parentRunId") == parent
    assert child_id != parent


def test_mlflow_fork_sets_forked_run_tags(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "mlruns")
    parent = _create_parent_run(save_dir, "exp")

    logger = _offline_logger(save_dir, "exp", run_id=None, fork_run_id=parent)
    fork_id = logger.run_id

    client = mlflow.MlflowClient(f"file://{save_dir}")
    run = client.get_run(fork_id)
    assert run.data.tags.get("forkedRun") == "True"
    assert run.data.tags.get("forkedRunId") == parent


def test_mlflow_in_place_resume_when_child_disabled(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "mlruns")
    parent = _create_parent_run(save_dir, "exp")

    logger = _offline_logger(save_dir, "exp", run_id=parent, fork_run_id=None, on_resume_create_child=False)

    # The same run continues in place: no child, no parentRunId link, status RUNNING.
    assert logger.run_id == parent
    assert "mlflow.parentRunId" not in logger.tags
    client = mlflow.MlflowClient(f"file://{save_dir}")
    assert client.get_run(parent).info.status == "RUNNING"


def test_mlflow_resume_extracts_server2server_parent_lineage(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "mlruns")
    parent = _create_parent_run(
        save_dir,
        "exp",
        tags={"server2server": "True"},
        params={"metadata.offline_run_id": "origin-parent"},
    )

    logger = _offline_logger(save_dir, "exp", run_id=parent, fork_run_id=None)

    assert logger._parent_run_server2server == "origin-parent"


def test_mlflow_fork_extracts_server2server_fork_lineage(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "mlruns")
    parent = _create_parent_run(
        save_dir,
        "exp",
        tags={"server2server": "True"},
        params={"metadata.offline_run_id": "origin-fork"},
    )

    logger = _offline_logger(save_dir, "exp", run_id=None, fork_run_id=parent)

    assert logger._fork_run_server2server == "origin-fork"


# ---------------------------------------------------------------------------
# Builder / pipeline error surface (builder.py + pipeline.py)
# ---------------------------------------------------------------------------


def test_builder_malformed_source_target_raises_config_error() -> None:
    cfg = OmegaConf.create({"training": {"checkpoint": {"source": {"_target_": "nonexistent.Source"}}}})
    model = nn.Linear(2, 2)

    def _build_and_run() -> None:
        pipeline = build_checkpoint_pipeline(cfg)
        asyncio.run(pipeline.execute(CheckpointContext(model=model, config=cfg)))

    with pytest.raises(CheckpointConfigError) as excinfo:
        _build_and_run()

    assert "stage" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Acquisition-layer rank-0 gate (run.py:_is_rank_zero)
# ---------------------------------------------------------------------------


def _missing_run_context(tmp_path: Path) -> CheckpointContext:
    root = tmp_path / "job" / "checkpoints"
    return CheckpointContext(config=OmegaConf.create({"system": {"output": {"checkpoints": {"root": str(root)}}}}))


def _clear_rank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _RANK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_rank_env_priority_rank_beats_local_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    context = _missing_run_context(tmp_path)

    # RANK is read first and stops the search: rank 0 -> missing checkpoint raises.
    with pytest.raises(RuntimeError, match="run_missing"):
        asyncio.run(RunSource(run_id="run_missing").process(context))


def test_rank_env_priority_slurm_procid_defers_when_only_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("SLURM_PROCID", "1")
    context = _missing_run_context(tmp_path)

    result = asyncio.run(RunSource(run_id="run_missing").process(context))
    assert result.checkpoint_path is None


@pytest.mark.parametrize("rank_value", ["abc", "-1"])
def test_malformed_or_negative_rank_treated_as_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank_value: str,
) -> None:
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("RANK", rank_value)
    context = _missing_run_context(tmp_path)

    with pytest.raises(RuntimeError, match="run_missing"):
        asyncio.run(RunSource(run_id="run_missing").process(context))


def test_resolve_path_invariant_to_dataset_count() -> None:
    single = OmegaConf.create(
        {
            "system": {"output": {"checkpoints": {"root": "/base/ckpts"}}},
            "dataloader": {"training": {"dataset": ["ds1"]}},
        },
    )
    multi = OmegaConf.create(
        {
            "system": {"output": {"checkpoints": {"root": "/base/ckpts"}}},
            "dataloader": {"training": {"dataset": ["ds1", "ds2", "ds3"]}},
        },
    )

    single_path = RunSource.resolve_path(single, "run_A", fork=False)
    multi_path = RunSource.resolve_path(multi, "run_A", fork=False)

    assert single_path == multi_path
    assert single_path == Path("/base", "run_A", "last.ckpt")
    # No dataset name leaks into the resolved path.
    assert "ds1" not in single_path.parts
    assert "ds2" not in multi_path.parts


# ---------------------------------------------------------------------------
# run_id fall-through to a fresh uuid4 (train.py:_derive_run_identity + run_id)
# ---------------------------------------------------------------------------


def test_derive_run_identity_noop_then_uuid_fallthrough() -> None:
    ns = SimpleNamespace(config=OmegaConf.create({"training": {}}), logger=None)

    AnemoiTrainer._derive_run_identity(ns)

    # No RunSource configured -> the internal identity keys are never written.
    assert OmegaConf.select(ns.config, "training.run_id", default=None) is None
    assert OmegaConf.select(ns.config, "training.fork_run_id", default=None) is None

    run_id = AnemoiTrainer.run_id.func(ns)
    assert len(run_id) == 36
    assert str(uuid.UUID(run_id)) == run_id


def test_run_id_is_uuid_for_keyless_run() -> None:
    ns = SimpleNamespace(
        config=OmegaConf.create({"training": {"run_id": None, "fork_run_id": None}}),
        logger=None,
    )

    run_id = AnemoiTrainer.run_id.func(ns)

    # Parses cleanly as a canonical uuid4 string.
    assert uuid.UUID(run_id).version == 4


# ---------------------------------------------------------------------------
# __init__ writes the resolved run_id back onto config (train.py:177-181)
# ---------------------------------------------------------------------------


def _fresh_trainer_config(checkpoints_path: Path) -> DictConfig:
    """A minimal, validation-skipped trainer config for a fresh (no-source) run."""
    config_dict = {
        "config_validation": False,
        "training": {},
        "system": {
            "output": {
                "root": "",
                "checkpoints": {"root": checkpoints_path},
                "plots": "",
                "profiler": "",
                "logs": {"root": "", "wandb": "", "mlflow": "", "tensorboard": ""},
            },
            "input": {},
        },
        "diagnostics": {"log": {"mlflow": {"enabled": False}}},
        "data": {},
        "dataloader": {},
        "graph": {},
        "model": {},
        "task": {},
    }
    return OmegaConf.create(config_dict)


def test_init_writes_resolved_run_id_back_into_config(tmp_path: Path) -> None:
    config = _fresh_trainer_config(tmp_path / "ckpts")

    with (
        patch("anemoi.training.train.train.AnemoiTrainer._check_dry_run"),
        patch("anemoi.training.train.train.AnemoiTrainer._log_information"),
        patch("pathlib.Path.exists", return_value=True),
    ):
        trainer = AnemoiTrainer(config=config)

    # A fresh run mints a uuid and writes it back onto training.run_id.
    assert trainer.config.training.run_id == trainer.run_id
    assert uuid.UUID(trainer.config.training.run_id).version == 4


# ---------------------------------------------------------------------------
# Remote sources record no local checkpoint path (http.py)
# ---------------------------------------------------------------------------


async def _fake_download(url: str, dest: Path, max_retries: int, timeout: int) -> None:
    """Stand in for the network download by writing a real checkpoint to ``dest``."""
    del url, max_retries, timeout  # signature must match download_with_retry; only dest is used
    torch.save({"state_dict": {"weight": torch.zeros(2, 2)}}, dest)


def test_http_source_leaves_checkpoint_path_unset() -> None:
    context = CheckpointContext()

    with patch("anemoi.training.checkpoint.utils.download_with_retry", new=_fake_download):
        result = asyncio.run(HTTPSource(url="https://models.example.int/model.ckpt").process(context))

    # The temp download is cleaned up and no local path is recorded, so a trainer
    # reading last_checkpoint from this context would get None (no warm resume).
    assert result.checkpoint_path is None
    assert result.checkpoint_data is not None


# ---------------------------------------------------------------------------
# Seed scaling for an env-sourced seed below the threshold (seeding.py:44-46)
# ---------------------------------------------------------------------------


def test_env_seed_below_threshold_is_scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANEMOI_BASE_SEED", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("CUSTOM_BASE_SEED", "42")

    assert get_base_seed(base_seed_env="CUSTOM_BASE_SEED") == 42000
