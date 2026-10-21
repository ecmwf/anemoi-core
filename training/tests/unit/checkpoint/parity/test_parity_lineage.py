# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity tests for run lineage, MLflow tags, rank guards and seeding.

The tests run on CPU with no network access. A fake writer replaces remote downloads.
"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import pytest
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.sources.run import RunIdSource
from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.seeding import get_base_seed


# ---------------------------------------------------------------------------
# _update_paths: add the run ID to the output paths
# ---------------------------------------------------------------------------


def _paths_namespace(
    *,
    run_id: str | None,
    parent_run_server2server: str | None,
    fork_run_id: str | None,
    checkpoints_root: str,
    plots: str,
) -> SimpleNamespace:
    """Make a minimal ``self`` for ``AnemoiTrainer._update_paths``."""
    config = OmegaConf.create(
        {
            "system": {"output": {"checkpoints": {"root": checkpoints_root}, "plots": plots}},
            "training": {},
        },
    )
    return SimpleNamespace(
        run_id=run_id,
        parent_run_server2server=parent_run_server2server,
        _run_identity=(run_id, fork_run_id),
        _resume_run_id=run_id,
        _fork_run_id=fork_run_id,
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

    # A fork changes only checkpoints.root, not plots.
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
# _check_dry_run
# ---------------------------------------------------------------------------


def _dry_run_namespace(
    *,
    fork_run_id: str | None,
    parent_dry_run: bool,
    start_from_checkpoint: bool,
) -> SimpleNamespace:
    """Make a minimal ``self`` for ``AnemoiTrainer._check_dry_run`` with an MLflow logger."""
    config = OmegaConf.create(
        {
            # This path does not exist.
            "system": {"output": {"checkpoints": {"root": "/nonexistent/checkpoint/dir"}}},
            "training": {},
        },
    )
    return SimpleNamespace(
        start_from_checkpoint=start_from_checkpoint,
        logger=SimpleNamespace(logger_name="mlflow"),
        mlflow_logger=SimpleNamespace(_parent_dry_run=parent_dry_run),
        _run_identity=(None, fork_run_id),
        _resume_run_id=None,
        _fork_run_id=fork_run_id,
        config=config,
    )


def test_check_dry_run_reaches_the_same_answer_on_every_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """All ranks get the same dry-run result.

    The result controls checkpoint loading, so a rank-0-only result makes other ranks fail.
    """
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    ns = _dry_run_namespace(fork_run_id=None, parent_dry_run=True, start_from_checkpoint=True)

    AnemoiTrainer._check_dry_run(ns)

    assert ns.dry_run is True
    assert ns.start_from_checkpoint is False


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
    # A fork keeps start_from_checkpoint if the parent is dry.
    assert ns.start_from_checkpoint is True


# ---------------------------------------------------------------------------
# _skip_lightning_restore
# ---------------------------------------------------------------------------


def _skip_restore_namespace(loading: dict | None) -> SimpleNamespace:
    checkpoint = {"loading": loading} if loading is not None else {}
    return SimpleNamespace(config=OmegaConf.create({"training": {"checkpoint": checkpoint}}))


@pytest.mark.parametrize(
    ("loading", "expected"),
    [
        ({"_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"}, True),
        ({"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"}, False),
        ({"_target_": "nonexistent.LoaderClass"}, True),
    ],
)
def test_skip_lightning_restore_matches_loading_strategy(loading: dict, expected: bool) -> None:
    ns = _skip_restore_namespace(loading)

    # An unknown _target_ is not a resume. The loader build then reports the error before fit().
    assert AnemoiTrainer._skip_lightning_restore(ns) is expected


# ---------------------------------------------------------------------------
# MLflow lineage tags on resume and fork
# ---------------------------------------------------------------------------


def _create_parent_run(
    save_dir: str,
    experiment_name: str,
    *,
    tags: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> str:
    """Make a finished MLflow run in an offline file store and return its ID."""
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
    child_id = logger.run_id  # creates the child run

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

    # The same run continues. There is no child run.
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
# Builder and pipeline errors
# ---------------------------------------------------------------------------


def test_builder_malformed_source_target_raises_config_error() -> None:
    cfg = OmegaConf.create({"training": {"checkpoint": {"source": {"_target_": "nonexistent.Source"}}}})
    model = nn.Linear(2, 2)

    def _build_and_run() -> None:
        pipeline = build_checkpoint_pipeline(cfg)
        pipeline.execute(CheckpointContext(model=model, config=cfg))

    with pytest.raises(CheckpointConfigError) as excinfo:
        _build_and_run()

    assert "stage" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Rank-0 gate in the sources
# ---------------------------------------------------------------------------


def _missing_run_context(tmp_path: Path) -> CheckpointContext:
    root = tmp_path / "job" / "checkpoints"
    return CheckpointContext(config=OmegaConf.create({"system": {"output": {"checkpoints": {"root": str(root)}}}}))


def _set_rank(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    monkeypatch.setattr(rank_zero_only, "rank", rank)


def test_missing_checkpoint_raises_on_rank_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_rank(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="run_missing"):
        RunIdSource(run_id="run_missing").process(_missing_run_context(tmp_path))


def test_missing_checkpoint_defers_on_other_ranks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_rank(monkeypatch, 1)
    result = RunIdSource(run_id="run_missing").process(_missing_run_context(tmp_path))
    assert result.checkpoint_path is None


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

    single_path = RunIdSource.resolve_path(single, "run_A", fork=False)
    multi_path = RunIdSource.resolve_path(multi, "run_A", fork=False)

    assert single_path == multi_path
    assert single_path == Path("/base", "run_A", "last.ckpt")
    # The path contains no dataset name.
    assert "ds1" not in single_path.parts
    assert "ds2" not in multi_path.parts


# ---------------------------------------------------------------------------
# run_id falls back to a new uuid4
# ---------------------------------------------------------------------------


def test_run_identity_noop_then_uuid_fallthrough() -> None:
    from anemoi.training.checkpoint.sources.run import run_identity_from_config

    # With no RunIdSource, run_id is a new uuid4.
    assert run_identity_from_config(OmegaConf.create({"training": {}})) == (None, None)

    ns = SimpleNamespace(_run_identity=(None, None), _resume_run_id=None, _fork_run_id=None, logger=None)
    run_id = AnemoiTrainer.run_id.func(ns)
    assert len(run_id) == 36
    assert str(uuid.UUID(run_id)) == run_id


def test_run_id_is_uuid_for_keyless_run() -> None:
    ns = SimpleNamespace(_run_identity=(None, None), _resume_run_id=None, _fork_run_id=None, logger=None)

    run_id = AnemoiTrainer.run_id.func(ns)

    assert uuid.UUID(run_id).version == 4


# ---------------------------------------------------------------------------
# __init__ writes run_id into the config
# ---------------------------------------------------------------------------


def _fresh_trainer_config(checkpoints_path: Path) -> DictConfig:
    """Return a minimal trainer config for a new run, with no validation."""
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

    # A new run writes its uuid into training.run_id.
    assert trainer.config.training.run_id == trainer.run_id
    assert uuid.UUID(trainer.config.training.run_id).version == 4


# ---------------------------------------------------------------------------
# A base seed from the environment is not scaled
# ---------------------------------------------------------------------------


def test_env_seed_is_returned_unscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANEMOI_BASE_SEED", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("CUSTOM_BASE_SEED", "42")

    # derive_seed uses np.random.SeedSequence, so small seeds need no scaling.
    assert get_base_seed(base_seed_env="CUSTOM_BASE_SEED") == 42
