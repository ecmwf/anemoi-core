# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.train.train import AnemoiTrainer

_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOCAL_SOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"


@pytest.fixture(scope="session")
def tmp_checkpoint_factory(tmp_path_factory: pytest.TempPathFactory) -> (Path, Path):
    def _create_checkpoint(
        rid: str | None = None,
        ckpt_path_name: str = "mock_checkpoints",
        ckpt_file_name: str = "last.ckpt",
        skip_creation: bool = False,
    ) -> (Path, Path):
        base = tmp_path_factory.mktemp(ckpt_path_name)
        checkpoint_dir = base / rid if rid else base
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = checkpoint_dir / ckpt_file_name
        if not skip_creation:
            ckpt_file.write_text("fake checkpoint")
        return ckpt_file, base  # full path to the .ckpt

    return _create_checkpoint


def _run_source(run_id: str, *, fork: bool = False) -> dict:
    return {"_target_": _RUN_SOURCE, "run_id": run_id, "fork": fork}


def _local_source(path: Path) -> dict:
    return {"_target_": _LOCAL_SOURCE, "path": str(path)}


def build_mock_config(
    *,
    source: dict | None = None,
    checkpoints_path: Path | None = None,
) -> DictConfig:
    """Build a minimal trainer config.

    ``source`` is the ``training.checkpoint.source`` block, or ``None`` for a fresh run.
    """
    training = {"checkpoint": {"source": source}} if source is not None else {}
    config_dict = {
        "config_validation": False,
        "training": training,
        "system": {
            "output": {
                "root": "",
                "checkpoints": {"root": checkpoints_path},
                "plots": "",
                "profiler": "",
                "logs": {
                    "root": "",
                    "wandb": "",
                    "mlflow": "",
                    "tensorboard": "",
                },
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


@pytest.fixture
def trainer_factory() -> AnemoiTrainer:
    def _make_trainer(mock_config: MagicMock) -> AnemoiTrainer:
        # ``Path.exists`` is patched only during construction. ``last_checkpoint``
        # is lazy, so later access sees the real filesystem.
        with (
            patch(
                "anemoi.training.train.train.LOGGER",
            ),
            patch(
                "anemoi.training.train.train.AnemoiTrainer._check_dry_run",
            ),
            patch(
                "anemoi.training.train.train.AnemoiTrainer._log_information",
            ),
            patch(
                "pathlib.Path.exists",
                return_value=True,
            ),
        ):
            return AnemoiTrainer(config=mock_config)

    return _make_trainer


# These tests check ``start_from_checkpoint`` and ``run_id`` at construction.
# The sources tests check path resolution.


def test_resume_run_source(trainer_factory: AnemoiTrainer, tmp_checkpoint_factory: pytest.TempPathFactory) -> None:
    """RunIdSource with fork=False resumes the same run and keeps the run_id."""
    run_id = "run-id-123"
    _, checkpoints_path = tmp_checkpoint_factory(rid=run_id, ckpt_path_name="mock_checkpoints")
    config = build_mock_config(source=_run_source(run_id), checkpoints_path=checkpoints_path)

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id


def test_fork_run_source(trainer_factory: AnemoiTrainer, tmp_checkpoint_factory: pytest.TempPathFactory) -> None:
    """RunIdSource with fork=True starts from the parent run with a new run_id."""
    parent_run_id = "fork-id-456"
    _, checkpoints_path = tmp_checkpoint_factory(rid=parent_run_id, ckpt_path_name="mock_checkpoints")
    config = build_mock_config(source=_run_source(parent_run_id, fork=True), checkpoints_path=checkpoints_path)

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id != parent_run_id


def test_local_source_is_start_from_checkpoint(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """A LocalSource path sets start_from_checkpoint."""
    expected_path, checkpoints_path = tmp_checkpoint_factory(
        ckpt_file_name="checkpoint_10.ckpt",
        ckpt_path_name="mock-pretrained-checkpoints",
    )
    config = build_mock_config(source=_local_source(expected_path), checkpoints_path=checkpoints_path)

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True


def test_no_source_is_fresh_run(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Without a source, the run is a fresh run."""
    _, checkpoints_path = tmp_checkpoint_factory()
    config = build_mock_config(source=None, checkpoints_path=checkpoints_path)

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is False
    assert trainer.last_checkpoint is None
    assert uuid.UUID(trainer.run_id).version == 4


@pytest.mark.parametrize(
    ("run_id", "fork_run_id", "server2server", "expected_checkpoints", "expected_plots"),
    [
        ("run-abc", None, None, Path("/base/ckpts/run-abc"), Path("/base/plots/run-abc")),
        ("local-run", None, "remote-parent", Path("/base/ckpts/remote-parent"), Path("/base/plots/remote-parent")),
        # A non-zero rank of a fork has no run_id: only checkpoints.root changes.
        (None, "fork-999", None, Path("/base/ckpts/fork-999"), "/base/plots"),
    ],
)
def test_update_paths_appends_the_lineage_id(
    run_id: str | None,
    fork_run_id: str | None,
    server2server: str | None,
    expected_checkpoints: Path,
    expected_plots: Path | str,
) -> None:
    trainer = SimpleNamespace(
        run_id=run_id,
        parent_run_server2server=server2server,
        _run_identity=(run_id, fork_run_id),
        config=OmegaConf.create(
            {"system": {"output": {"checkpoints": {"root": "/base/ckpts"}, "plots": "/base/plots"}}},
        ),
    )

    AnemoiTrainer._update_paths(trainer)

    assert trainer.config.system.output.checkpoints.root == expected_checkpoints
    assert trainer.config.system.output.plots == expected_plots


@pytest.mark.parametrize(
    ("rank", "fork_run_id", "start_from_checkpoint"),
    [
        (0, None, False),
        # All ranks get the same result, because the result controls checkpoint loading.
        (1, None, False),
        # A fork starts from its parent checkpoint even if the parent is dry.
        (0, "fork-1", True),
    ],
)
def test_check_dry_run_for_a_dry_mlflow_parent(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    fork_run_id: str | None,
    start_from_checkpoint: bool,
) -> None:
    """A parent run without a checkpoint directory is dry. A resume from it starts from scratch."""
    monkeypatch.setattr(rank_zero_only, "rank", rank)
    trainer = SimpleNamespace(
        start_from_checkpoint=True,
        logger=SimpleNamespace(logger_name="mlflow"),
        mlflow_logger=SimpleNamespace(_parent_dry_run=True),
        _run_identity=(None, fork_run_id),
        config=OmegaConf.create({"system": {"output": {"checkpoints": {"root": "/nonexistent/checkpoint/dir"}}}}),
    )

    AnemoiTrainer._check_dry_run(trainer)

    assert trainer.dry_run is True
    assert trainer.start_from_checkpoint is start_from_checkpoint
