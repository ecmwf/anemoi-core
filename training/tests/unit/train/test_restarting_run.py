# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.train.train import AnemoiTrainer

_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunSource"
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
    """Build a minimal trainer config on the ``training.checkpoint.source`` surface.

    ``source`` is the ``training.checkpoint.source`` block (a ``RunSource`` or
    ``LocalSource`` ``_target_`` dict), or ``None`` for a fresh run with nothing
    to resume.
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
        # ``Path.exists`` is patched True only for the duration of construction;
        # ``last_checkpoint`` is a lazily-evaluated cached_property, so tests that
        # assert a missing checkpoint access it after this context exits and see
        # the real filesystem.
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


# These tests cover the trainer-level run-lineage wiring at construction time
# (``start_from_checkpoint`` detection and ``run_id`` derivation). The checkpoint
# *path resolution* itself now lives in the acquisition layer and is exercised
# there (``sources/test_run.py`` for RunSource resolve_path / resume / fork /
# missing-checkpoint, ``sources/test_local.py`` for the explicit-file path and
# CheckpointNotFoundError); the trainer reads the resolved path back from the
# executed pipeline context (``AnemoiTrainer.last_checkpoint``, covered in
# ``test_checkpoint_loading.py``).


def test_resume_run_source(trainer_factory: AnemoiTrainer, tmp_checkpoint_factory: pytest.TempPathFactory) -> None:
    """RunSource (fork=False) resumes the same run: start_from_checkpoint set, run_id preserved."""
    run_id = "run-id-123"
    _, checkpoints_path = tmp_checkpoint_factory(rid=run_id, ckpt_path_name="mock_checkpoints")
    config = build_mock_config(source=_run_source(run_id), checkpoints_path=checkpoints_path)

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id


def test_fork_run_source(trainer_factory: AnemoiTrainer, tmp_checkpoint_factory: pytest.TempPathFactory) -> None:
    """RunSource (fork=True) starts from a parent run but mints a new run id."""
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
    """A LocalSource explicit path marks the run as starting from a checkpoint."""
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
    """Without a training.checkpoint.source there is nothing to resume."""
    _, checkpoints_path = tmp_checkpoint_factory()
    config = build_mock_config(source=None, checkpoints_path=checkpoints_path)

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is False
    assert trainer.last_checkpoint is None
