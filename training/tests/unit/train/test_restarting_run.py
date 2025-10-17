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

from anemoi.training.train.train import AnemoiTrainer


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


def build_mock_config(
    *,
    run_id: str | None = None,
    fork_run_id: str | None = None,
    warm_start: str | None = None,
    checkpoints_path: Path | None = None,
    warm_start_path: Path | None = None,
) -> MagicMock:
    config = MagicMock()
    config.config_validation = False
    config.training.run_id = run_id
    config.training.fork_run_id = fork_run_id
    config.training.load_weights_only = False
    config.training.transfer_learning = False
    config.hardware.files.warm_start = warm_start
    config.hardware.paths.checkpoints = checkpoints_path
    config.hardware.paths.warm_start = warm_start_path
    config.diagnostics.log.mlflow.enabled = False
    return config


@pytest.fixture
def trainer_factory() -> AnemoiTrainer:
    def _make_trainer(mock_config: MagicMock) -> AnemoiTrainer:
        with patch("anemoi.training.train.train.OmegaConf.to_object", return_value=mock_config), patch(
            "anemoi.training.train.train.DictConfig",
            return_value=mock_config,
        ), patch("anemoi.training.train.train.UnvalidatedBaseSchema", return_value=mock_config), patch(
            "anemoi.training.train.train.LOGGER",
        ), patch(
            "anemoi.training.train.train.AnemoiTrainer._check_dry_run",
        ), patch(
            "anemoi.training.train.train.AnemoiTrainer._log_information",
        ), patch(
            "pathlib.Path.exists",
            return_value=True,
        ):
            return AnemoiTrainer(config=mock_config)

    return _make_trainer


def test_restart_run_id(trainer_factory: AnemoiTrainer, tmp_checkpoint_factory: pytest.TempPathFactory) -> None:
    run_id = "run-id-123"
    expected_path, checkpoints_path = tmp_checkpoint_factory(rid=run_id, ckpt_path_name="mock_checkpoints")

    config = build_mock_config(
        run_id=run_id,
        checkpoints_path=checkpoints_path,
        warm_start_path=None,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id
    assert trainer.last_checkpoint == expected_path


def test_restart_fork_run_id(trainer_factory: AnemoiTrainer, tmp_checkpoint_factory: pytest.TempPathFactory) -> None:
    fork_run_id = "fork-id-456"
    # Forking assumes that the checkpoint is still stored in the same root path as where we
    # will be writing new checkpoints in the training run
    expected_path, checkpoints_path = tmp_checkpoint_factory(rid=fork_run_id, ckpt_path_name="mock_checkpoints")

    config = build_mock_config(
        fork_run_id=fork_run_id,
        checkpoints_path=checkpoints_path,
        warm_start_path=None,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id != fork_run_id
    assert trainer.last_checkpoint == expected_path


def test_restart_warm_start_path(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test by assuming warm start is unlinked to run or fork run ids."""
    warm_start = "checkpoint_10.ckpt"
    warm_start_fork_run_id = "fork-id-446"
    warm_start_path = "mock-pretrained-checkpoints"
    expected_path, warm_start_path = tmp_checkpoint_factory(
        rid=warm_start_fork_run_id,
        ckpt_file_name=warm_start,
        ckpt_path_name=warm_start_path,
    )
    _, checkpoints_path = tmp_checkpoint_factory()  # path where writing the checkpoints it's different
    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start_path=warm_start_path / Path(warm_start_fork_run_id),  # needs to pass full path
        warm_start=warm_start,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id != warm_start_fork_run_id
    assert trainer.last_checkpoint == expected_path


def test_restart_warm_start_fork_run_id(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test by assuming warm start is linked to fork run id."""
    warm_start = "checkpoint_10.ckpt"
    fork_run_id = "fork-id-446"
    warm_start_path = "mock-pretrained-checkpoints"
    expected_path, warm_start_path = tmp_checkpoint_factory(
        rid=fork_run_id,
        ckpt_file_name=warm_start,
        ckpt_path_name=warm_start_path,
    )
    _, checkpoints_path = tmp_checkpoint_factory()  # path where writing the checkpoints it's different

    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start_path=warm_start_path / Path(fork_run_id),  #
        fork_run_id=fork_run_id,
        warm_start=warm_start,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id != fork_run_id
    assert trainer.last_checkpoint == expected_path


def test_restart_warm_start_run_id(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test by assuming warm start is linked to run id.

    This resembles the case where we want to resume run
    using a checkpoint different from last.ckpt.
    """
    warm_start = "checkpoint_10.ckpt"
    run_id = "id-222"
    warm_start_path = "mock-checkpoints"
    expected_path, warm_start_path = tmp_checkpoint_factory(
        rid=run_id,
        ckpt_file_name=warm_start,
        ckpt_path_name=warm_start_path,
    )
    _, checkpoints_path = tmp_checkpoint_factory(
        ckpt_path_name=warm_start_path,
    )  # path where writing the checkpoints it's different

    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start_path=warm_start_path / Path(run_id),  #
        warm_start=warm_start,
        run_id=run_id,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id
    assert trainer.last_checkpoint == expected_path


def test_restart_warm_start_misconfiguration_path(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test to assert misconfiguration is missing path for warm_start."""
    warm_start = "checkpoint_10.ckpt"
    run_id = "id-222"
    _, checkpoints_path = tmp_checkpoint_factory()

    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start=warm_start,
        run_id=run_id,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id

    # This should raise because the warm start file is not in the expected location
    with pytest.raises(
        AssertionError,
        match=r"Please configure config.hardware.paths.warm_start correctly, found: None",
    ):
        _ = trainer.last_checkpoint


def test_restart_warm_start_misconfiguration_filename(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test to assert misconfiguration is missing file for warm_start."""
    run_id = "id-222"
    warm_start_path = "mock-checkpoints"
    _, warm_start_path = tmp_checkpoint_factory(
        rid=run_id,
        ckpt_path_name=warm_start_path,
    )
    _, checkpoints_path = tmp_checkpoint_factory(
        ckpt_path_name=warm_start_path,
    )  # path where writing the checkpoints it's different

    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start_path=warm_start_path / Path(run_id),  #
        run_id=run_id,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id
    # This should raise because the warm start file is not in the expected location
    with pytest.raises(
        AssertionError,
        match=r"Please configure config.hardware.files.warm_start correctly, found: None",
    ):
        _ = trainer.last_checkpoint


def test_warm_start_file_not_found(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test to assert file not found for warm_start."""
    warm_start = "checkpoint_10.ckpt"
    run_id = "id-222"
    warm_start_path = "mock-checkpoints"
    _, warm_start_path = tmp_checkpoint_factory(
        rid=run_id,
        ckpt_file_name=warm_start,
        ckpt_path_name=warm_start_path,
        skip_creation=True,
    )
    _, checkpoints_path = tmp_checkpoint_factory(
        ckpt_path_name=warm_start_path,
    )  # path where writing the checkpoints it's different

    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start_path=warm_start_path / Path(run_id),  #
        warm_start=warm_start,
        run_id=run_id,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id
    with pytest.raises(AssertionError, match=r"Warm start checkpoint not found"):
        _ = trainer.last_checkpoint


def test_restart_run_id_file_not_found(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test to assert file not found for resuming."""
    run_id = "run-id-123"
    _, checkpoints_path = tmp_checkpoint_factory(rid=run_id, ckpt_path_name="mock_checkpoints", skip_creation=True)

    config = build_mock_config(
        run_id=run_id,
        checkpoints_path=checkpoints_path,
        warm_start_path=None,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
    assert trainer.run_id == run_id
    with pytest.raises(RuntimeError, match=r"Could not find last checkpoint"):
        _ = trainer.last_checkpoint
