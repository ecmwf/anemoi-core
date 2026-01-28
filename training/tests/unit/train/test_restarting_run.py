# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path
from unittest.mock import patch

import pytest

from anemoi.training.config_types import Settings
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
) -> Settings:
    diagnostics_log = {
        "wandb": {
            "enabled": False,
            "offline": False,
            "log_model": False,
            "project": "Anemoi",
            "entity": None,
            "gradients": False,
            "parameters": False,
        },
        "tensorboard": {"enabled": False},
        "mlflow": {
            "enabled": False,
            "offline": False,
            "authentication": False,
            "tracking_uri": None,
            "experiment_name": "test",
            "project_name": "Anemoi",
            "system": False,
            "terminal": False,
            "run_name": None,
            "on_resume_create_child": True,
            "expand_hyperparams": [],
            "http_max_retries": 1,
            "max_params_length": 2000,
            "save_dir": None,
        },
        "interval": 100,
    }
    diagnostics_checkpoint = {
        "every_n_minutes": {"save_frequency": None, "num_models_saved": 0},
        "every_n_epochs": {"save_frequency": None, "num_models_saved": 0},
        "every_n_train_steps": {"save_frequency": None, "num_models_saved": 0},
    }
    diagnostics_plot = {
        "asynchronous": False,
        "datashader": False,
        "frequency": {"batch": 1, "epoch": 1},
        "parameters": [],
        "sample_idx": 0,
        "precip_and_related_fields": [],
        "colormaps": {},
        "datasets_to_plot": [],
        "callbacks": [],
    }
    diagnostics_benchmark = {
        "memory": {"enabled": False, "steps": 0, "warmup": 0, "extra_plots": False, "trace_rank0_only": False},
        "time": {"enabled": False, "verbose": False},
        "speed": {"enabled": False},
        "system": {"enabled": False},
        "model_summary": {"enabled": False},
        "snapshot": {"enabled": False, "steps": 0, "warmup": 0},
    }

    config = Settings.model_validate(
        {
            "system": {
                "output": {
                    "root": "",
                    "checkpoints": {"root": checkpoints_path or Path()},
                    "plots": "",
                    "logs": {"root": ""},
                },
                "input": {"warm_start": None},
                "hardware": {"num_gpus_per_model": 1, "num_gpus_per_node": 1, "num_nodes": 1},
            },
            "data": {},
            "dataloader": {
                "dataset": None,
                "training": {"datasets": {}},
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
                "grid_indices": {},
                "batch_size": {"training": 1, "validation": 1, "test": 1},
                "num_workers": {"training": 0, "validation": 0, "test": 0},
                "limit_batches": {"training": 1, "validation": 1, "test": 1},
                "pin_memory": False,
                "prefetch_factor": 1,
                "read_group_size": 1,
                "validation_rollout": 1,
            },
            "graph": {},
            "model": {"compile": None},
            "training": {
                "run_id": run_id,
                "fork_run_id": fork_run_id,
                "load_weights_only": False,
                "transfer_learning": False,
                "deterministic": False,
                "num_sanity_val_steps": 0,
                "max_epochs": 1,
                "max_steps": 1,
                "precision": "32-true",
                "model_task": "anemoi.training.train.tasks.GraphForecaster",
                "multistep_input": 1,
                "accum_grad_batches": 1,
                "loss_gradient_scaling": False,
                "ensemble_size_per_device": 1,
                "lr": {"rate": 1.0, "iterations": 1},
                "rollout": {"start": 1, "epoch_increment": 0, "max": 1},
                "gradient_clip": {"val": 0.0, "algorithm": "value"},
                "swa": {"enabled": False, "lr": 0},
                "submodules_to_freeze": [],
            },
            "diagnostics": {
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "checkpoint": diagnostics_checkpoint,
                "log": diagnostics_log,
                "progress_bar": {
                    "_target_": "pytorch_lightning.callbacks.TQDMProgressBar",
                    "refresh_rate": 1,
                },
                "debug": {"anomaly_detection": False},
                "plot": diagnostics_plot,
                "callbacks": [],
                "benchmark_profiler": diagnostics_benchmark,
                "check_val_every_n_epoch": 1,
                "print_memory_summary": False,
            },
        },
    )
    config.system.output.checkpoints.root = checkpoints_path or Path()
    config.system.input.warm_start = warm_start_path / warm_start if warm_start_path and warm_start else None
    return config


@pytest.fixture
def trainer_factory() -> AnemoiTrainer:
    def _make_trainer(mock_config: Settings) -> AnemoiTrainer:
        with patch("anemoi.training.train.train.LOGGER"), patch(
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


def test_restart_warm_start(
    trainer_factory: AnemoiTrainer,
    tmp_checkpoint_factory: pytest.TempPathFactory,
) -> None:
    """Test by assuming warm start is linked to run id.

    This resembles the case where we want to resume run
    using a checkpoint different from last.ckpt.
    """
    warm_start = "checkpoint_10.ckpt"
    warm_start_path = "mock-checkpoints"
    expected_path, warm_start_path = tmp_checkpoint_factory(
        ckpt_file_name=warm_start,
        ckpt_path_name=warm_start_path,
    )
    _, checkpoints_path = tmp_checkpoint_factory(
        ckpt_path_name=warm_start_path,
    )  # path where writing the checkpoints it's different

    config = build_mock_config(
        checkpoints_path=checkpoints_path,
        warm_start_path=warm_start_path,
        warm_start=warm_start,
    )

    trainer = trainer_factory(config)

    assert trainer.start_from_checkpoint is True
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
    with pytest.raises(FileNotFoundError, match=r"Warm start checkpoint not found"):
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
