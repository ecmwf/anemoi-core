# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import inspect
from pathlib import Path

import omegaconf
import pytest
import yaml
from hydra.utils import instantiate

from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
from anemoi.training.schemas.diagnostics import MlflowSchema


@pytest.fixture(scope="session")
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    # returns a session-scoped temporary directory
    return str(tmp_path_factory.mktemp("mlruns"))


@pytest.fixture
def tmp_uri(monkeypatch: pytest.MonkeyPatch, tmp_path: str) -> str:
    uri = f"sqlite:///{Path(tmp_path) / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def default_offline_config(tmp_path: str) -> omegaconf.DictConfig:
    base = """
    diagnostics:
      log:
        mlflow:
            _target_: anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger
            offline: Trure
            authentication: False
            tracking_uri: 'https:test.int'
            experiment_name: 'anemoi-debug'
            project_name: 'Anemoi'
            system: False
            terminal: True
            run_name: null
            on_resume_create_child: True
            expand_hyperparams:
                - config
            http_max_retries: 35
            max_params_length: 2000
            save_dir: '/scratch/example'
    """

    cfg = omegaconf.OmegaConf.create(yaml.safe_load(base))
    cfg.diagnostics.log.mlflow.save_dir = tmp_path

    return cfg


@pytest.fixture
def default_logger(tmp_path: str, tmp_uri: str) -> AnemoiMLflowLogger:
    return AnemoiMLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run",
        offline=True,
        tracking_uri=tmp_uri,
        authentication=False,
        save_dir=tmp_path,
    )


def sqlite_uri_for(save_dir: str) -> str:
    return f"sqlite:///{Path(save_dir).resolve() / 'mlflow.db'}"


def create_run(save_dir: str, experiment_name: str) -> str:
    import contextlib

    import mlflow

    mlflow.set_tracking_uri(sqlite_uri_for(save_dir))
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        return mlflow.active_run().info.run_id


def test_offline_logger(default_offline_config: omegaconf.DictConfig) -> None:
    mlflow_logger = instantiate(default_offline_config.diagnostics.log.mlflow)
    save_dir = default_offline_config.diagnostics.log.mlflow.save_dir
    assert mlflow_logger.tracking_uri == sqlite_uri_for(save_dir)


def test_offline_db_created(default_offline_config: omegaconf.DictConfig) -> None:
    save_dir = Path(default_offline_config.diagnostics.log.mlflow.save_dir)
    instantiate(default_offline_config.diagnostics.log.mlflow)
    assert (save_dir / "mlflow.db").exists(), "mlflow.db was not created in save_dir"


def test_offline_artifact_location(default_offline_config: omegaconf.DictConfig) -> None:
    import mlflow

    save_dir = Path(default_offline_config.diagnostics.log.mlflow.save_dir).resolve()
    mlflow_logger = instantiate(default_offline_config.diagnostics.log.mlflow)
    # Trigger experiment creation by accessing the experiment property
    _ = mlflow_logger.experiment
    mlflow.set_tracking_uri(mlflow_logger.tracking_uri)
    experiment = mlflow.get_experiment_by_name(default_offline_config.diagnostics.log.mlflow.experiment_name)
    assert experiment is not None
    assert experiment.artifact_location == str(save_dir / "mlartifacts")


def test_offline_legacy_store_warning(
    tmp_path_factory: pytest.TempPathFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    legacy_dir = tmp_path_factory.mktemp("legacy_mlruns")
    (legacy_dir / "0").mkdir()  # numeric subdir mimics old filesystem store layout
    with (
        caplog.at_level(logging.WARNING, logger="anemoi.training.diagnostics.mlflow.logger"),
        contextlib.suppress(Exception),
    ):
        AnemoiMLflowLogger(
            experiment_name="test",
            offline=True,
            authentication=False,
            save_dir=str(legacy_dir),
        )
    assert any(
        "legacy" in msg.lower() for msg in caplog.messages
    ), "Expected a deprecation warning about legacy MLflow filesystem store"


def test_offline_resumed_logger(default_offline_config: omegaconf.DictConfig) -> None:
    save_dir = default_offline_config.diagnostics.log.mlflow.save_dir
    run_id = create_run(
        save_dir=save_dir,
        experiment_name=default_offline_config.diagnostics.log.mlflow.experiment_name,
    )
    logger_resumed = instantiate(
        default_offline_config.diagnostics.log.mlflow,
        run_id=run_id,
        fork_run_id=None,
    )
    assert logger_resumed.tracking_uri == sqlite_uri_for(save_dir)


def test_offline_forked_logger(default_offline_config: omegaconf.DictConfig) -> None:
    save_dir = default_offline_config.diagnostics.log.mlflow.save_dir
    fork_run_id = create_run(
        save_dir=save_dir,
        experiment_name=default_offline_config.diagnostics.log.mlflow.experiment_name,
    )
    logger_forked = instantiate(
        default_offline_config.diagnostics.log.mlflow,
        run_id=None,
        fork_run_id=fork_run_id,
    )
    assert logger_forked.tracking_uri == sqlite_uri_for(save_dir)


def test_mlflowlogger_params_limit(default_logger: AnemoiMLflowLogger) -> None:

    default_logger._max_params_length = 3
    params = {"lr": 0.001, "path": "era5", "anemoi.version": 1.5, "bounding": True}
    # # Expect an exception when logging too many hyperparameters
    with pytest.raises(ValueError, match=r"Too many params:"):
        default_logger.log_hyperparams(params)


def test_mlflowlogger_metric_deduplication(default_logger: AnemoiMLflowLogger) -> None:

    default_logger.log_metrics({"foo": 1.0}, step=5)
    default_logger.log_metrics({"foo": 1.0}, step=5)  # duplicate
    # Only the first metric should be logged
    assert len(default_logger._logged_metrics) == 1
    assert next(iter(default_logger._logged_metrics))[0] == "foo"  # key
    assert next(iter(default_logger._logged_metrics))[1] == 5  # step


def test_mlflow_schema() -> None:
    config = {
        "_target_": "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger",
        "enabled": False,
        "offline": False,
        "authentication": False,
        "tracking_uri": None,  # You had ??? — using None as placeholder
        "experiment_name": "anemoi-debug",
        "project_name": "Anemoi",
        "system": False,
        "terminal": False,
        "run_name": None,  # If set to null, the run name will be a random UUID
        "on_resume_create_child": True,
        "expand_hyperparams": ["config"],  # Which keys in hyperparams to expand
        "http_max_retries": 35,
        "max_params_length": 2000,
    }
    schema = MlflowSchema(**config)

    assert schema.target_ == "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"
    assert schema.save_dir is None


def test_mlflow_schema_backward_compatibility() -> None:
    config = {
        "enabled": False,
        "offline": False,
        "authentication": False,
        "tracking_uri": None,  # You had ??? — using None as placeholder
        "experiment_name": "anemoi-debug",
        "project_name": "Anemoi",
        "system": False,
        "terminal": False,
        "run_name": None,  # If set to null, the run name will be a random UUID
        "on_resume_create_child": True,
        "expand_hyperparams": ["config"],  # Which keys in hyperparams to expand
        "http_max_retries": 35,
        "max_params_length": 2000,
    }
    schema = MlflowSchema(**config)

    assert schema.target_ == "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"
    assert schema.save_dir is None


def test_mlflow_schema_covers_logger_init_params() -> None:
    """Assert every BaseAnemoiMLflowLogger __init__ param has a field in MlflowSchema.

    run_id and fork_run_id are excluded because they are injected at runtime
    by get_mlflow_logger, not read from the config schema.
    """
    # Parameters injected at runtime rather than sourced from config
    runtime_params = {"self", "run_id", "fork_run_id"}

    logger_params = set(inspect.signature(AnemoiMLflowLogger.__init__).parameters) - runtime_params
    schema_fields = set(MlflowSchema.model_fields)

    missing = logger_params - schema_fields
    assert not missing, f"MlflowSchema is missing fields for logger params: {missing}"
