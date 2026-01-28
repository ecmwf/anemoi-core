from pathlib import Path

import omegaconf
import pytest
import yaml
from hydra.utils import instantiate

from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger


@pytest.fixture(scope="session")
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    # returns a session-scoped temporary directory
    return str(tmp_path_factory.mktemp("mlruns"))


@pytest.fixture
def tmp_uri(monkeypatch: pytest.MonkeyPatch, tmp_path: str) -> Path:
    uri = (Path(tmp_path) / "mlruns").as_uri()
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


def create_run(save_dir: str, experiment_name: str) -> str:
    import contextlib

    import mlflow

    mlflow.set_tracking_uri(f"file://{save_dir}")
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        return mlflow.active_run().info.run_id


def test_offline_logger(default_offline_config: omegaconf.DictConfig) -> None:
    mlflow_logger = instantiate(default_offline_config.diagnostics.log.mlflow)
    assert not mlflow_logger.tracking_uri


def test_offline_resumed_logger(default_offline_config: omegaconf.DictConfig) -> None:

    run_id = create_run(
        save_dir=default_offline_config.diagnostics.log.mlflow.save_dir,
        experiment_name=default_offline_config.diagnostics.log.mlflow.experiment_name,
    )
    logger_resumed = instantiate(
        default_offline_config.diagnostics.log.mlflow,
        run_id=run_id,
        fork_run_id=None,
    )
    assert logger_resumed.tracking_uri == default_offline_config.diagnostics.log.mlflow.save_dir


def test_offline_forked_logger(default_offline_config: omegaconf.DictConfig) -> None:
    fork_run_id = create_run(
        save_dir=default_offline_config.diagnostics.log.mlflow.save_dir,
        experiment_name=default_offline_config.diagnostics.log.mlflow.experiment_name,
    )
    logger_forked = instantiate(
        default_offline_config.diagnostics.log.mlflow,
        run_id=None,
        fork_run_id=fork_run_id,
    )
    assert logger_forked.tracking_uri == default_offline_config.diagnostics.log.mlflow.save_dir


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
