from pathlib import Path

import pytest

from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
from anemoi.training.schemas.diagnostics import MlflowSchema


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
def default_logger(tmp_path: str, tmp_uri: str) -> AnemoiMLflowLogger:
    return AnemoiMLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run",
        offline=True,
        tracking_uri=tmp_uri,
        authentication=False,
        save_dir=tmp_path,
    )


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
