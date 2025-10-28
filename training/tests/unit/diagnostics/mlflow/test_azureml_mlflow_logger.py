from pathlib import Path
import mlflow
from mlflow import MlflowClient
import pytest

from anemoi.training.diagnostics.mlflow.azureml import AnemoiAzureMLflowLogger
from anemoi.training.diagnostics.mlflow.azureml import AzureIdentity
from anemoi.training.schemas.diagnostics import AzureMlflowSchema


@pytest.fixture(scope="session")
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    # returns a session-scoped temporary directory
    return str(tmp_path_factory.mktemp("mlruns"))


@pytest.fixture
def tmp_uri(monkeypatch, tmp_path):
    uri = (Path(tmp_path) / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def tmp_client(tmp_uri):
    return MlflowClient(tmp_uri)


@pytest.fixture
def default_logger(tmp_path, tmp_uri) -> AnemoiAzureMLflowLogger:
    logger = AnemoiAzureMLflowLogger(
        identity=AzureIdentity("managed"),
        experiment_name="test_experiment",
        run_name="test_run",
        offline=False,
        tracking_uri=tmp_uri,
        authentication=False,
        save_dir=tmp_path,
    )
    return logger


def test_azure_mlflowlogger_no_log_params(default_logger, tmp_client):
    mlflow.set_experiment("ci-test")
    exp = tmp_client.get_experiment_by_name("ci-test")
    params = {"lr": 0.001, "path": "era5", "anemoi.version": 1.5, "bounding": True}
    default_logger.log_hyperparams(params)
    # assert that no params were actually logged
    runs = tmp_client.search_runs(experiment_ids=[exp.experiment_id])
    assert not any(bool(run.data.metrics) for run in runs)


def test_azure_mlflowlogger_metric_deduplication(default_logger):
    default_logger.log_metrics({"foo": 1.0}, step=5)
    default_logger.log_metrics({"foo": 1.0}, step=5)  # duplicate
    # Only the first metric should be logged
    assert len(default_logger._logged_metrics) == 1
    assert next(iter(default_logger._logged_metrics))[0] == "foo"  # key
    assert next(iter(default_logger._logged_metrics))[1] == 5  # step


def test_azure_mlflow_schema():
    config = {
    '_target_':"anemoi.training.diagnostics.mlflow.azureml.AnemoiAzureMLflowLogger",
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
    schema = AzureMlflowSchema(**config)

    assert schema.target_ == "anemoi.training.diagnostics.mlflow.azureml.AnemoiAzureMLflowLogger"
