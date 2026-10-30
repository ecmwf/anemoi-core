# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import RestException

from anemoi.training.diagnostics.mlflow.azureml import AnemoiAzureMLflowLogger
from anemoi.training.diagnostics.mlflow.azureml import AzureIdentity
from anemoi.training.schemas.diagnostics import AzureMlflowSchema


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
def tmp_client(tmp_uri: Path) -> MlflowClient:
    return MlflowClient(tmp_uri)


@pytest.fixture
def default_logger(tmp_path: str, tmp_uri: Path) -> AnemoiAzureMLflowLogger:
    return AnemoiAzureMLflowLogger(
        identity=AzureIdentity("managed"),
        experiment_name="test_experiment",
        run_name="test_run",
        offline=False,
        tracking_uri=tmp_uri,
        authentication=False,
        save_dir=tmp_path,
    )


def test_azure_mlflowlogger_no_log_params(default_logger: AnemoiAzureMLflowLogger, tmp_client: MlflowClient) -> None:
    mlflow.set_experiment("ci-test")
    exp = tmp_client.get_experiment_by_name("ci-test")
    params = {"lr": 0.001, "path": "era5", "anemoi.version": 1.5, "bounding": True}
    default_logger.log_hyperparams(params)
    # assert that no params were actually logged
    runs = tmp_client.search_runs(experiment_ids=[exp.experiment_id])
    assert not any(bool(run.data.metrics) for run in runs)


def test_azure_mlflowlogger_metric_deduplication(default_logger: AnemoiAzureMLflowLogger) -> None:
    default_logger.log_metrics({"foo": 1.0}, step=5)
    default_logger.log_metrics({"foo": 1.0}, step=5)  # duplicate
    # Only the first metric should be logged
    assert len(default_logger._logged_metrics) == 1
    assert next(iter(default_logger._logged_metrics))[0] == "foo"  # key
    assert next(iter(default_logger._logged_metrics))[1] == 5  # step


def test_azure_mlflow_schema() -> None:
    config = {
        "_target_": "anemoi.training.diagnostics.mlflow.azureml.AnemoiAzureMLflowLogger",
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


def test_azure_mlflow_schema_covers_logger_init_params() -> None:
    """Assert every AnemoiAzureMLflowLogger __init__ param has a field in AzureMlflowSchema.

    run_id and fork_run_id are excluded because they are injected at runtime
    by get_mlflow_logger, not read from the config schema.
    """
    # Parameters injected at runtime rather than sourced from config
    runtime_params = {"self", "run_id", "fork_run_id"}

    logger_params = set(inspect.signature(AnemoiAzureMLflowLogger.__init__).parameters) - runtime_params
    schema_fields = set(AzureMlflowSchema.model_fields)

    missing = logger_params - schema_fields
    assert not missing, f"AzureMlflowSchema is missing fields for logger params: {missing}"


def _refusing_log_metrics(monkeypatch: pytest.MonkeyPatch, errors: list) -> tuple[list, list]:
    """Make the MLflow write raise the given errors in turn (None = succeed); returns the calls made."""
    from pytorch_lightning.loggers.mlflow import MLFlowLogger

    from anemoi.training.diagnostics.mlflow import logger as logger_module

    calls, pauses = [], []

    def log_metrics(_self: MLFlowLogger, metrics: dict, step: int | None = None) -> None:
        calls.append((dict(metrics), step))
        error = errors.pop(0) if errors else None
        if error is not None:
            raise error

    monkeypatch.setattr(MLFlowLogger, "log_metrics", log_metrics)
    monkeypatch.setattr(logger_module.time, "sleep", pauses.append)
    return calls, pauses


def _etag_conflict() -> RestException:
    # as AzureML's MLflow server answered on 2026-10-29 (plum_jewel, at the end of a validation)
    return RestException(
        {"error_code": "RESOURCE_ALREADY_EXISTS", "message": 'Etag conflict on abc with etag "56074f6a"'},
    )


def test_a_write_conflict_is_retried(default_logger: AnemoiAzureMLflowLogger, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, pauses = _refusing_log_metrics(monkeypatch, [_etag_conflict(), None])
    default_logger.log_metrics({"val_loss": 1.0}, step=8000)
    assert calls == [({"val_loss": 1.0}, 8000)] * 2
    assert pauses == [2.0]


def test_a_persistent_write_conflict_drops_the_metrics_and_training_continues(
    default_logger: AnemoiAzureMLflowLogger,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls, pauses = _refusing_log_metrics(monkeypatch, [_etag_conflict() for _ in range(10)])
    default_logger.log_metrics({"val_loss": 1.0, "val_mse": 2.0}, step=8000)  # no exception
    assert len(calls) == 4
    assert pauses == [2.0, 5.0, 10.0]
    assert "they are not logged, and training continues" in caplog.text


def test_other_tracking_errors_still_raise(
    default_logger: AnemoiAzureMLflowLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refusing_log_metrics(monkeypatch, [RestException({"error_code": "INVALID_PARAMETER_VALUE", "message": "bad"})])
    with pytest.raises(RestException, match="bad"):
        default_logger.log_metrics({"val_loss": 1.0}, step=1)


def test_a_duplicate_metric_is_not_retried(
    default_logger: AnemoiAzureMLflowLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = RestException(
        {"error_code": "INTERNAL_ERROR", "message": "duplicate key value violates unique constraint"},
    )
    calls, pauses = _refusing_log_metrics(monkeypatch, [duplicate])
    default_logger.log_metrics({"val_loss": 1.0}, step=1)
    assert len(calls) == 1
    assert pauses == []
