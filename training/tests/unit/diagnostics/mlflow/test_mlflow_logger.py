# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
from collections.abc import Iterator
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


# --- lineage tags on resume and fork ---------------------------------------------------


@pytest.fixture
def lineage_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A fresh offline MLflow store per test; the global tracking URI is restored afterwards."""
    import mlflow

    previous_uri = mlflow.get_tracking_uri()
    yield str(tmp_path_factory.mktemp("lineage") / "mlruns")
    mlflow.set_tracking_uri(previous_uri)


def _create_parent_run(
    save_dir: str,
    *,
    tags: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> str:
    """Create a finished run in an offline file store, with optional tags and params, and return its id."""
    import contextlib

    import mlflow

    mlflow.set_tracking_uri(f"file://{save_dir}")
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        mlflow.create_experiment("exp")
    mlflow.set_experiment("exp")
    with mlflow.start_run(tags=tags or {}):
        for key, value in (params or {}).items():
            mlflow.log_param(key, value)
        return mlflow.active_run().info.run_id


def _offline_logger(save_dir: str, **kwargs: object) -> AnemoiMLflowLogger:
    return AnemoiMLflowLogger(
        experiment_name="exp",
        offline=True,
        tracking_uri=None,
        authentication=False,
        save_dir=save_dir,
        **kwargs,
    )


def _tags(save_dir: str, run_id: str) -> dict[str, str]:
    import mlflow

    return mlflow.MlflowClient(f"file://{save_dir}").get_run(run_id).data.tags


def test_resume_creates_a_child_run_tagged_resumed_and_linked_to_the_parent(lineage_store: str) -> None:
    parent = _create_parent_run(lineage_store)

    child_id = _offline_logger(lineage_store, run_id=parent, fork_run_id=None).run_id

    tags = _tags(lineage_store, child_id)
    assert child_id != parent
    assert tags.get("resumedRun") == "True"
    assert tags.get("mlflow.parentRunId") == parent


def test_fork_tags_the_new_run_with_the_forked_run_id(lineage_store: str) -> None:
    parent = _create_parent_run(lineage_store)

    fork_id = _offline_logger(lineage_store, run_id=None, fork_run_id=parent).run_id

    tags = _tags(lineage_store, fork_id)
    assert tags.get("forkedRun") == "True"
    assert tags.get("forkedRunId") == parent


def test_resume_without_a_child_run_continues_the_parent_in_place(lineage_store: str) -> None:
    """With on_resume_create_child=False the parent run itself continues: no child, no parent link."""
    import mlflow

    parent = _create_parent_run(lineage_store)

    logger = _offline_logger(lineage_store, run_id=parent, fork_run_id=None, on_resume_create_child=False)

    assert logger.run_id == parent
    assert "mlflow.parentRunId" not in logger.tags
    assert mlflow.MlflowClient(f"file://{lineage_store}").get_run(parent).info.status == "RUNNING"


@pytest.mark.parametrize(
    ("lineage", "attribute"),
    [("resume", "_parent_run_server2server"), ("fork", "_fork_run_server2server")],
)
def test_server2server_parent_lineage_is_read_from_the_parent_run(
    lineage_store: str,
    lineage: str,
    attribute: str,
) -> None:
    """A parent synced server-to-server exposes its original offline run id for the path lookup."""
    parent = _create_parent_run(
        lineage_store,
        tags={"server2server": "True"},
        params={"metadata.offline_run_id": "origin-run"},
    )
    ids = {"run_id": parent, "fork_run_id": None} if lineage == "resume" else {"run_id": None, "fork_run_id": parent}

    logger = _offline_logger(lineage_store, **ids)

    assert getattr(logger, attribute) == "origin-run"
