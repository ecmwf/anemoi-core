# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from anemoi.training.diagnostics.mlflow.utils import MAX_PARAMS_LENGTH_PARAM

pytest.importorskip("mlflow_export_import")


@pytest.fixture
def mlflow_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    # the module needs TMPDIR or SCRATCH at import time
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    from anemoi.training.utils import mlflow_sync

    return mlflow_sync


def make_params(num_params: int, max_params_length: int | None = None) -> dict[str, str]:
    """Make `num_params` params left after `clean_config_params`, the run's limit included."""
    params = {}
    if max_params_length is not None:
        params[MAX_PARAMS_LENGTH_PARAM] = str(max_params_length)
    params.update({f"config.param_{i}": str(i) for i in range(num_params - len(params))})
    params["system.hardware.num_gpus"] = "4"  # removed by clean_config_params
    return params


def test_check_params_length_uses_run_limit(mlflow_sync: ModuleType) -> None:
    mlflow_sync.MlFlowSync.check_params_length(make_params(2500, max_params_length=3000))
    with pytest.raises(ValueError, match=r"Too many params: 3001 > 3000"):
        mlflow_sync.MlFlowSync.check_params_length(make_params(3001, max_params_length=3000))


def test_check_params_length_default_limit(mlflow_sync: ModuleType) -> None:
    mlflow_sync.MlFlowSync.check_params_length(make_params(2000))
    with pytest.raises(ValueError, match=r"Too many params: 2001 > 2000"):
        mlflow_sync.MlFlowSync.check_params_length(make_params(2001))


def test_check_params_length_does_not_mutate(mlflow_sync: ModuleType) -> None:
    params = make_params(10)
    mlflow_sync.MlFlowSync.check_params_length(params)
    assert "system.hardware.num_gpus" in params


def _mock_sync(mocker: MockerFixture, mlflow_sync: ModuleType, params: dict[str, str]) -> MagicMock:
    run = MagicMock()
    run.info.run_id = "offline-run"
    run.info.status = "FINISHED"
    run.info.lifecycle_stage = "active"
    run.data.params = params
    run.data.tags = {}
    run.inputs.model_inputs = []

    client = MagicMock()
    client.get_run.return_value = run
    client.create_run.return_value.info.run_id = "server-run"
    mocker.patch.object(mlflow_sync.mlflow, "MlflowClient", return_value=client)
    mocker.patch.object(mlflow_sync.mlflow, "set_tracking_uri")
    mocker.patch.object(mlflow_sync, "create_http_client")
    mocker.patch.object(mlflow_sync.mlflow_utils, "strip_underscores", return_value={"user_id": "user"})
    mocker.patch.object(mlflow_sync, "close_and_clean_temp")
    mocker.patch.object(mlflow_sync, "_get_metrics_with_steps", return_value={})
    mocker.patch.object(mlflow_sync, "_inputs_to_dict", return_value=[])
    mocker.patch.object(mlflow_sync, "_import_inputs")
    mocker.patch.object(mlflow_sync, "_log_metrics")
    mocker.patch.object(mlflow_sync.MlFlowSync, "_check_source_tracking_uri", return_value=False)
    mocker.patch.object(mlflow_sync.MlFlowSync, "check_run_is_logged", return_value=False)
    mocker.patch.object(mlflow_sync.MlFlowSync, "_get_dst_experiment_id", return_value="0")
    mocker.patch.object(mlflow_sync.MlFlowSync, "_get_artifacts_path", return_value="artifacts")
    return client


def test_sync_params_added_by_sync_not_counted(mocker: MockerFixture, mlflow_sync: ModuleType) -> None:
    """A run exactly at its limit still syncs, although the sync adds lineage params."""
    params = make_params(8, max_params_length=10)
    params["config.training.fork_run_id"] = "None"
    params["metadata.run_id"] = "offline-run"
    assert len(mlflow_sync.clean_config_params(dict(params))) == 10
    client = _mock_sync(mocker, mlflow_sync, params)

    mlflow_sync.MlFlowSync("source", "dest", "offline-run").sync()

    logged = [p for call in client.log_batch.call_args_list for p in call.kwargs.get("params") or []]
    assert len(logged) > 10
    assert {"metadata.offline_run_id", "config.training.offline_run_id"} <= {p.key for p in logged}
    client.set_terminated.assert_called_once_with("server-run", "FINISHED")


def test_sync_too_many_params_fails_before_creating_run(mocker: MockerFixture, mlflow_sync: ModuleType) -> None:
    client = _mock_sync(mocker, mlflow_sync, make_params(11, max_params_length=10))

    with pytest.raises(ValueError, match=r"Too many params: 11 > 10"):
        mlflow_sync.MlFlowSync("source", "dest", "offline-run").sync()

    client.create_run.assert_not_called()
