# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``MlFlowSync._update_params_tags_runs`` tells a forked run apart by its ``forkedRun`` tag.

The logger tags a forked run with ``forkedRun`` / ``forkedRunId``; the trainer does not
write ``training.fork_run_id`` onto the config, so the logged params carry no
``config.training.fork_run_id`` to key on.
"""

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

_EXPORT_IMPORT_MODULES = [
    "mlflow_export_import",
    "mlflow_export_import.common",
    "mlflow_export_import.common.utils",
    "mlflow_export_import.client",
    "mlflow_export_import.client.client_utils",
    "mlflow_export_import.run",
    "mlflow_export_import.run.export_run",
    "mlflow_export_import.run.import_run",
    "mlflow_export_import.run.run_data_importer",
]

SRC_RUN_ID = "offline-child"
DST_RUN_ID = "online-child"
OFFLINE_PARENT_RUN_ID = "offline-parent"
ONLINE_PARENT_RUN_ID = "online-parent"


@pytest.fixture
def mlflow_sync_module() -> Iterator[ModuleType]:
    """Import ``mlflow_sync`` with the optional ``mlflow-export-import`` package stubbed out."""
    stubs = {name: MagicMock() for name in _EXPORT_IMPORT_MODULES}
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("anemoi.training.utils.mlflow_sync", None)
        yield importlib.import_module("anemoi.training.utils.mlflow_sync")


@pytest.fixture
def sync(mlflow_sync_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> object:
    """An ``MlFlowSync`` whose destination server already holds the synced parent run."""
    mlflow = mlflow_sync_module.mlflow
    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(mlflow, "get_experiment_by_name", MagicMock(return_value=MagicMock(experiment_id="0")))
    # One matching run, shaped like the ``search_runs`` DataFrame where it is read.
    synced_parent = SimpleNamespace(
        empty=False,
        shape=(1, 1),
        columns=[],
        iloc=[SimpleNamespace(run_id=ONLINE_PARENT_RUN_ID)],
    )
    monkeypatch.setattr(mlflow, "search_runs", MagicMock(return_value=synced_parent))
    return mlflow_sync_module.MlFlowSync(
        source_tracking_uri="file:///offline",
        dest_tracking_uri="http://localhost:5000",
        run_id=SRC_RUN_ID,
    )


def test_run_without_forked_run_tag_maps_its_own_run_id(sync: object) -> None:
    """A run without the ``forkedRun`` tag gets its own run id rewritten to the destination id."""
    params = {"metadata.run_id": SRC_RUN_ID}

    params, tags = sync._update_params_tags_runs(params, {}, dst_run_id=DST_RUN_ID, src_run_id=SRC_RUN_ID)

    assert params["metadata.run_id"] == DST_RUN_ID
    assert params["metadata.offline_run_id"] == SRC_RUN_ID
    assert params["config.training.run_id"] == DST_RUN_ID
    assert params["config.training.offline_run_id"] == SRC_RUN_ID
    assert "config.training.fork_run_id" not in params
    assert tags == {}


def test_run_with_forked_run_tag_maps_parent_and_own_run_id(sync: object) -> None:
    """A run tagged ``forkedRun`` takes the fork branch, even though its own ``metadata.run_id`` matches."""
    params = {"metadata.run_id": SRC_RUN_ID}
    tags = {"forkedRun": "True", "forkedRunId": OFFLINE_PARENT_RUN_ID}

    params, tags = sync._update_params_tags_runs(params, tags, dst_run_id=DST_RUN_ID, src_run_id=SRC_RUN_ID)

    # The parent now points at its synced online run; the offline id is kept alongside.
    assert tags["forkedRunId"] == ONLINE_PARENT_RUN_ID
    assert tags["offline.forkedRunId"] == OFFLINE_PARENT_RUN_ID
    assert params["config.training.fork_run_id"] == ONLINE_PARENT_RUN_ID
    assert params["config.training.offline_fork_run_id"] == OFFLINE_PARENT_RUN_ID
    # The forked run's own id is rewritten too.
    assert params["metadata.run_id"] == DST_RUN_ID
    assert params["metadata.offline_run_id"] == SRC_RUN_ID
