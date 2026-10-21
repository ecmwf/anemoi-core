# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml
from mlflow.entities import FileInfo
from mlflow.entities import Metric
from mlflow.entities import Param
from mlflow.entities import Run
from mlflow.entities import RunData
from mlflow.entities import RunInfo
from mlflow.entities import RunStatus
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException

from anemoi.training.diagnostics.mlflow.utils import MAX_PARAMS_LENGTH_PARAM
from anemoi.training.utils.mlflow_incremental_sync import SYNC_TAG
from anemoi.training.utils.mlflow_incremental_sync import MlFlowIncrementalSync

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

RUNNING, FINISHED, FAILED = RunStatus.RUNNING, RunStatus.FINISHED, RunStatus.FAILED


class FakeServer:
    """In-memory stand-in for an MLflow tracking server, with the quirks the sync must handle.

    - Metric values are read back rounded to 15 significant digits, like the ECMWF server.
    - A batch holding a point the server already has is rejected whole, like MLflow's SQL
      store for large batches; ``rejections`` counts it.
    """

    def __init__(self) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict] = {}
        self.rejections = 0
        self.hide_latest_metrics = False  # make every key look new, to force a resend

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        return SimpleNamespace(experiment_id=self.experiments[name]) if name in self.experiments else None

    def create_experiment(self, name: str) -> str:
        self.experiments[name] = str(len(self.experiments) + 1)
        return self.experiments[name]

    def create_run(
        self,
        experiment_id: str,
        start_time: int | None = None,
        tags: dict | None = None,
        run_name: str | None = None,
    ) -> SimpleNamespace:
        run_id = uuid.uuid4().hex
        self.runs[run_id] = {
            "experiment_id": experiment_id,
            "status": "RUNNING",
            "start_time": start_time or 0,
            "end_time": None,
            "run_name": run_name,
            "params": {},
            "tags": dict(tags or {}),
            "metrics": defaultdict(list),
            "artifacts": {},
        }
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def search_runs(self, experiment_ids: list[str], filter_string: str, max_results: int = 1000) -> list[Run]:
        kind, key, value = re.fullmatch(r"(tags|params)\.`(.+)` = '(.+)'", filter_string).groups()
        return [
            self.get_run(run_id)
            for run_id, run in self.runs.items()
            if run["experiment_id"] in experiment_ids and run[kind].get(key) == value
        ][:max_results]

    def get_run(self, run_id: str) -> Run:
        run = self.runs[run_id]
        latest = (
            []
            if self.hide_latest_metrics
            else [max(p, key=lambda m: (m.step, m.timestamp)) for p in run["metrics"].values()]
        )
        info = RunInfo(
            run_id=run_id,
            experiment_id=run["experiment_id"],
            user_id="",
            status=run["status"],
            start_time=run["start_time"],
            end_time=run["end_time"],
            lifecycle_stage="active",
            run_name=run["run_name"],
        )
        data = RunData(
            metrics=[self._rounded(m) for m in latest],
            params=[Param(k, v) for k, v in run["params"].items()],
            tags=[RunTag(k, v) for k, v in run["tags"].items()],
        )
        return Run(info, data)

    @staticmethod
    def _rounded(metric: Metric) -> Metric:
        return Metric(metric.key, float(f"{metric.value:.15g}"), metric.timestamp, metric.step)

    def log_batch(
        self,
        run_id: str,
        metrics: Sequence[Metric] = (),
        params: Sequence[Param] = (),
        tags: Sequence[RunTag] = (),
    ) -> None:
        run = self.runs[run_id]
        identity = {(k, m.timestamp, m.step, m.value) for k, points in run["metrics"].items() for m in points}
        if any((m.key, m.timestamp, m.step, m.value) in identity for m in metrics):
            self.rejections += 1
            msg = "UNIQUE constraint failed: metrics.key, metrics.timestamp, metrics.step, metrics.value"
            raise MlflowException(msg)
        for param in params:
            if run["params"].get(param.key, param.value) != param.value:
                msg = f"Changing param values is not allowed: {param.key}"
                raise MlflowException(msg)
            run["params"][param.key] = param.value
        run["tags"] |= {tag.key: tag.value for tag in tags}
        for metric in metrics:
            run["metrics"][metric.key].append(metric)

    def get_metric_history(self, run_id: str, key: str) -> list[Metric]:
        return [self._rounded(m) for m in self.runs[run_id]["metrics"].get(key, [])]

    def list_artifacts(self, run_id: str, path: str | None = None) -> list[FileInfo]:
        prefix = f"{path}/" if path else ""
        entries = {}
        for name, size in self.runs[run_id]["artifacts"].items():
            if name.startswith(prefix):
                head, _, rest = name[len(prefix) :].partition("/")
                entries[prefix + head] = FileInfo(prefix + head, bool(rest), None if rest else size)
        return list(entries.values())

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        name = f"{artifact_path}/{Path(local_path).name}" if artifact_path else Path(local_path).name
        self.runs[run_id]["artifacts"][name] = Path(local_path).stat().st_size

    def set_terminated(self, run_id: str, status: str, end_time: int | None = None) -> None:
        self.runs[run_id] |= {"status": status, "end_time": end_time}

    # helpers for the tests
    def points(self, run_id: str, key: str) -> list[tuple[int, int]]:
        return sorted((m.timestamp, m.step) for m in self.runs[run_id]["metrics"][key])


def write_run(
    store: Path,
    run_id: str,
    params: dict | None = None,
    tags: dict | None = None,
    metrics: dict | None = None,
    status: int = RUNNING,
    deleted: bool = False,
    with_params: bool = True,
) -> Path:
    """Write (or extend) an offline run the way the MLflow file store lays it out.

    ``with_params=False`` writes no base params, like a run the training has just created.
    """
    path = store / "123" / run_id
    meta = {
        "run_id": run_id,
        "run_name": f"name-{run_id[:4]}",
        "status": status,
        "start_time": 1000,
        "end_time": 2000 if status != RUNNING else None,
        "lifecycle_stage": "deleted" if deleted else "active",
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.yaml").write_text(yaml.safe_dump(meta))
    base_params = {"config.training.run_id": run_id, "config.training.fork_run_id": "None", "metadata.run_id": run_id}
    if not with_params:
        base_params = {}
    for kind, values in (("params", base_params | (params or {})), ("tags", tags or {})):
        for key, value in values.items():
            (path / kind / key).parent.mkdir(parents=True, exist_ok=True)
            (path / kind / key).write_text(value)
    for key, lines in (metrics or {}).items():
        (path / "metrics" / key).parent.mkdir(parents=True, exist_ok=True)
        with (path / "metrics" / key).open("a") as f:
            f.write(lines)
    return path


def lines(*points: tuple[int, float, int]) -> str:
    return "".join(f"{ts} {value!r} {step}\n" for ts, value, step in points)


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "mlflow"


def make_sync(store: Path, server: FakeServer) -> MlFlowIncrementalSync:
    return MlFlowIncrementalSync(store, "http://server", "exp", extra_tags={"sync.user": "me"}, client=server)


PARENT, CHILD, FORK = "a" * 32, "b" * 32, "c" * 32


def test_created_once_and_noop_when_unchanged(store: Path, server: FakeServer, mocker: "MockerFixture") -> None:
    write_run(store, PARENT, params={"lr": "0.1"}, metrics={"loss": lines((1, 1.0, 0), (2, 0.5, 1))})
    sync = make_sync(store, server)

    dst = sync.sync(PARENT)
    assert list(server.runs) == [dst]
    run = server.runs[dst]
    assert run["tags"][SYNC_TAG] == PARENT
    assert run["tags"]["offlineRun"] == "True"
    assert run["tags"]["sync.user"] == "me"
    assert run["params"]["lr"] == "0.1"
    assert run["params"]["config.training.run_id"] == dst
    assert run["params"]["metadata.offline_run_id"] == PARENT
    assert run["run_name"] == f"name-{PARENT[:4]}"
    assert run["status"] == "RUNNING"

    log_batch = mocker.spy(server, "log_batch")
    assert make_sync(store, server).sync(PARENT) == dst
    assert len(server.runs) == 1
    log_batch.assert_not_called()


def test_only_new_points_are_sent(store: Path, server: FakeServer, mocker: "MockerFixture") -> None:
    write_run(store, PARENT, metrics={"loss": lines((1, 1.0, 0), (2, 0.5, 1))})
    dst = make_sync(store, server).sync(PARENT)

    write_run(store, PARENT, metrics={"loss": lines((3, 0.25, 2), (4, 0.125, 3))})
    log_batch = mocker.spy(server, "log_batch")
    make_sync(store, server).sync(PARENT)

    sent = [m for call in log_batch.call_args_list for m in call.kwargs.get("metrics", [])]
    assert [(m.timestamp, m.step) for m in sent] == [(3, 2), (4, 3)]
    assert server.points(dst, "loss") == [(1, 0), (2, 1), (3, 2), (4, 3)]


def test_points_at_the_latest_step_are_compared_by_identity(store: Path, server: FakeServer) -> None:
    # values the server reads back rounded: comparing by value would resend them
    value = 0.12345678901234567
    write_run(store, PARENT, metrics={"lr": lines((1, value, 5), (2, value, 5))})
    dst = make_sync(store, server).sync(PARENT)

    write_run(store, PARENT, metrics={"lr": lines((3, value, 5))})
    make_sync(store, server).sync(PARENT)

    assert server.points(dst, "lr") == [(1, 5), (2, 5), (3, 5)]
    assert server.rejections == 0


def test_torn_line_is_skipped_then_picked_up(store: Path, server: FakeServer) -> None:
    path = write_run(store, PARENT, metrics={"loss": lines((1, 1.0, 0)) + "2 0.5"})
    dst = make_sync(store, server).sync(PARENT)
    assert server.points(dst, "loss") == [(1, 0)]

    with (path / "metrics" / "loss").open("a") as f:
        f.write(" 1\n")
    make_sync(store, server).sync(PARENT)
    assert server.points(dst, "loss") == [(1, 0), (2, 1)]


def test_rejected_batch_is_retried_without_known_points(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT, metrics={"loss": lines((1, 1.0, 0), (2, 0.5, 1))})
    dst = make_sync(store, server).sync(PARENT)

    write_run(store, PARENT, metrics={"loss": lines((3, 0.25, 2))})
    server.hide_latest_metrics = True  # the sync now resends every point
    make_sync(store, server).sync(PARENT)

    assert server.rejections == 1
    assert server.points(dst, "loss") == [(1, 0), (2, 1), (3, 2)]


def test_too_many_params_fails_before_creating_the_run(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT, params={MAX_PARAMS_LENGTH_PARAM: "2", "a": "1", "b": "2"})
    with pytest.raises(ValueError, match=r"Too many params: \d+ > 2\."):
        make_sync(store, server).sync(PARENT)
    assert server.runs == {}


def test_params_length_uses_the_run_limit(store: Path, server: FakeServer, mocker: "MockerFixture") -> None:
    mocker.patch("anemoi.training.diagnostics.mlflow.utils.MAX_PARAMS_LENGTH", 2)
    write_run(store, PARENT, params={MAX_PARAMS_LENGTH_PARAM: "10", "a": "1", "b": "2"})
    assert make_sync(store, server).sync(PARENT) in server.runs


def test_artifacts_missing_or_resized_are_uploaded(store: Path, server: FakeServer, mocker: "MockerFixture") -> None:
    path = write_run(store, PARENT)
    (path / "artifacts" / "plots").mkdir(parents=True)
    (path / "artifacts" / "config.yaml").write_text("a: 1\n")
    (path / "artifacts" / "plots" / "loss.png").write_bytes(b"png")
    (path / "artifacts" / "plots" / ".artifact.uploading.x1y2").write_bytes(b"being written")
    dst = make_sync(store, server).sync(PARENT)
    assert server.runs[dst]["artifacts"] == {"config.yaml": 5, "plots/loss.png": 3}

    log_artifact = mocker.spy(server, "log_artifact")
    make_sync(store, server).sync(PARENT)
    log_artifact.assert_not_called()

    (path / "artifacts" / "plots" / "loss.png").write_bytes(b"bigger png")
    make_sync(store, server).sync(PARENT)
    log_artifact.assert_called_once()
    assert server.runs[dst]["artifacts"]["plots/loss.png"] == 10


def test_resumed_run_syncs_its_parent_first(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT)
    write_run(
        store,
        CHILD,
        params={"metadata.run_id": PARENT, "config.training.run_id": PARENT},
        tags={"resumedRun": "True", "mlflow.parentRunId": PARENT},
    )
    sync = make_sync(store, server)
    child = sync.sync(CHILD)
    parent = sync.find_run(PARENT)

    assert parent is not None
    assert len(server.runs) == 2
    tags, params = server.runs[child]["tags"], server.runs[child]["params"]
    assert tags["mlflow.parentRunId"] == parent
    assert tags["mlflow.offline.parentRunId"] == PARENT
    assert params["config.training.run_id"] == parent
    assert params["config.training.offline_self_run_id"] == CHILD
    assert server.runs[parent]["params"]["config.training.run_id"] == parent


def test_forked_run_points_at_its_source(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT)
    write_run(
        store,
        FORK,
        params={"config.training.fork_run_id": PARENT},
        tags={"forkedRun": "True", "forkedRunId": PARENT},
    )
    sync = make_sync(store, server)
    fork = sync.sync(FORK)
    source = sync.find_run(PARENT)

    tags, params = server.runs[fork]["tags"], server.runs[fork]["params"]
    assert tags["forkedRunId"] == source
    assert tags["offline.forkedRunId"] == PARENT
    assert params["config.training.fork_run_id"] == source
    assert params["config.training.run_id"] == fork


def test_parent_synced_one_shot_is_reused(store: Path, server: FakeServer) -> None:
    # a one-shot sync tags nothing: the parent is found by param, preferring the run that is not resumed
    experiment_id = server.create_experiment("exp")
    one_shot = server.create_run(experiment_id).info.run_id
    server.runs[one_shot]["params"]["metadata.offline_run_id"] = PARENT
    resumed = server.create_run(experiment_id, tags={"resumedRun": "True"}).info.run_id
    server.runs[resumed]["params"]["metadata.offline_run_id"] = PARENT
    write_run(store, PARENT)
    write_run(
        store,
        CHILD,
        params={"metadata.run_id": PARENT},
        tags={"resumedRun": "True", "mlflow.parentRunId": PARENT},
    )

    child = make_sync(store, server).sync(CHILD)

    assert server.runs[child]["tags"]["mlflow.parentRunId"] == one_shot
    assert len(server.runs) == 3  # the parent was not synced again


def test_status(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT, status=RUNNING)
    write_run(
        store,
        CHILD,
        params={"metadata.run_id": PARENT},
        tags={"resumedRun": "True", "mlflow.parentRunId": PARENT},
        status=FAILED,
    )
    write_run(store, FORK, status=FINISHED)
    sync = make_sync(store, server)

    assert server.runs[sync.sync(FORK)]["status"] == "FINISHED"  # offline status once it has ended
    child = sync.sync(CHILD, status="KILLED")  # given status wins
    assert server.runs[child]["status"] == "KILLED"
    assert server.runs[child]["end_time"] == 2000
    assert server.runs[sync.find_run(PARENT)]["status"] == "RUNNING"  # the status applies to the run only
    with pytest.raises(ValueError, match="status"):
        sync.sync(PARENT, status="RUNNING")


def test_deleted_run_is_not_synced(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT, deleted=True)
    assert make_sync(store, server).sync(PARENT) is None
    assert server.runs == {}


def test_run_without_params_is_skipped_until_ready(store: Path, server: FakeServer) -> None:
    # a sync running between the training creating the run and writing its params
    write_run(store, PARENT, with_params=False, metrics={"loss": lines((1, 1.0, 0))})
    assert make_sync(store, server).sync(PARENT) is None
    assert server.runs == {}

    write_run(store, PARENT)
    dst = make_sync(store, server).sync(PARENT)
    assert list(server.runs) == [dst]
    assert server.runs[dst]["params"]["config.training.run_id"] == dst
    assert server.points(dst, "loss") == [(1, 0)]


def test_several_server_runs_for_one_offline_run_fail(store: Path, server: FakeServer) -> None:
    write_run(store, PARENT)
    experiment_id = server.create_experiment("exp")
    server.create_run(experiment_id, tags={SYNC_TAG: PARENT})
    server.create_run(experiment_id, tags={SYNC_TAG: PARENT})
    with pytest.raises(RuntimeError, match="synced to 2 server runs"):
        make_sync(store, server).sync(PARENT)


def test_unknown_run_fails(store: Path, server: FakeServer) -> None:
    store.mkdir()
    with pytest.raises(FileNotFoundError):
        make_sync(store, server).sync(PARENT)


def test_module_does_not_need_mlflow_export_import() -> None:
    code = (
        "import sys; import anemoi.training.utils.mlflow_incremental_sync; "
        "assert 'mlflow_export_import' not in sys.modules; "
        "assert 'anemoi.training.utils.mlflow_sync' not in sys.modules"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "SCRATCH")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)  # noqa: S603
