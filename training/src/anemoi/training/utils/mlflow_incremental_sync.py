# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Incremental sync of an offline MLflow run to a tracking server.

Unlike :class:`anemoi.training.utils.mlflow_sync.MlFlowSync`, which copies a finished run
into a new server run on every call, every call here updates the *same* server run and only
sends what the server does not have yet. It is safe to run any number of times, including
while the training is still writing the offline store:

- the server run is found by the tag ``sync.offline_run_id`` and created once;
- params: missing keys only (server params are immutable);
- tags: changed values only;
- metrics: points after the server's latest step per key;
- artifacts: missing files, or files whose size changed;
- status: left ``RUNNING`` until the offline run ends, or until a status is given.

The run ids stored in params and tags are rewritten the same way as the one-shot sync
(:func:`anemoi.training.diagnostics.mlflow.utils.rewrite_lineage`), so the server runs look
alike and resuming or forking from them works the same.

A run must be synced either one-shot or incrementally, not both: the two modes find their
server run differently and would create one each. A resumed or forked run may point at a
related run synced either way.

This module must not import :mod:`anemoi.training.utils.mlflow_sync`, which requires
``mlflow-export-import`` and ``TMPDIR``/``SCRATCH`` at import time.
"""

from __future__ import annotations

import getpass
import logging
import time
from pathlib import Path

import mlflow
import yaml
from mlflow.entities import Metric
from mlflow.entities import Param
from mlflow.entities import RunStatus
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.validation import MAX_PARAMS_TAGS_PER_BATCH

from anemoi.training.diagnostics.mlflow.utils import check_params_length
from anemoi.training.diagnostics.mlflow.utils import clean_config_params
from anemoi.training.diagnostics.mlflow.utils import get_max_params_length
from anemoi.training.diagnostics.mlflow.utils import rewrite_lineage

LOGGER = logging.getLogger(__name__)

SYNC_TAG = "sync.offline_run_id"
TERMINAL_STATUSES = ("FINISHED", "FAILED", "KILLED")
MAX_LINEAGE_DEPTH = 10
#: Params read by ``rewrite_lineage``; a run is synced only once they are written.
LINEAGE_PARAMS = frozenset({"config.training.fork_run_id", "metadata.run_id"})


def _read_tree(root: Path) -> dict[str, str]:
    """Read a ``params/`` or ``tags/`` directory: one file per key, ``/`` in a key is a subdirectory."""
    if not root.is_dir():
        return {}
    return {path.relative_to(root).as_posix(): path.read_text() for path in sorted(root.rglob("*")) if path.is_file()}


def _read_metric_file(path: Path) -> tuple[list[tuple[int, float, int]], bool]:
    """Read the complete ``timestamp value step`` lines of a metric file.

    The training may be appending to the file while it is read: an unterminated last line is
    skipped (the second item of the result is then True) and picked up by the next sync.
    """
    data = path.read_bytes()
    torn = bool(data) and not data.endswith(b"\n")
    if torn:
        data = data[: data.rfind(b"\n") + 1]
    points = []
    for line in data.decode().splitlines():
        parts = line.split()
        if len(parts) < 2:
            msg = f"{path}: malformed metric line {line!r}"
            raise ValueError(msg)
        step = int(parts[2]) if len(parts) > 2 else 0
        points.append((int(parts[0]), float(parts[1]), step))
    return points, torn


class OfflineRun:
    """A run of an offline MLflow store, read as plain files.

    The MLflow file store is not used: it fails on a metric line being written, and recent
    MLflow versions refuse to open it by default.
    """

    def __init__(self, store: Path, run_id: str) -> None:
        meta_files = list(store.glob(f"*/{run_id}/meta.yaml"))
        if len(meta_files) != 1:
            msg = f"Offline run {run_id} not found (or found more than once) in {store}"
            raise FileNotFoundError(msg)
        self.run_id = run_id
        self.path = meta_files[0].parent
        self.meta = yaml.safe_load(meta_files[0].read_text()) or {}
        self.params = _read_tree(self.path / "params")
        self.tags = _read_tree(self.path / "tags")

    @staticmethod
    def exists(store: Path, run_id: str) -> bool:
        return any(store.glob(f"*/{run_id}/meta.yaml"))

    @property
    def status(self) -> str:
        return RunStatus.to_string(int(self.meta.get("status", RunStatus.RUNNING)))

    def metrics(self) -> tuple[dict[str, list[tuple[int, float, int]]], int]:
        """Metric points ``(timestamp, value, step)`` per key, and the number of torn lines skipped."""
        metrics, torn = {}, 0
        metrics_dir = self.path / "metrics"
        if metrics_dir.is_dir():
            for path in sorted(metrics_dir.rglob("*")):
                if path.is_file():
                    points, skipped = _read_metric_file(path)
                    metrics[path.relative_to(metrics_dir).as_posix()] = points
                    torn += skipped
        return metrics, torn

    def artifacts(self) -> dict[str, Path]:
        """Artifact files by name, without the ones MLflow is still writing.

        Recent MLflow versions write an artifact to ``.artifact.uploading.*`` and rename it
        once complete.
        """
        artifacts_dir = self.path / "artifacts"
        if not artifacts_dir.is_dir():
            return {}
        return {
            path.relative_to(artifacts_dir).as_posix(): path
            for path in artifacts_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".artifact.uploading")
        }


class MlFlowIncrementalSync:
    """Sync offline runs to a tracking server incrementally.

    Parameters
    ----------
    source : str | Path
        The offline MLflow store (``<outdir>/logs/mlflow``).
    destination : str
        The destination tracking URI.
    experiment_name : str, optional
        The destination experiment, created if needed, by default ``anemoi-debug``.
    authentication : bool, optional
        The destination server requires authentication, by default False.
    extra_tags : dict | None, optional
        Tags added to every synced run, by default None.
    client : mlflow.MlflowClient | None, optional
        The destination client, by default an ``AnemoiMlflowClient``.
    """

    def __init__(
        self,
        source: str | Path,
        destination: str,
        experiment_name: str = "anemoi-debug",
        authentication: bool = False,
        extra_tags: dict | None = None,
        client: mlflow.MlflowClient | None = None,
    ) -> None:
        self.store = Path(str(source).removeprefix("file://"))
        self.experiment_name = experiment_name
        self.extra_tags = extra_tags or {}
        # Artifact uploads to ``mlflow-artifacts:`` URIs resolve the global tracking URI,
        # not the client's.
        mlflow.set_tracking_uri(destination)
        if client is None:
            from anemoi.utils.mlflow.client import AnemoiMlflowClient

            # re-authenticates before every call, so the token stays valid on long syncs
            client = AnemoiMlflowClient(destination, authentication=authentication, check_health=False)
        self.client = client
        self._experiment_id: str | None = None
        self._synced: dict[str, str] = {}  # offline run id -> server run id

    @property
    def experiment_id(self) -> str:
        if self._experiment_id is None:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            self._experiment_id = (
                experiment.experiment_id if experiment else self.client.create_experiment(self.experiment_name)
            )
        return self._experiment_id

    def sync(self, run_id: str, status: str | None = None) -> str | None:
        """Sync the offline run ``run_id``.

        Parameters
        ----------
        run_id : str
            The offline run id.
        status : str | None, optional
            The final status to set on the server run (``FINISHED``, ``FAILED`` or ``KILLED``).
            By default the server run takes the offline status once the offline run has ended,
            and stays ``RUNNING`` until then. Applies to ``run_id`` only, not to the related
            runs synced with it.

        Returns
        -------
        str | None
            The server run id, or None when the run was not synced: it is deleted, or its
            params are not written yet (the training is starting).
        """
        if status is not None and status not in TERMINAL_STATUSES:
            msg = f"status must be one of {TERMINAL_STATUSES}, got {status!r}"
            raise ValueError(msg)
        self._synced.clear()
        return self._sync(run_id, status, chain=())

    def find_run(self, run_id: str) -> str | None:
        """Server run id of an offline run synced incrementally, or None."""
        runs = self.client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.`{SYNC_TAG}` = '{run_id}'",
            max_results=10,
        )
        if len(runs) > 1:
            msg = f"Offline run {run_id} is synced to {len(runs)} server runs: {[r.info.run_id for r in runs]}"
            raise RuntimeError(msg)
        return runs[0].info.run_id if runs else None

    def _sync(self, run_id: str, status: str | None, chain: tuple[str, ...]) -> str | None:
        if run_id in chain or len(chain) > MAX_LINEAGE_DEPTH:
            msg = f"Cannot sync {run_id}: lineage cycle or chain too deep ({' -> '.join(chain)})"
            raise RuntimeError(msg)
        if run_id in self._synced:
            return self._synced[run_id]

        run = OfflineRun(self.store, run_id)
        if run.meta.get("lifecycle_stage") == "deleted":
            LOGGER.warning("Not syncing run %s: it is deleted", run_id)
            return None
        # The training writes the params just after creating the run: a sync running in
        # between finds none of the run-id params the lineage rewrite needs. Skip until then.
        if not run.params.keys() >= LINEAGE_PARAMS:
            LOGGER.info("Offline run %s has no run-id params yet (training starting), not syncing it yet", run_id)
            return None

        dst_run_id = self.find_run(run_id)
        created = dst_run_id is None
        if created:
            self._check_params_length(run.params)
            dst_run_id = self.client.create_run(
                self.experiment_id,
                start_time=run.meta.get("start_time"),
                tags={SYNC_TAG: run_id},  # set at creation: a failed sync never leaves an untagged run
                run_name=run.meta.get("run_name"),
            ).info.run_id
            LOGGER.info("Created server run %s for offline run %s", dst_run_id, run_id)
        self._synced[run_id] = dst_run_id

        params, tags = rewrite_lineage(
            dict(run.params),
            dict(run.tags),
            dst_run_id=dst_run_id,
            src_run_id=run_id,
            resolve=lambda related_run_id: self._resolve(related_run_id, (*chain, run_id)),
        )
        params = clean_config_params(params)
        tags |= {"offlineRun": "True", SYNC_TAG: run_id, MLFLOW_USER: getpass.getuser()} | self.extra_tags

        server_run = self.client.get_run(dst_run_id)
        counts = {
            "params": self._log_params(dst_run_id, params, server_run.data.params),
            "tags": self._log_tags(dst_run_id, tags, server_run.data.tags),
            "metrics": self._log_metrics(dst_run_id, run, server_run),
            "artifacts": self._log_artifacts(dst_run_id, run),
        }

        final_status = status or (run.status if run.status in TERMINAL_STATUSES else None)
        if final_status and server_run.info.status != final_status:
            end_time = run.meta.get("end_time") or int(time.time() * 1000)
            self.client.set_terminated(dst_run_id, final_status, end_time=end_time)
            counts["status"] = final_status

        LOGGER.info(
            "Synced offline run %s to server run %s: %s",
            run_id,
            dst_run_id,
            ", ".join(f"{key} {value}" for key, value in counts.items()),
        )
        return dst_run_id

    def _resolve(self, related_run_id: str, chain: tuple[str, ...]) -> str | None:
        """Server run id of a related run: the parent of a resumed run, or the source of a forked run.

        The related run is looked up in this order:

        1. the ``sync.offline_run_id`` tag: it was synced incrementally;
        2. the ``metadata.offline_run_id`` param, the way the one-shot sync looks it up
           (``MlFlowSync._find_dst_run_id``): it was synced one-shot. A resumed run carries
           its parent's id in this param, so the run that is not resumed is preferred;
        3. the offline store: it is not on the server yet, so it is synced incrementally first
           and the run can point at it.

        Returns None when the related run is found nowhere; ``rewrite_lineage`` then uses the
        run's own server id, like the one-shot sync.
        """
        dst_run_id = self._synced.get(related_run_id) or self.find_run(related_run_id)
        if dst_run_id:
            return dst_run_id

        runs = self.client.search_runs(
            [self.experiment_id],
            filter_string=f"params.`metadata.offline_run_id` = '{related_run_id}'",
        )
        if runs:
            not_resumed = [r for r in runs if r.data.tags.get("resumedRun") != "True"]
            return (not_resumed or runs)[0].info.run_id

        if OfflineRun.exists(self.store, related_run_id):
            LOGGER.info("Related run %s is not on the server yet, syncing it first", related_run_id)
            return self._sync(related_run_id, status=None, chain=chain)

        LOGGER.warning("Related run %s is neither on the server nor in %s", related_run_id, self.store)
        return None

    @staticmethod
    def _check_params_length(params: dict[str, str]) -> None:
        # The run's own limit, as for the one-shot sync: a run trained with a raised
        # `max_params_length` must sync with it too.
        check_params_length(len(clean_config_params(dict(params))), get_max_params_length(params))

    def _log_params(self, dst_run_id: str, params: dict[str, str], server_params: dict[str, str]) -> int:
        new_params = [Param(key, value) for key, value in params.items() if key not in server_params]
        changed = [key for key, value in params.items() if key in server_params and server_params[key] != value]
        if changed:
            LOGGER.warning("%d params differ on the server and are left unchanged: %s", len(changed), changed[:5])
        for i in range(0, len(new_params), MAX_PARAMS_TAGS_PER_BATCH):
            self.client.log_batch(dst_run_id, params=new_params[i : i + MAX_PARAMS_TAGS_PER_BATCH])
        return len(new_params)

    def _log_tags(self, dst_run_id: str, tags: dict[str, str], server_tags: dict[str, str]) -> int:
        new_tags = [RunTag(key, value) for key, value in tags.items() if server_tags.get(key) != value]
        for i in range(0, len(new_tags), MAX_PARAMS_TAGS_PER_BATCH):
            self.client.log_batch(dst_run_id, tags=new_tags[i : i + MAX_PARAMS_TAGS_PER_BATCH])
        return len(new_tags)

    def _server_points(self, dst_run_id: str, key: str, step: int | None = None) -> set[tuple[int, int]]:
        """Identities ``(timestamp, step)`` of the server's points for ``key``.

        Points are never identified by value: a server may return values rounded (the ECMWF
        server returns 15 significant digits), so a value read back can differ from the one sent.
        """
        history = self.client.get_metric_history(dst_run_id, key)
        return {(m.timestamp, m.step) for m in history if step is None or m.step == step}

    def _log_metrics(self, dst_run_id: str, run: OfflineRun, server_run: mlflow.entities.Run) -> int:
        # Points are never sent twice: some servers reject a whole batch when it holds a point
        # they already have (MLflow's SQL store, for batches above 100 points per key). So only
        # the points after the server's latest step per key are sent, and the points at that
        # step are compared with the server's. Each key is sent in step order, so the server
        # always holds a prefix of it, even after a failed sync.
        # the latest point per key, with its step (``data.metrics`` only holds the values)
        latest_step = {m.key: m.step for m in server_run.data._metric_objs}
        offline_metrics, torn = run.metrics()
        if torn:
            LOGGER.debug("Skipped %d metric lines being written", torn)
        new_metrics = []
        for key, points in offline_metrics.items():
            step = latest_step.get(key)
            known = self._server_points(dst_run_id, key, step) if step is not None else set()
            new_metrics.extend(
                Metric(key, value, timestamp, point_step)
                for timestamp, value, point_step in sorted(points, key=lambda p: (p[2], p[0]))
                if step is None or point_step > step or (point_step == step and (timestamp, step) not in known)
            )
        for i in range(0, len(new_metrics), MAX_METRICS_PER_BATCH):
            self._log_metric_batch(dst_run_id, new_metrics[i : i + MAX_METRICS_PER_BATCH])
        return len(new_metrics)

    def _log_metric_batch(self, dst_run_id: str, batch: list[Metric]) -> None:
        """Log a batch of points; if the server rejects one it already has, drop those and retry once."""
        try:
            self.client.log_batch(dst_run_id, metrics=batch)
        except MlflowException as error:
            if "UNIQUE" not in str(error) and "duplicate key" not in str(error):
                raise
            known = {key: self._server_points(dst_run_id, key) for key in {m.key for m in batch}}
            new_points = [m for m in batch if (m.timestamp, m.step) not in known[m.key]]
            LOGGER.warning("Server already has some points, retrying with %d of %d", len(new_points), len(batch))
            if new_points:
                self.client.log_batch(dst_run_id, metrics=new_points)

    def _list_server_artifacts(self, dst_run_id: str, path: str | None = None) -> dict[str, int]:
        sizes = {}
        for info in self.client.list_artifacts(dst_run_id, path):
            if info.is_dir:
                sizes |= self._list_server_artifacts(dst_run_id, info.path)
            else:
                sizes[info.path] = info.file_size
        return sizes

    def _log_artifacts(self, dst_run_id: str, run: OfflineRun) -> int:
        server_sizes = self._list_server_artifacts(dst_run_id)
        uploaded = 0
        for name, path in sorted(run.artifacts().items()):
            try:
                if server_sizes.get(name) == path.stat().st_size:
                    continue
                self.client.log_artifact(dst_run_id, str(path), str(Path(name).parent) if "/" in name else None)
                uploaded += 1
            except FileNotFoundError:
                # removed or replaced by the training since it was listed: the next sync gets it
                LOGGER.debug("Artifact %s disappeared, skipped", name)
        return uploaded
