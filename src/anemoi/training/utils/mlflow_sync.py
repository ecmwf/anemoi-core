# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import tempfile
from itertools import starmap
from pathlib import Path


def export_log_output_file_path() -> tempfile._TemporaryFileWrapper:
    # to set up the file where to send the output logs
    if not os.getenv("TMPDIR") and not os.getenv("SCRATCH"):
        error_msg = "Please set one of those variables TMPDIR:{} or SCRATCH:{} to proceed.".format(
            os.environ["SCRATCH"],
            os.environ["TMPDIR"],
        )
        raise ValueError(error_msg)

    tmpdir = os.environ["TMPDIR"] if os.getenv("TMPDIR") else os.environ["SCRATCH"]
    user = os.environ["USER"]
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(dir=tmpdir, prefix=f"{user}_")  # noqa: SIM115
    os.environ["MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE"] = temp.name
    return temp


temp = export_log_output_file_path()

import mlflow  # noqa: E402
from mlflow.entities import RunStatus  # noqa: E402
from mlflow.entities import RunTag  # noqa: E402
from mlflow.tracking.context.default_context import _get_user  # noqa: E402
from mlflow.utils.mlflow_tags import MLFLOW_USER  # noqa: E402
from mlflow.utils.validation import MAX_METRICS_PER_BATCH  # noqa: E402
from mlflow.utils.validation import MAX_PARAMS_TAGS_PER_BATCH  # noqa: E402

try:
    import mlflow_export_import.common.utils as mlflow_utils
    from mlflow_export_import.client.client_utils import create_http_client
    from mlflow_export_import.run.export_run import _get_metrics_with_steps
    from mlflow_export_import.run.export_run import _inputs_to_dict
    from mlflow_export_import.run.import_run import _import_inputs
    from mlflow_export_import.run.run_data_importer import _log_data
    from mlflow_export_import.run.run_data_importer import _log_metrics
    from mlflow_export_import.run.run_data_importer import _log_params
except ImportError:
    msg = "The 'mlflow_export_import' package is not installed. Please install it from https://github.com/mlflow/mlflow-export-import"
    raise ImportError(msg) from None

LOGGER = logging.getLogger(__name__)


# # This functions are based on the existing functions in mlflow_export_import.run.run_data_importer.py
def _log_tags(client: mlflow.MlflowClient, run_dct: dict, run_id: str, batch_size: int, src_user_id: str) -> None:
    def get_data(run_dct: dict, *args) -> list:
        del args  # unused
        tags = run_dct["tags"]
        tags = list(starmap(RunTag, tags.items()))
        user_id = _get_user()
        tags.append(RunTag(MLFLOW_USER, user_id))

        return tags

    def log_data(run_id: str, tags: list) -> None:
        client.log_batch(run_id, tags=tags)

    args_get = {
        "src_user_id": src_user_id,
    }

    _log_data(run_dct, run_id, batch_size, get_data, log_data, args_get)


def import_run_data(mlflow_client: mlflow.MlflowClient, run_dct: dict, run_id: str, src_user_id: str) -> None:
    _log_params(mlflow_client, run_dct, run_id, MAX_PARAMS_TAGS_PER_BATCH)
    _log_metrics(mlflow_client, run_dct, run_id, MAX_METRICS_PER_BATCH)
    _log_tags(
        mlflow_client,
        run_dct,
        run_id,
        MAX_PARAMS_TAGS_PER_BATCH,
        src_user_id,
    )


class MlFlowSync:
    """Class to sync an offline run to the destination tracking uri."""

    def __init__(
        self,
        source_tracking_uri: str,
        dest_tracking_uri: str,
        run_id: str,
        experiment_name: str = "anemoi_debug",
        export_deleted_runs: bool = False,
        log_level: str = "INFO",
    ) -> None:
        self.source_tracking_uri = source_tracking_uri
        self.dest_tracking_uri = dest_tracking_uri
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.export_deleted_runs = export_deleted_runs
        self.log_level = log_level

        LOGGER.setLevel(self.log_level)

    @staticmethod
    def update_run_id(params: dict, key: str, new_run_id: str, offline_run_id: str) -> dict:
        params[f"config.training.{key}"] = new_run_id
        params[f"config.training.offline_{key}"] = offline_run_id

        if key == "run_id":
            params[f"metadata.offline_{key}"] = offline_run_id
            params[f"metadata.{key}"] = new_run_id
        return params

    def update_parent_run_info(self, tags: dict, tag_key: str, tag_dest: str, dst_run_id: str) -> dict:
        mlflow.set_tracking_uri(self.dest_tracking_uri)

        # Check if there is already a parent run in the destination tracking uri
        runs = mlflow.search_runs(
            experiment_ids=mlflow.get_experiment_by_name(self.experiment_name).experiment_id,
            filter_string=f"params.metadata.offline_run_id = '{tags[tag_key]}'",
        )

        if not runs.empty:
            if (runs.shape[0] > 1) and ("tags.resumedRun" in runs.columns):
                new_parent_run_id = runs[runs["tags.resumedRun"] != "True"].run_id.iloc[0]
            else:
                new_parent_run_id = runs.iloc[0].run_id  # read the parent run online run_id
        else:
            new_parent_run_id = dst_run_id
        tags[tag_dest] = tags[tag_key]  # keep offline parent run_id
        tags[tag_key] = new_parent_run_id  # update new online parent run_id
        return tags

    def check_run_is_logged(self, status: str = "FINISHED") -> bool:
        """Blocks sync if top-level parent run or single runs are unavailable."""
        run_logged = False
        if status == "FINISHED":
            mlflow.set_tracking_uri(self.dest_tracking_uri)
            synced_runs = mlflow.search_runs(
                experiment_ids=mlflow.get_experiment_by_name(self.experiment_name).experiment_id,
                filter_string=f"params.metadata.offline_run_id = '{self.run_id}'",
            )
            if not synced_runs.empty:  # single run (no child) already logged
                run_logged = True
        return run_logged

    def sync(
        self,
    ) -> None:
        """Sync an offline run to the destination tracking uri."""
        src_mlflow_client = mlflow.MlflowClient(self.source_tracking_uri)
        dest_mlflow_client = mlflow.MlflowClient(self.dest_tracking_uri)
        http_client = create_http_client(dest_mlflow_client)
        # GET SOURCE RUN ##
        run = src_mlflow_client.get_run(self.run_id)
        run_logged = self.check_run_is_logged(status=run.info.status)
        if run_logged:
            LOGGER.info("Run already imported %s into experiment %s", self.run_id, self.experiment_name)
            return

        if run.info.lifecycle_stage == "deleted" and not self.export_deleted_runs:
            LOGGER.warning(
                "Not exporting run %s because its lifecycle_stage is  %s",
                run.info.run_id,
                run.info.lifecycle_stage,
            )

            return
        msg = {
            "run_id": run.info.run_id,
            "lifecycle_stage": run.info.lifecycle_stage,
            "experiment_id": run.info.experiment_id,
        }
        LOGGER.info("Exporting run: %s", msg)

        run_info = mlflow_utils.strip_underscores(run.info)
        src_user_id = run_info["user_id"]

        exp = dest_mlflow_client.get_experiment_by_name(self.experiment_name)
        dst_run = dest_mlflow_client.create_run(exp.experiment_id)
        dst_run_id = dst_run.info.run_id

        tags = dict(sorted(run.data.tags.items()))

        params = run.data.params
        # So far there is no easy way to force mlflow to use a specific run_id, that means
        # that when we online sync the offline runs those will have run run_ids. To keep
        # track of online and offline governance in that case we update run_ids info

        if (params["config.training.fork_run_id"] == "None") and (params["metadata.run_id"] == run.info.run_id):
            params = self.update_run_id(params, "run_id", new_run_id=dst_run_id, offline_run_id=run.info.run_id)

        elif "forkedRun" in tags:
            try:
                tags = self.update_parent_run_info(
                    tags=tags,
                    tag_key="forkedRunId",
                    tag_dest="offline.forkedRunId",
                    dst_run_id=dst_run_id,
                )
                params = self.update_run_id(
                    params,
                    "fork_run_id",
                    new_run_id=tags["forkedRunId"],
                    offline_run_id=tags["offline.forkedRunId"],
                )
                params = self.update_run_id(params, "run_id", new_run_id=dst_run_id, offline_run_id=run.info.run_id)

            except AttributeError:
                LOGGER.warning("No forked run parent found")

        elif "resumedRun" in tags:
            try:
                tags = self.update_parent_run_info(
                    tags=tags,
                    tag_key="mlflow.parentRunId",
                    tag_dest="mlflow.offline.parentRunId",
                    dst_run_id=dst_run_id,
                )
                params = self.update_run_id(
                    params,
                    "run_id",
                    new_run_id=tags["mlflow.parentRunId"],
                    offline_run_id=tags["mlflow.offline.parentRunId"],
                )

                params["config.training.offline_run_id_folder"] = run.info.run_id

            except AttributeError:
                LOGGER.warning("No parent run found")

        tags["offlineRun"] = "True"

        src_run_dct = {
            "params": run.data.params,
            "metrics": _get_metrics_with_steps(src_mlflow_client, run),
            "tags": tags,
            "inputs": _inputs_to_dict(run.inputs),
        }

        try:
            import_run_data(
                dest_mlflow_client,
                src_run_dct,
                dst_run_id,
                src_user_id,
            )
            _import_inputs(http_client, src_run_dct, dst_run_id)

            path = Path(self.source_tracking_uri, run.info.experiment_id, self.run_id, "artifacts")
            if path.exists():
                mlflow.set_tracking_uri(self.dest_tracking_uri)
                dest_mlflow_client.log_artifacts(dst_run_id, path)
            dest_mlflow_client.set_terminated(dst_run_id, RunStatus.to_string(RunStatus.FINISHED))

        except Exception as e:
            dest_mlflow_client.set_terminated(dst_run_id, RunStatus.to_string(RunStatus.FAILED))
            import traceback

            traceback.print_exc()
            raise Exception(e, "Importing run %s of experiment %s failed", dst_run_id, exp.name) from e  # noqa: TRY002

        LOGGER.info("Imported run %s into experiment %s", dst_run_id, self.experiment_name)

        temp.close()
