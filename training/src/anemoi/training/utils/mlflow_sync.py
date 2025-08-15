# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import shutil
import tempfile
from itertools import starmap
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow.entities

from anemoi.training.diagnostics.mlflow import MAX_PARAMS_LENGTH
from anemoi.training.diagnostics.mlflow.utils import clean_config_params


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
    os.environ["MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY"] = tmpdir
    return temp


def close_and_clean_temp(server2server: str, artifact_path: Path) -> None:
    temp.close()
    os.environ.pop("MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE")
    os.environ.pop("MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY")
    if server2server:
        shutil.rmtree(artifact_path)


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

except ImportError:
    msg = "The 'mlflow-export-import' package is not installed. Please install it from https://github.com/mlflow/mlflow-export-import"
    raise ImportError(msg) from None

LOGGER = logging.getLogger(__name__)


# # This functions are based on the existing functions in mlflow_export_import.run.run_data_importer.py
def _log_tags(
    client: mlflow.MlflowClient,
    run_dct: dict,
    run_id: str,
    batch_size: int,
    src_user_id: str,
) -> None:
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


def _log_params(client: mlflow.MlflowClient, run_dct: dict[str, Any], run_id: str, batch_size: int) -> None:
    def get_data(run_dct: dict[str, Any], args: Any = None) -> list[mlflow.entities.Param]:  # noqa: ARG001
        cleaned_run_dct = clean_config_params(run_dct["params"])
        LOGGER.info("Logging %s parameters", len(cleaned_run_dct))
        if len(cleaned_run_dct) > MAX_PARAMS_LENGTH:
            msg = (
                f"Too many params: {len(cleaned_run_dct)} > {MAX_PARAMS_LENGTH}",
                "Please revisit the fields being logged and add redundant or irrelevant "
                "ones to the clean_config_params function.",
            )
            raise ValueError(msg)
        return [mlflow.entities.Param(k, v) for k, v in cleaned_run_dct.items()]

    def log_data(run_id: str, params: dict[str, Any]) -> None:
        client.log_batch(run_id, params=params)

    _log_data(run_dct, run_id, batch_size, get_data, log_data)


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
    def update_run_id(params: dict, key: str, new_run_id: str, src_run_id: str, run_type: str) -> dict:
        params[f"config.training.{key}"] = new_run_id
        params[f"config.training.{run_type}_{key}"] = src_run_id

        if key == "run_id":
            params[f"metadata.{run_type}_{key}"] = src_run_id
            params[f"metadata.{key}"] = new_run_id
        return params

    def update_parent_run_info(self, tags: dict, tag_key: str, tag_dest: str, dst_run_id: str, run_type: str) -> dict:
        mlflow.set_tracking_uri(self.dest_tracking_uri)

        # Check if there is already a parent run in the destination tracking uri
        runs = mlflow.search_runs(
            experiment_ids=mlflow.get_experiment_by_name(self.experiment_name).experiment_id,
            filter_string=f"params.metadata.{run_type}_run_id = '{tags[tag_key]}'",
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

    def check_run_is_logged(self, status: str = "FINISHED", server2server: bool = False) -> bool:
        """Blocks sync if top-level parent run or single runs are unavailable."""
        run_logged = False
        if status == "FINISHED":
            mlflow.set_tracking_uri(self.dest_tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            run_type = "server2server" if server2server else "offline"
            if experiment:
                synced_runs = mlflow.search_runs(
                    experiment_ids=experiment.experiment_id,
                    filter_string=f"params.metadata.{run_type}_run_id = '{self.run_id}'",
                )
                if not synced_runs.empty:  # single run (no child) already logged
                    run_logged = True
        return run_logged

    def _check_source_tracking_uri(self) -> bool:
        parsed_url = urlparse(self.source_tracking_uri)
        return all([parsed_url.scheme, parsed_url.netloc])  # True if source_tracking_uri is a remote server

    def _get_dst_experiment_id(self, dest_mlflow_client: str) -> str:
        experiment = dest_mlflow_client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return dest_mlflow_client.create_experiment(self.experiment_name)
        return experiment.experiment_id

    def _get_artifacts_path(self, server2server: str, run: mlflow.entities.Run) -> Path:
        if server2server:
            # Download each artifact
            temp_dir = os.getenv("MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY")
            artifact_path = Path(temp_dir, run.info.run_id)
            artifact_path.mkdir(parents=True, exist_ok=True)
        else:
            artifact_path = Path(
                self.source_tracking_uri,
                run.info.experiment_id,
                run.info.run_id,
                "artifacts",
            )

        return artifact_path

    def _download_artifacts(
        self,
        client: mlflow.tracking.client.MlflowClient,
        run_id: mlflow.entities.Run,
        artifact_path: Path,
    ) -> None:

        mlflow.set_tracking_uri(self.source_tracking_uri)  # OTHERWISE IT WILL NOT WORK
        artifacts = client.list_artifacts(run_id)
        LOGGER.info(
            "Downloading artifacts %s for run %s to %s",
            len(artifacts),
            run_id,
            artifact_path,
        )
        for artifact in artifacts:
            # Download artifact file from the server
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact.path, dst_path=artifact_path)

    def _update_params_tags_runs(
        self,
        params: dict,
        tags: dict,
        dst_run_id: str,
        src_run_id: str,
        run_type: str = "offline",
    ) -> (dict, dict):

        if (params["config.training.fork_run_id"] == "None") and (params["metadata.run_id"] == src_run_id):
            params = self.update_run_id(
                params,
                "run_id",
                new_run_id=dst_run_id,
                src_run_id=src_run_id,
                run_type=run_type,
            )

        elif "forkedRun" in tags:
            try:
                tags = self.update_parent_run_info(
                    tags=tags,
                    tag_key="forkedRunId",
                    tag_dest=f"{run_type}.forkedRunId",
                    dst_run_id=dst_run_id,
                    run_type=run_type,
                )
                params = self.update_run_id(
                    params,
                    "fork_run_id",
                    new_run_id=tags["forkedRunId"],
                    src_run_id=tags[f"{run_type}.forkedRunId"],
                    run_type=run_type,
                )
                params = self.update_run_id(
                    params,
                    "run_id",
                    new_run_id=dst_run_id,
                    src_run_id=src_run_id,
                    run_type=run_type,
                )

            except AttributeError:
                LOGGER.warning("No forked run parent found")

        elif "resumedRun" in tags:
            try:
                tags = self.update_parent_run_info(
                    tags=tags,
                    tag_key="mlflow.parentRunId",
                    tag_dest=f"mlflow.{run_type}.parentRunId",
                    dst_run_id=dst_run_id,
                    run_type=run_type,
                )
                params = self.update_run_id(
                    params,
                    "run_id",
                    new_run_id=tags["mlflow.parentRunId"],
                    src_run_id=tags[f"mlflow.{run_type}.parentRunId"],
                    run_type=run_type,
                )

                # in the offline case that's the local folder name for the resumed run
                # in the server2server case that's the source server run_id of the resumed run
                params[f"config.training.{run_type}_self_run_id"] = src_run_id

            except AttributeError:
                LOGGER.warning("No parent run found")

        return params, tags

    def sync(
        self,
    ) -> None:
        """Sync an offline run to the destination tracking uri."""
        src_mlflow_client = mlflow.MlflowClient(self.source_tracking_uri)
        dest_mlflow_client = mlflow.MlflowClient(self.dest_tracking_uri)
        http_client = create_http_client(dest_mlflow_client)
        # GET SOURCE RUN ##
        run = src_mlflow_client.get_run(self.run_id)
        server2server = self._check_source_tracking_uri()
        run_logged = self.check_run_is_logged(status=run.info.status)
        if run_logged:
            LOGGER.info(
                "Run already imported %s into experiment %s",
                self.run_id,
                self.experiment_name,
            )
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

        exp_id = self._get_dst_experiment_id(dest_mlflow_client=dest_mlflow_client)
        dst_run = dest_mlflow_client.create_run(exp_id)
        dst_run_id = dst_run.info.run_id

        tags = dict(sorted(run.data.tags.items()))

        params = run.data.params
        # So far there is no easy way to force mlflow to use a specific run_id, that means
        # that when we online sync the offline runs those will have different run_ids. To keep
        # track of online and offline governance in that case we update run_ids info

        artifact_path = self._get_artifacts_path(server2server, run)

        if server2server:
            tags["server2server"] = "True"
            self._download_artifacts(src_mlflow_client, run.info.run_id, artifact_path)
            params, tags = self._update_params_tags_runs(
                params,
                tags,
                dst_run_id,
                run.info.run_id,
                run_type="server2server",
            )

        else:
            tags["offlineRun"] = "True"
            params, tags = self._update_params_tags_runs(params, tags, dst_run_id, run.info.run_id, run_type="offline")

        src_run_dct = {
            "params": params,
            "metrics": _get_metrics_with_steps(src_mlflow_client, run),
            "tags": tags,
            "inputs": _inputs_to_dict(run.inputs),
        }

        try:
            LOGGER.info("Starting to export run data")

            import_run_data(
                dest_mlflow_client,
                src_run_dct,
                dst_run_id,
                src_user_id,
            )
            _import_inputs(http_client, src_run_dct, dst_run_id)

            mlflow.set_tracking_uri(self.dest_tracking_uri)
            dest_mlflow_client.log_artifacts(dst_run_id, artifact_path)
            dest_mlflow_client.set_terminated(dst_run_id, RunStatus.to_string(RunStatus.FINISHED))

        except BaseException:
            dest_mlflow_client.set_terminated(dst_run_id, RunStatus.to_string(RunStatus.FAILED))
            import traceback

            traceback.print_exc()
            LOGGER.exception(
                "Importing run %s of experiment %s failed",
                dst_run_id,
                self.experiment_name,
            )

        finally:
            close_and_clean_temp(server2server, artifact_path)

        LOGGER.info("Imported run %s into experiment %s", dst_run_id, self.experiment_name)
