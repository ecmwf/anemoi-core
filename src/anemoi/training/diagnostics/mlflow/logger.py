# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import io
import logging
import os
import re
import sys
import time
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from weakref import WeakValueDictionary

from packaging.version import Version
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.loggers.mlflow import _convert_params
from pytorch_lightning.loggers.mlflow import _flatten_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.diagnostics.mlflow.auth import TokenAuth
from anemoi.training.diagnostics.mlflow.utils import health_check
from anemoi.training.utils.jsonify import map_config_to_primitives

if TYPE_CHECKING:
    from argparse import Namespace

    import mlflow

LOGGER = logging.getLogger(__name__)


class LogsMonitor:
    """Class for logging terminal output.

    Inspired by the class for logging terminal output in aim.
    Aim-Code: https://github.com/aimhubio/aim/blob/94646d2d317ec7a43303a16530f7963e4e652921/aim/ext/resource/tracker.py#L20

    Note: If there is an error, the terminal output logging ends before the error message is printed into the log file.
    In order for the user to see the error message, the user must look at the slurm output file.
    We provide the SLURM job id in the very beginning of the log file and print the final status of the run in the end.

    Parameters
    ----------
    artifact_save_dir : str
        Directory to save the terminal logs.
    experiment : MLflow experiment object.
        MLflow experiment object.
    run_id: str
        MLflow run ID.
    log_time_interval : int
        Interval (in seconds) at which to write buffered terminal outputs, default 30

    """

    _buffer_registry = WeakValueDictionary()
    _old_out_write = None
    _old_err_write = None

    def __init__(
        self,
        artifact_save_dir: str | Path,
        experiment: MLFlowLogger.experiment,
        run_id: str,
        log_time_interval: float = 30.0,
    ) -> None:
        """Initialize the LogsMonitor.

        Parameters
        ----------
        artifact_save_dir : str | Path
            Directory for artifact saves.
        experiment : MLFlowLogger.experiment
            Experiment from MLFlow
        run_id : str
            Run ID
        log_time_interval : float, optional
            Logging time interval in seconds, by default 30.0

        """
        # active run
        self.experiment = experiment
        self.run_id = run_id

        # terminal log capturing
        self._log_capture_interval = 1
        self._log_time_interval = log_time_interval
        self._old_out = None
        self._old_err = None
        self._io_buffer = io.BytesIO()

        # Start thread to collect stats and logs at intervals
        self._th_collector = Thread(target=self._log_collector, daemon=True)
        self._shutdown = False
        self._started = False

        # open your files here
        self.artifact_save_dir = artifact_save_dir
        self.file_save_path = Path(artifact_save_dir, "terminal_log.txt")
        self.file_save_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _install_stream_patches(cls) -> None:
        cls._old_out_write = sys.stdout.write
        cls._old_err_write = sys.stderr.write

        def new_out_write(data: str | bytes) -> None:
            # out to buffer
            cls._old_out_write(data)
            if isinstance(data, str):
                data = data.encode()
            for buffer in cls._buffer_registry.values():
                buffer.write(data)

        def new_err_write(data: str | bytes) -> None:
            # err to buffer
            cls._old_err_write(data)
            if isinstance(data, str):
                data = data.encode()
            for buffer in cls._buffer_registry.values():
                buffer.write(data)

        sys.stdout.write = new_out_write
        sys.stderr.write = new_err_write

    @classmethod
    def _uninstall_stream_patches(cls) -> None:
        sys.stdout.write = cls._old_out_write
        sys.stderr.write = cls._old_err_write

    def start(self) -> None:
        """Start collection."""
        if self._started:
            return
        self._started = True
        # install the stream patches if not done yet
        if not self._buffer_registry:
            self._install_stream_patches()
        self._buffer_registry[id(self)] = self._io_buffer
        # Start thread to asynchronously collect logs
        self._th_collector.start()
        LOGGER.info("Terminal Log Path: %s", self.file_save_path)
        if os.getenv("SLURM_JOB_ID"):
            LOGGER.info("SLURM job id: %s", os.getenv("SLURM_JOB_ID"))

    def finish(self, status: str) -> None:
        """Stop the monitoring and close the log file."""
        if not self._started:
            return
        LOGGER.info(
            ("Stopping terminal log monitoring and saving buffered terminal outputs. Final status: %s"),
            status.upper(),
        )
        self._shutdown = True
        # read and store remaining buffered logs
        self._store_buffered_logs()
        # unregister the buffer
        del self._buffer_registry[id(self)]
        # uninstall stream patching if no buffer is left in the registry
        if not self._buffer_registry:
            self._uninstall_stream_patches()

        with self.file_save_path.open("a") as logfile:
            logfile.write("\n\n")
            logfile.flush()
            logfile.close()

    def _log_collector(self) -> None:
        """Log collecting thread body.

        Main monitoring loop, which consistently collect and log outputs.
        """
        log_capture_time_counter = 0

        while True:
            if self._shutdown:
                break

            time.sleep(self._log_time_interval)  # in seconds
            log_capture_time_counter += self._log_time_interval

            if log_capture_time_counter > self._log_capture_interval:
                self._store_buffered_logs()
                log_capture_time_counter = 0

    def _store_buffered_logs(self) -> None:
        _buffer_size = self._io_buffer.tell()
        if not _buffer_size:
            return
        self._io_buffer.seek(0)
        # read and reset the buffer
        data = self._io_buffer.read(_buffer_size)
        self._io_buffer.seek(0)
        # handle the buffered data and store
        # split lines and keep \n at the end of each line
        lines = [e + b"\n" for e in data.split(b"\n") if e]

        ansi_csi_re = re.compile(b"\001?\033\\[((?:\\d|;)*)([a-dA-D])\002?")

        def _handle_csi(line: bytes) -> bytes:
            # removes the cursor up and down symbols from the line
            # skip tqdm status bar updates ending with "curser up" but the last one in buffer to save space
            def _remove_csi(line: bytes) -> bytes:
                # replacing the leftmost non-overlapping occurrences of
                # pattern ansi_csi_re in string line by the replacement ""
                return re.sub(ansi_csi_re, b"", line)

            for match in ansi_csi_re.finditer(line):
                arg, command = match.groups()
                arg = int(arg.decode()) if arg else 1
                if command == b"A" and (
                    b"0%" not in line and not self._shutdown and b"[INFO]" not in line and b"[DEBUG]" not in line
                ):  # cursor up
                    # only keep x*10% status updates from tqmd status bars that end with a cursor up
                    # always keep shutdown commands
                    # always keep logger info and debug prints
                    line = b""
            return _remove_csi(line)

        line = None
        with self.file_save_path.open("a") as logfile:
            for line in lines:
                # handle cursor up and down symbols
                cleaned_line = _handle_csi(line)
                # handle each line for carriage returns
                cleaned_line = cleaned_line.rsplit(b"\r")[-1]
                logfile.write(cleaned_line.decode())

            logfile.flush()
        self.experiment.log_artifact(self.run_id, str(self.file_save_path))


class AnemoiMLflowLogger(MLFlowLogger):
    """A custom MLflow logger that logs terminal output."""

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        project_name: str = "anemoi",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        save_dir: str | None = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        resumed: bool | None = False,
        forked: bool | None = False,
        run_id: str | None = None,
        fork_run_id: str | None = None,
        offline: bool | None = False,
        authentication: bool | None = None,
        log_hyperparams: bool | None = True,
        on_resume_create_child: bool | None = True,
    ) -> None:
        """Initialize the AnemoiMLflowLogger.

        Parameters
        ----------
        experiment_name : str, optional
            Name of experiment, by default "lightning_logs"
        project_name : str, optional
            Name of the project, by default "anemoi"
        run_name : str | None, optional
            Name of run, by default None
        tracking_uri : str | None, optional
            Tracking URI of server, by default os.getenv("MLFLOW_TRACKING_URI")
        save_dir : str | None, optional
            Directory to save logs to, by default "./mlruns"
        log_model : Literal[True, False, "all"], optional
            Log model checkpoints to server (expensive), by default False
        prefix : str, optional
            Prefix for experiments, by default ""
        resumed : bool | None, optional
            Whether the run was resumed or not, by default False
        forked : bool | None, optional
            Whether the run was forked or not, by default False
        run_id : str | None, optional
            Run id of current run, by default None
        fork_run_id : str | None, optional
            Fork Run id from parent run, by default None
        offline : bool | None, optional
            Whether to run offline or not, by default False
        authentication : bool | None, optional
            Whether to authenticate with server or not, by default None
        log_hyperparams : bool | None, optional
            Whether to log hyperparameters, by default True
        on_resume_create_child: bool | None, optional
            Whether to create a child run when resuming a run, by default False
        """
        self._resumed = resumed
        self._forked = forked
        self._flag_log_hparams = log_hyperparams

        self._fork_run_server2server = None
        self._parent_run_server2server = None

        enabled = authentication and not offline
        self.auth = TokenAuth(tracking_uri, enabled=enabled)

        if rank_zero_only.rank == 0:
            if offline:
                LOGGER.info("MLflow is logging offline.")
            else:
                LOGGER.info("MLflow token authentication %s for %s", "enabled" if enabled else "disabled", tracking_uri)
                self.auth.authenticate()
                health_check(tracking_uri)

        run_id, run_name, tags = self._get_mlflow_run_params(
            project_name=project_name,
            run_name=run_name,
            config_run_id=run_id,
            fork_run_id=fork_run_id,
            tracking_uri=tracking_uri,
            on_resume_create_child=on_resume_create_child,
        )
        # Before creating the run we need to overwrite the tracking_uri and save_dir if offline
        if offline:
            # OFFLINE - When we run offline we can pass a save_dir pointing to a local path
            tracking_uri = None

        else:
            # ONLINE - When we pass a tracking_uri to mlflow then it will ignore the
            # saving dir and save all artifacts/metrics to the remote server database
            save_dir = None

        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            run_id=run_id,
        )

    def _check_server2server_lineage(self, run: mlflow.entities.Run) -> bool:
        """Address lineage and metadata for server2server runs.

        Those are runs that have been sync from one remote server to another
        """
        server2server = run.data.tags.get("server2server", "False") == "True"
        LOGGER.info("Server2Server: %s", server2server)
        if server2server:
            parent_run_across_servers = run.data.params.get(
                "metadata.offline_run_id",
                run.data.params.get("metadata.server2server_run_id"),
            )
            if self._forked:
                # if we want to fork a resume run we need to set the parent_run_across_servers
                # but just to restore the checkpoint
                self._fork_run_server2server = parent_run_across_servers
            else:
                self._parent_run_server2server = parent_run_across_servers

    def _get_mlflow_run_params(
        self,
        project_name: str,
        run_name: str,
        config_run_id: str,
        fork_run_id: str,
        tracking_uri: str,
        on_resume_create_child: bool,
    ) -> tuple[str | None, str, dict[str, Any]]:

        run_id = None
        tags = {"projectName": project_name}

        # create a tag with the command used to run the script
        command = os.environ.get("ANEMOI_TRAINING_CMD", sys.argv[0])
        tags["command"] = command.split("/")[-1]  # get the python script name
        tags["mlflow.source.name"] = command
        if len(sys.argv) > 1:
            # add the arguments to the command tag
            tags["command"] = tags["command"] + " " + " ".join(sys.argv[1:])

        if config_run_id or fork_run_id:
            "Either run_id or fork_run_id must be provided to resume a run."
            import mlflow

            self.auth.authenticate()
            mlflow_client = mlflow.MlflowClient(tracking_uri)

            if config_run_id and on_resume_create_child:
                parent_run_id = config_run_id  # parent_run_id
                parent_run = mlflow_client.get_run(parent_run_id)
                run_name = parent_run.info.run_name
                self._check_server2server_lineage(parent_run)
                tags["mlflow.parentRunId"] = parent_run_id
                tags["resumedRun"] = "True"  # tags can't take boolean values
            elif config_run_id and not on_resume_create_child:
                run_id = config_run_id
                run = mlflow_client.get_run(run_id)
                run_name = run.info.run_name
                self._check_server2server_lineage(run)
                mlflow_client.update_run(run_id=run_id, status="RUNNING")
                tags["resumedRun"] = "True"
            else:
                parent_run_id = fork_run_id
                tags["forkedRun"] = "True"
                tags["forkedRunId"] = parent_run_id
                run = mlflow_client.get_run(parent_run_id)
                self._check_server2server_lineage(run)

        if not run_name:
            import uuid

            run_name = f"{uuid.uuid4()!s}"

        if os.getenv("SLURM_JOB_ID"):
            tags["SLURM_JOB_ID"] = os.getenv("SLURM_JOB_ID")

        return run_id, run_name, tags

    @property
    def experiment(self) -> MLFlowLogger.experiment:
        if rank_zero_only.rank == 0:
            self.auth.authenticate()
        return super().experiment

    @rank_zero_only
    def log_system_metrics(self) -> None:
        """Log system metrics (CPU, GPU, etc)."""
        import mlflow
        from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

        mlflow.enable_system_metrics_logging()
        system_monitor = SystemMetricsMonitor(
            self.run_id,
            resume_logging=self.run_id is not None,
        )
        self.run_id_to_system_metrics_monitor = {}
        self.run_id_to_system_metrics_monitor[self.run_id] = system_monitor
        system_monitor.start()

    @rank_zero_only
    def log_terminal_output(self, artifact_save_dir: str | Path = "") -> None:
        """Log terminal logs to MLflow."""
        # path for logging terminal logs
        # for now the 'terminal_logs' file is kept in the same folder as the plots
        artifact_save_dir = Path(artifact_save_dir, self.run_id, "plots")

        log_monitor = LogsMonitor(
            artifact_save_dir,
            self.experiment,
            self.run_id,
        )
        self.run_id_to_log_monitor = {}
        self.run_id_to_log_monitor[self.run_id] = log_monitor
        log_monitor.start()

    @staticmethod
    def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
        """Clean up params to avoid issues with mlflow.

        Too many logged params will make the server take longer to render the
        experiment.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters to clean up.

        Returns
        -------
        dict[str, Any]
            Cleaned up params ready for MlFlow.
        """
        prefixes_to_remove = ["hardware", "data", "dataloader", "model", "training", "diagnostics", "metadata.config"]
        keys_to_remove = [key for key in params if any(key.startswith(prefix) for prefix in prefixes_to_remove)]
        for key in keys_to_remove:
            del params[key]
        return params

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | Namespace) -> None:
        """Overwrite the log_hyperparams method to flatten config params using '.'."""
        if self._flag_log_hparams:
            params = _convert_params(params)

            # this is needed to resolve optional missing config values to a string, instead of raising a missing error
            if config := params.get("config"):
                params["config"] = map_config_to_primitives(config)

            params = _flatten_dict(params, delimiter=".")  # Flatten dict with '.' to not break API queries
            params = self._clean_params(params)

            import mlflow
            from mlflow.entities import Param

            # Truncate parameter values.
            truncation_length = 250
            if Version(mlflow.VERSION) >= Version("1.28.0"):
                truncation_length = 500
            params_list = [Param(key=k, value=str(v)[:truncation_length]) for k, v in params.items()]

            for idx in range(0, len(params_list), 100):
                self.experiment.log_batch(run_id=self.run_id, params=params_list[idx : idx + 100])

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        # save the last obtained refresh token to disk
        self.auth.save()

        # finalize logging and system metrics monitor
        if getattr(self, "run_id_to_system_metrics_monitor", None):
            self.run_id_to_system_metrics_monitor[self.run_id].finish()
        if getattr(self, "run_id_to_log_monitor", None):
            self.run_id_to_log_monitor[self.run_id].finish(status)

        super().finalize(status)
