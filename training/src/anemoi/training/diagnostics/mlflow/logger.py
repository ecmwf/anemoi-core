# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import io
import logging
import os
import re
import sys
import time
from abc import ABC
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from threading import Thread
from typing import Any
from typing import Literal
from weakref import WeakValueDictionary

import mlflow
from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.loggers.mlflow import _convert_params
from pytorch_lightning.loggers.mlflow import _flatten_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing_extensions import override

from anemoi.training.diagnostics.mlflow import LOG_MODEL
from anemoi.training.diagnostics.mlflow import MAX_PARAMS_LENGTH
from anemoi.training.diagnostics.mlflow.utils import FixedLengthSet
from anemoi.training.diagnostics.mlflow.utils import clean_config_params
from anemoi.training.diagnostics.mlflow.utils import expand_iterables
from anemoi.utils.mlflow.auth import TokenAuth
from anemoi.utils.mlflow.utils import health_check

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


class BaseAnemoiMLflowLogger(MLFlowLogger, ABC):
    """A custom MLflow logger that logs terminal output."""

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        project_name: str = "anemoi",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        save_dir: str | None = "./mlruns",
        log_model: Literal["all"] | bool = LOG_MODEL,
        prefix: str = "",
        run_id: str | None = None,
        fork_run_id: str | None = None,
        offline: bool | None = False,
        authentication: bool | None = None,
        system: bool | None = True,
        terminal: bool | None = True,
        log_hyperparams: bool | None = True,
        expand_hyperparams: list[str] | None = None,
        on_resume_create_child: bool | None = True,
        max_params_length: int | None = MAX_PARAMS_LENGTH,
        http_max_retries: int | None = 35,
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
        run_id : str | None, optional
            Run id of current run, by default None
        fork_run_id : str | None, optional
            Fork Run id from parent run, by default None
        offline : bool | None, optional
            Whether to run offline or not, by default False
        authentication : bool | None, optional
            Whether to authenticate with server or not, by default None
        system: bool | None, optional
            If True, log system metrics, by default True
        terminal: bool | None, optional
            If True, log terminal output to mlflow, by default True
        log_hyperparams : bool | None, optional
            Whether to log hyperparameters, by default True
        expand_hyperparams: list[str] | None, optional
            keys to expand within params. Any key being expanded will have lists converted
            according to `expand_iterables`. By default ['config']
        on_resume_create_child: bool | None, optional
            Whether to create a child run when resuming a run, by default False
        max_params_length: int | None, optional
            Maximum number of params to be logged to Mlflow
        http_max_retries: int | None, optional
            Maximum number of retries for MLflow HTTP requests, default 35
        """
        self.authentication = authentication
        self.offline = offline
        self.log_system = system
        self.log_terminal = terminal
        self.expand_hyperparams = expand_hyperparams if expand_hyperparams else ["config"]
        self._resumed = run_id is not None
        self._forked = fork_run_id is not None
        self._flag_log_hparams = log_hyperparams
        if self._resumed and not on_resume_create_child:
            LOGGER.info(
                (
                    "Resuming run without creating child run - MLFlow logs will not update the"
                    "initial runs hyperparameters with those of the resumed run."
                    "To update the initial run's hyperparameters, set "
                    "`diagnostics.log.mlflow.on_resume_create_child: True`."
                ),
            )
            self._flag_log_hparams = False

        self._fork_run_server2server = None
        self._parent_run_server2server = None
        self._parent_dry_run = False
        self._max_params_length = max_params_length

        # max http retries
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(http_max_retries)
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = str(http_max_retries + 1)

        # these are the default values, but set them explicitly in case they change
        os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = "2"
        os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_JITTER"] = "1"

        # Report max parameters length
        LOGGER.info("Maximum number of params allowed to be logged is: %s", max_params_length)

        self.tracking_uri = tracking_uri
        # Before creating the run we need to overwrite the tracking_uri and save_dir if offline
        if self.offline:
            if self._resumed or self._forked:
                self.tracking_uri = save_dir
            else:
                # OFFLINE - When we run offline we can pass a save_dir pointing to a local path
                self.tracking_uri = None
                if save_dir is None:
                    # otherwise, by default we create a dir called "None"... not ideal
                    save_dir = "./mlruns"

        else:
            # ONLINE - When we pass a tracking_uri to mlflow then it will ignore the
            # saving dir and save all artifacts/metrics to the remote server database
            LOGGER.info("AnemoiMLFlow logging to %s", self.tracking_uri)
            save_dir = None

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        self._init_authentication()
        run_id, run_name, tags = self._get_mlflow_run_params(
            project_name=project_name,
            run_name=run_name,
            config_run_id=run_id,
            fork_run_id=fork_run_id,
            on_resume_create_child=on_resume_create_child,
        )

        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=self.tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            run_id=run_id,
        )

        # Track logged metrics to prevent duplicate logs
        # 2000 has been chosen as this should contain metrics form many steps
        self._logged_metrics = FixedLengthSet(maxlen=2000)  # Track (key, step)

    @abstractmethod
    def _init_authentication(self) -> None:
        """Initialize authentication specific to each logger."""

    def _check_dry_run(self, run: mlflow.entities.Run) -> None:
        """Check if the parent run is a dry run.

        A dry run is a run that is used as template base run
        but do not contain any checkpoints.
        """
        dry_run = run.data.tags.get("dry_run", "False") == "True"
        LOGGER.info("Parent run is a Dry Run: %s", dry_run)
        self._parent_dry_run = dry_run

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
            mlflow_client = mlflow.MlflowClient(self.tracking_uri)

            # This block is used when a run ID is specified with child runs option activated
            if config_run_id and on_resume_create_child and not fork_run_id:
                parent_run_id = config_run_id  # parent_run_id
                parent_run = mlflow_client.get_run(parent_run_id)
                run_name = parent_run.info.run_name
                self._check_server2server_lineage(parent_run)
                self._check_dry_run(parent_run)
                tags["mlflow.parentRunId"] = parent_run_id
                tags["resumedRun"] = "True"  # tags can't take boolean values
            # This block is used when a run ID is specified without child runs option activated
            elif config_run_id and not on_resume_create_child and not fork_run_id:
                run_id = config_run_id
                run = mlflow_client.get_run(run_id)
                run_name = run.info.run_name
                self._check_server2server_lineage(run)
                self._check_dry_run(run)
                mlflow_client.update_run(run_id=run_id, status="RUNNING")
                tags["resumedRun"] = "True"
            # This block is used when a run is forked and an existing run ID is specified
            # Child run option is activated
            elif config_run_id and fork_run_id:
                parent_run_id = config_run_id  # parent_run_id which is the main run ID
                parent_run = mlflow_client.get_run(parent_run_id)
                run_name = parent_run.info.run_name
                self._check_server2server_lineage(parent_run)
                self._check_dry_run(parent_run)
                tags["mlflow.parentRunId"] = config_run_id  # We want to be linked to the main run ID
                tags["resumedRun"] = "True"  # We want to be linked to the main run ID
                tags["forkedRun"] = "True"  # This is a forked run
                tags["forkedRunId"] = fork_run_id  # This is a forked run
            # This block is used when a run is forked without no child runs
            else:
                parent_run_id = fork_run_id
                tags["forkedRun"] = "True"
                tags["forkedRunId"] = parent_run_id
                run = mlflow_client.get_run(parent_run_id)
                self._check_server2server_lineage(run)
                self._check_dry_run(run)

        if not run_name:
            import uuid

            run_name = f"{uuid.uuid4()!s}"

        if os.getenv("SLURM_JOB_ID"):
            tags["SLURM_JOB_ID"] = os.getenv("SLURM_JOB_ID")

        return run_id, run_name, tags

    @override
    @property
    def experiment(self) -> MLFlowLogger.experiment:
        if rank_zero_only.rank == 0:
            self.auth.authenticate()
        return super().experiment

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        cleaned_metrics = metrics.copy()
        for k in metrics:
            metric_id = (k, step or 0)
            if metric_id in self._logged_metrics:
                cleaned_metrics.pop(k)
                continue
            self._logged_metrics.add(metric_id)
        return super().log_metrics(metrics=cleaned_metrics, step=step)

    @rank_zero_only
    def log_system_metrics(self) -> None:
        """Log system metrics (CPU, GPU, etc)."""
        import mlflow
        from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
        from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor
        from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

        from anemoi.training.diagnostics.mlflow.system_metrics.cpu_monitor import CPUMonitor
        from anemoi.training.diagnostics.mlflow.system_metrics.gpu_monitor import GreenGPUMonitor
        from anemoi.training.diagnostics.mlflow.system_metrics.gpu_monitor import RedGPUMonitor

        class CustomSystemMetricsMonitor(SystemMetricsMonitor):
            def __init__(self, run_id: str, resume_logging: bool = False):
                super().__init__(run_id, resume_logging=resume_logging)

                self.monitors = [CPUMonitor(), DiskMonitor(), NetworkMonitor()]

                # Try init both and catch the error when one init fails
                try:
                    gpu_monitor = GreenGPUMonitor()
                    self.monitors.append(gpu_monitor)
                except (ImportError, RuntimeError) as e:
                    LOGGER.warning("Failed to init Nvidia GPU Monitor: %s", e)
                try:
                    gpu_monitor = RedGPUMonitor()
                    self.monitors.append(gpu_monitor)
                except (ImportError, RuntimeError) as e:
                    LOGGER.warning("Failed to init AMD GPU Monitor: %s", e)

        mlflow.enable_system_metrics_logging()
        # https://mlflow.org/docs/latest/system-metrics/
        # By default, system metrics are sampled every 10 seconds
        # we choose to update this to 100 - system metrics are logged every 1 min 30 seconds
        mlflow.set_system_metrics_sampling_interval(interval=100)
        system_monitor = CustomSystemMetricsMonitor(
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

    @rank_zero_only
    def log_hyperparams(
        self,
        params: dict[str, Any] | Namespace,
    ) -> None:
        """Overwrite the log_hyperparams method.

        - flatten config params using '.'.
        - expand keys within params to avoid truncation.
        - log hyperparameters as an artifact.

        Parameters
        ----------
        params : dict[str, Any] | Namespace
            params to log
        """
        if self._flag_log_hparams:
            params = _convert_params(params)

            # this is needed to resolve optional missing config values to a string, instead of raising a missing error
            if config := params.get("config"):
                params["config"] = config.model_dump(by_alias=True)

            self.log_hyperparams_as_mlflow_artifact(
                client=self.experiment,
                run_id=self.run_id,
                params=params,
            )
            self.log_hyperparams_in_mlflow(
                self.experiment,
                self.run_id,
                params,
                expand_keys=self.expand_hyperparams,
                max_params_length=self._max_params_length,
            )

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

    @classmethod
    def log_hyperparams_in_mlflow(
        cls,
        client: MlflowClient,
        run_id: str,
        params: dict[str, Any] | Namespace,
        *,
        expand_keys: list[str] | None = None,
        clean_params: bool = True,
        max_params_length: int | None = MAX_PARAMS_LENGTH,
    ) -> None:
        """Log hyperparameters to MLflow server.

        - flatten config params using '.'.
        - expand keys within params to avoid truncation.
        - log hyperparameters as an artifact.

        Parameters
        ----------
        client : MlflowClient
            MLflow client.
        run_id : str
            Run ID.
        params : dict[str, Any] | Namespace
            params to log.
        expand_keys : list[str] | None, optional
            keys to expand within params. Any key being expanded will
            have lists converted according to `expand_iterables`,
            by default None.
        max_params_length: int | None, optional
            Maximum number of params to be logged to Mlflow
        """
        import mlflow
        from mlflow.entities import Param

        try:  # Check maximum param value length is available and use it
            truncation_length = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        except AttributeError:  # Fallback (in case of MAX_PARAM_VAL_LENGTH not available)
            truncation_length = 250  # Historical default value

        cls.log_hyperparams_as_mlflow_artifact(client=client, run_id=run_id, params=params)

        expanded_params = {}
        params = params.copy()
        for key in expand_keys or []:
            if key in params:
                expanded_params.update(
                    expand_iterables(params, size_threshold=None, delimiter="."),
                )
        expanded_params.update(params)

        expanded_params = _flatten_dict(
            expanded_params,
            delimiter=".",
        )  # Flatten dict with '.' to not break API queries

        if clean_params:
            expanded_params = clean_config_params(expanded_params)

        LOGGER.info("Logging %s parameters", len(expanded_params))
        if len(expanded_params) > max_params_length:
            msg = (
                f"Too many params: {len(expanded_params)} > {max_params_length}",
                "Please revisit the fields being logged and add redundant or irrelevant "
                "ones to the clean_config_params function.",
            )
            raise ValueError(msg)

        # Truncate parameter values.
        params_list = [Param(key=k, value=str(v)[:truncation_length]) for k, v in expanded_params.items()]
        for idx in range(0, len(params_list), 100):
            client.log_batch(run_id=run_id, params=params_list[idx : idx + 100])

    @staticmethod
    def log_hyperparams_as_mlflow_artifact(
        client: MlflowClient,
        run_id: str,
        params: dict[str, Any] | Namespace,
    ) -> None:
        """Log hyperparameters as an artifact."""
        import json
        import tempfile
        from json import JSONEncoder

        class StrEncoder(JSONEncoder):
            def default(self, o: Any) -> str:
                return str(o)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            with Path.open(path, "w") as f:
                json.dump(params, f, cls=StrEncoder)
            client.log_artifact(run_id=run_id, local_path=path)

    @override
    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: Checkpoint) -> None:
        """Logs model checkpoint as an artifact, sanitizing the path correctly."""
        if not self._log_model:
            return

        import shutil
        import tempfile

        # Get the path to the checkpoint file saved by the callback
        model_path = checkpoint_callback.last_model_path or checkpoint_callback.best_model_path
        if not model_path or not Path(model_path).exists():
            LOGGER.warning("after_save_checkpoint failed: Checkpoint path not found.")
            return

        # Use a temporary directory to handle the sanitized file
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create a safe filename by replacing the colon
            safe_filename = Path(Path(model_path).name.replace(":", "-"))

            # 2. Create the full path for the temporary, sanitized file
            tmp_model_path = Path(tmpdir) / safe_filename

            # 3. Copy the original checkpoint to this new temporary path
            shutil.copy2(model_path, tmp_model_path)

            # 4. Define ONLY the destination directory for MLflow
            artifact_subdir = "checkpoints"

            # 5. Log the sanitized temporary file into the 'checkpoints' directory
            msg = f"Logging checkpoint '{model_path}' to MLflow artifact path '{artifact_subdir}/{safe_filename}'"
            LOGGER.info(msg)
            try:
                self.experiment.log_artifact(self.run_id, tmp_model_path, artifact_subdir)
            except Exception:
                LOGGER.exception()
                raise


class AnemoiMLflowLogger(BaseAnemoiMLflowLogger):
    def _init_authentication(self) -> None:
        """Authentication for a standard MLFlow server."""
        enabled = self.authentication and not self.offline
        self.auth = TokenAuth(self.tracking_uri, enabled=enabled)

        if rank_zero_only.rank == 0:
            if self.offline:
                LOGGER.info("MLflow is logging offline.")
            else:
                LOGGER.info(
                    "MLflow token authentication %s for %s",
                    "enabled" if enabled else "disabled",
                    self.tracking_uri,
                )
                self.auth.authenticate()
                health_check(self.tracking_uri)
