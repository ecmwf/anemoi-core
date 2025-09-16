# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only

if TYPE_CHECKING:
    import importlib

    import pytorch_lightning as pl
    from omegaconf import DictConfig
    from pytorch_lightning.utilities.types import STEP_OUTPUT

    from anemoi.training.train.tasks.base import BaseGraphModule

    if importlib.util.find_spec("ipywidgets") is not None:
        from tqdm.auto import tqdm as _tqdm
    else:
        from tqdm import tqdm as _tqdm

from torch.profiler import profile

from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

LOGGER = logging.getLogger(__name__)


def check_torch_version() -> bool:
    torch_version = torch.__version__
    version_nums = torch_version.split(".")
    major_version = int(version_nums[0])
    minor_version = int(version_nums[1])
    if major_version == 2 and minor_version >= 1:
        return True
    LOGGER.error("Memory snapshot is only supported for torch >= 2.1")
    return False


def convert_to_seconds(time_str: str) -> float:
    import re

    pattern = r"(\d+(\.\d+)?)\s*([a-zA-Z]+)"
    # Use regex to find matches
    match = re.match(pattern, time_str)

    # Check if match is found
    if match:
        # Extract numeric part and unit part
        numeric_part = float(match.group(1))
        unit = match.group(3)

        # Convert the unit to seconds
        if unit == "s":
            return numeric_part
        if unit == "ds":
            return numeric_part / 10  # Convert decaseconds to seconds
        if unit == "cs":
            return numeric_part / 100  # Convert centiseconds to seconds
        if unit == "ms":
            return numeric_part / 1000  # Convert milliseconds to seconds
        error_msg = (
            "Invalid unit. Supported units are: 's' (seconds)'"
            "'ds' (decaseconds), 'cs' (centiseconds) and 'ms' (miliseconds) .",
        )
        raise ValueError(error_msg)
    error_msg = "Invalid time format. The time should be in the format: 'numeric_part unit'. For example: '10 ms'"
    raise ValueError(error_msg)


PROFILER_ACTIONS = [
    r"\[Strategy]\w+\.batch_to_device",
    r"\[Strategy]\w+\.backward",
    r"\[Strategy]\w+\.training_step",
    r"\[Strategy]\w+\.validation_step",
    r"\[Strategy]\w+\.batch_to_device",
    "run_training_epoch",
    "run_training_batch",
    r"\[_EvaluationLoop\]\.\w+",
    r"\[_TrainingEpochLoop\]\.\w+",
    r"\[LightningDataModule]\w+\.train_dataloader",
    r"\[LightningDataModule]\w+\.val_dataloader",
    r"\[LightningDataModule]\w+\.state_dict",
    r"\[LightningDataModule]\w+\.setup",
    r"\[LightningDataModule]\w+\.prepare_data",
    r"\[LightningDataModule]\w+\.teardown",
    r"\[LightningModule]\w+\.optimizer_step",
    r"\[LightningModule]\w+\.configure_gradient_clipping",
    r"\[LightningModule]\w+\.on_validation_model_eval",
    r"\[LightningModule]\w+\.optimizer_zero_grad",
    r"\[LightningModule]\w+\.transfer_batch_to_device",
    r"\[LightningModule]\w+\.on_validation_model_train",
    r"\[LightningModule]\w+\.configure_optimizers",
    r"\[LightningModule]\w+\.lr_scheduler_step",
    r"\[LightningModule]\w+\.configure_sharded_model",
    r"\[LightningModule]\w+\.setup",
    r"\[LightningModule]\w+\.prepare_data",
    r"\[Callback\](.*Plot*)",
    r"\[Callback\](.*Checkpoint*)",
]

GPU_METRICS_DICT = {
    "GPU device utilization (%)": "gpu",
    "GPU memory use (%)": "memory",
    "GPU memory allocated (%)": "memoryAllocated",
    "GPU memory allocated (GB)": "memoryAllocatedBytes",
}


class WandBSystemSummarizer:
    """Summarize System Metrics provided by W&B logger."""

    def __init__(self, wandb_logger: pl.loggers.WandbLogger):

        run_dict = wandb_logger._wandb_init
        self.run_id_path = f"{run_dict['entity']}/{run_dict['project']}/{run_dict['id']}"

    def get_wandb_metrics(self) -> (pd.DataFrame, dict):
        """Fetches system metrics and metadata from a W&B run."""
        import wandb

        run = wandb.Api().run(self.run_id_path)
        system_metrics = run.history(stream="events")
        metadata_dict = run.metadata
        system_metrics = system_metrics.dropna()
        return system_metrics, metadata_dict

    def summarize_gpu_metrics(self, df: pd.DataFrame) -> dict[str, float]:
        """Given the System Metrics DataFrame, summarized the GPU metrics.

        - gpu.{gpu_index}.memory - GPU memory utilization in percent for each GPU
        - gpu.{gpu_index}.memoryAllocated - GPU memory allocated as % of the total available memory for each GPU
        - gpu.{gpu_index}.memoryAllocatedBytes - GPU memory allocated in bytes for each GPU
        - gpu.{gpu_index}.gpu - GPU utilization in percent for each GPU
        """
        average_metric = {}
        col_names = df.columns
        for gpu_metric_name, gpu_metric in GPU_METRICS_DICT.items():
            pattern = rf"system.gpu.\d.{gpu_metric}$"
            sub_gpu_cols = [string for string in col_names if re.match(pattern, string)]
            metrics_per_gpu = df[sub_gpu_cols].mean(axis=0)
            if gpu_metric == "memoryAllocatedBytes":
                metrics_per_gpu = metrics_per_gpu * 1e-9
            average_metric[gpu_metric_name] = metrics_per_gpu.mean()
            # Just add metrics per gpu to the report if we have more than 1 GPU
            if metrics_per_gpu.shape[0] > 1:
                metrics_per_gpu.index = ["   " + index for index in metrics_per_gpu.index]
                average_metric.update(dict(metrics_per_gpu))
        return average_metric

    def summarize_system_metrics(self) -> dict[str, float]:
        r"""Summarizes the System metrics from a W&B run.

        Some of the metrics included are:
        - cpu.{}.cpu_percent - CPU usage of the system on a per-core basis.
        - system.memory - Represents the total system memory usage as a percentage of the total available memory.
        - system.cpu - Percentage of CPU usage by the process, normalized by the number of available CPUs
        - system.disk.\\.usageGB - (Represents the total system disk usage in gigabytes (GB))
        - system.proc.memory.percent - Indicates the memory usage of the process as a % of the total available memory

        More information about W&B system metrics can be found here:
        https://docs.wandb.ai/guides/app/features/system-metrics
        """
        system_metrics_df, metadata_dict = self.get_wandb_metrics(self.run_id_path)

        col_names = system_metrics_df.columns
        system_metrics = {}

        n_cpus = metadata_dict["cpu_count"]
        cpu_cols = list(filter(lambda k: "cpu." in k, col_names))
        system_metrics["avg CPU usage (%)"] = (system_metrics_df[cpu_cols].sum(axis=1) / n_cpus).mean()

        system_metrics_gpu = self.summarize_gpu_metrics(system_metrics_df)
        system_metrics.update(system_metrics_gpu)

        system_metrics["avg Memory usage (%)"] = system_metrics_df["system.memory"].mean()
        system_metrics["avg Disk usage (GB)"] = system_metrics_df["system.disk.\\.usageGB"].mean()
        system_metrics["avg Disk usage  (%)"] = system_metrics_df["system.disk.\\.usagePercent"].mean()

        system_metrics["execution time (sec)"] = system_metrics_df["_runtime"].iloc[-1]  # in seconds
        return system_metrics


class MLFlowSystemSummarizer:
    """Summarize System Metrics provided by MlFlow logger."""

    def __init__(self, mlflow_logger: pl.loggers.MLFlowLogger):
        self.run_id = mlflow_logger.run_id
        self.mlflow_client = mlflow_logger._mlflow_client

    @property
    def system_metrics(self) -> list[str]:
        run = self.mlflow_client.get_run(self.run_id)
        return [metric for metric in run.data.metrics if "system" in metric]

    def _clean_metric_name(self, metric_name: str) -> str:
        return (
            metric_name.replace("system.", "avg ")
            .replace("_", " ")
            .replace("megabytes", "MB")
            .replace("percentage", "%")
        )

    def _get_mean(self, pattern: str, df: pd.DataFrame) -> float:
        # Filter rows containing the pattern in the 'metric' column
        filtered_rows = df[df["metric"].str.contains(pattern)]
        return filtered_rows.loc[:, "value"].astype(np.float32).mean()

    def _extract_gpu_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Define the pattern you want to search for
        pattern = r"gpu\s\d+\s+utilization"
        df.loc[len(df.index)] = ["avg GPU utilization (%)", self._get_mean(pattern, df)]

        pattern = r"gpu\s\d+\s+memory\s+usage\s+%"
        df.loc[len(df.index)] = ["avg GPU memory usage %", self._get_mean(pattern, df)]

        pattern = r"gpu\s\d+\s+memory\s+usage\s+MB"
        df.loc[len(df.index)] = ["avg GPU memory usage MB", self._get_mean(pattern, df)]

        return df

    def summarize_mlflow_system_metrics(self) -> pd.DataFrame:
        rows = []
        for metric in self.system_metrics:
            metric = self.mlflow_client.get_metric_history(self.run_id, metric)
            avg_value = sum(m.value for m in metric) / len(metric)
            metric_name = self._clean_metric_name(metric[0].key)
            rows.append({"metric": metric_name, "value": f"{avg_value:.2f}"})
        return self._extract_gpu_metrics(pd.DataFrame(rows))


class DummyProfiler(Profiler):
    """Placeholder profiler."""

    def __init__(self):
        super().__init__()

    def start(self, *args, **kwargs) -> None:
        pass

    def stop(self, *args, **kwargs) -> None:
        pass


def _convert_npint_to_int(obj: Any) -> dict | list | int | str | float:
    """Recursively converts all np.int64 values in the input to Python int."""
    # Recursively converts all np.int64 to int
    if isinstance(obj, dict):
        return {k: _convert_npint_to_int(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_npint_to_int(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)  # Convert np.int64 to int
    return obj


class PatchedProfile(profile):

    def _get_distributed_info(self) -> dict[str, str]:
        dist_info = super()._get_distributed_info()
        return _convert_npint_to_int(dist_info)


class BenchmarkProfiler(Profiler):
    """Custom PyTorch Lightning profiler for benchmarking."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.config = config
        self.warmup = self.config.diagnostics.benchmark_profiler.memory.warmup
        if not self.warmup:
            self.warmup = 0
        self.num_steps = self.config.diagnostics.benchmark_profiler.memory.steps

        if self.config.diagnostics.benchmark_profiler.memory.extra_plots:
            assert (
                self.num_steps <= self.config.training.num_sanity_val_steps
            ), "Sanity steps should be less than snapshot steps, to avoid memory issues"

        self.dirpath = None
        self.create_output_path()
        # the profilers need to be initialised before the setup method because
        # actions like configuring callbacks would trigger the profiler
        self.memory_profiler = DummyProfiler  # dummy profiler to be used as placeholder
        self.time_profiler = DummyProfiler  # dummy profiler to be used as placeholder

    @rank_zero_only
    def create_output_path(self) -> None:
        self.dirpath = Path(self.config.hardware.paths.profiler)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def broadcast_profiler_path(self, string_var: str, src_rank: int) -> str:
        from lightning_fabric.utilities.distributed import group as _group

        string_var = [string_var]
        dist.broadcast_object_list(string_var, src_rank, group=_group.WORLD)
        return string_var[0]

    def setup(self, stage: str, local_rank: int | None = None, log_dir: str | None = None) -> None:
        del log_dir
        # THE STRATEGY IS ALREADY INITIALISED AND TORCH DISTRIBUTED IS ACTIVE
        # we need to broadcast the profiler path to all ranks to save the memory traces
        self.dirpath = Path(self.broadcast_profiler_path(str(self.dirpath), 0))
        self._stage = stage
        self._local_rank = local_rank
        self._create_time_profilers()
        self._create_memory_profilers()

    def _create_time_profilers(self) -> None:
        """Creates profilers for time and memory measurements."""
        if self.config.diagnostics.benchmark_profiler.time.enabled:
            self.time_profiler = SimpleProfiler(
                dirpath=self.dirpath,
            )

    def _create_memory_profilers(self) -> None:
        if self.config.diagnostics.benchmark_profiler.memory.enabled:
            import os

            def trace_handler(dir_name: str, stage: str | None = None) -> callable:

                def handler_fn(prof: pl.profilers.Profiler) -> None:
                    import socket
                    import time

                    worker_name = f"{socket.gethostname()}_{os.getpid()}"
                    file_name = str(dir_name / f"{worker_name}.{stage}.{time.time_ns()}.pt.trace.json")
                    LOGGER.info("Saving memory trace to %s", file_name)
                    prof.export_chrome_trace(file_name)

                return handler_fn

            global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # WON'T WORK WHEN RUNNING WITHOUT SLURM
            if not (self.config.diagnostics.benchmark_profiler.memory.trace_rank0_only and global_rank != 0):
                from pytorch_lightning.profilers.pytorch import _KINETO_AVAILABLE

                assert (
                    _KINETO_AVAILABLE
                ), "Kineto is not available. Please ensure Kineto is avaialble to be able to use the memory profiler"

                torch.profiler.profile = (
                    PatchedProfile  # patch the profile(KinetoProfile) object to serialise the distributed info
                )
                self.memory_profiler = PyTorchProfiler(
                    with_stack=True,
                    emit_nvtx=False,
                    profile_memory=True,
                    export_to_chrome=True,
                    record_shapes=True,
                    group_by_input_shapes=True,
                    dirpath=self.dirpath,
                    on_trace_ready=trace_handler(self.dirpath),
                    schedule=torch.profiler.schedule(
                        wait=0,
                        warmup=self.warmup,
                        active=self.num_steps,
                        repeat=1,
                        skip_first=self.config.training.num_sanity_val_steps,
                    ),
                )
        self.time_rows_dict = None  # updated if we create a memory profile report

    def start(self, action_name: str) -> None:
        """Starts recording for a specific action.

        Parameters
        ----------
        action_name : str
            Name of the action.
        """
        self.time_profiler.start(action_name)
        self.memory_profiler.start(action_name)

    def stop(self, action_name: str) -> None:
        """Stops recording for a specific action.

        Parameters
        ----------
        action_name : str
            Name of the action.
        """
        self.time_profiler.stop(action_name)
        self.memory_profiler.stop(action_name)

    def _trim_time_report(self, recorded_actions: dict) -> dict[str, float]:
        all_actions_names = recorded_actions.keys()
        df = pd.DataFrame({"Strings": all_actions_names})
        combined_pattern = "|".join(PROFILER_ACTIONS)
        filtered_df = df[df["Strings"].str.contains(combined_pattern, regex=True, na=False)]
        trimmed_actions_names = filtered_df["Strings"].tolist()
        return {key: recorded_actions[key] for key in trimmed_actions_names}

    def get_time_profiler_df(self, precision: int = 5) -> pd.DataFrame:
        """Retrieves a DataFrame with time profiling information.

        Parameters
        ----------
        precision : int
            Precision for rounding, by default 5

        Returns
        -------
        pd.DataFrame
            DataFrame with time profiling information.
        """
        if self.config.diagnostics.benchmark_profiler.time.verbose is False:
            self.time_profiler.recorded_durations = self._trim_time_report(
                recorded_actions=self.time_profiler.recorded_durations,
            )
        time_df = pd.DataFrame(self.time_profiler.recorded_durations.items())
        time_df[2] = time_df[1].apply(len)
        time_df[3] = time_df[1].apply(np.mean)
        time_df[1] = time_df[1].apply(sum)
        time_df.columns = ["name", "total_time", "n_calls", "avg_time"]

        def replace_function(value: str) -> str:
            # Replace 'apple' with 'fruit'
            return re.sub(r"\{.*?\}", "", value)  # Remove anything between braces

        time_df["name"] = time_df["name"].apply(replace_function)
        pattern = r"\[(.*?)\]|(.*)"
        time_df["category"] = time_df["name"].str.extract(pattern, expand=False)[0].fillna(time_df["name"])

        pattern = re.compile(r"\[Callback\](.*?)\.")
        # Apply the regular expression to the column
        callbacks_subcategories = "*Callback_" + time_df[time_df["category"] == "Callback"]["name"].str.extract(pattern)
        indexer = time_df[time_df["category"] == "Callback"].index
        time_df.loc[indexer, "category"] = callbacks_subcategories[0].tolist()

        # Check if 'Callback' is present in the 'category' column
        time_df["is_callback"] = time_df["category"].str.contains("Callback", case=False)

        # Group by the 'is_callback' column and apply groupby operation only on rows with 'Callback' in 'category'
        grouped_data = (
            time_df[time_df["is_callback"]]
            .groupby("category")
            .agg({"n_calls": "sum", "avg_time": "sum", "total_time": "sum"})
            .reset_index()
        )
        grouped_data["name"] = grouped_data["category"]

        time_df = pd.concat([time_df[~time_df["is_callback"]], grouped_data])
        time_df = time_df.drop("is_callback", axis=1)
        time_df = time_df.round(precision)
        time_df = time_df.sort_values(by="category", ascending=False)

        self.time_report_fname = self.dirpath / "time_profiler.csv"
        self._save_report(time_df, self.time_report_fname)
        return time_df

    @staticmethod
    def to_df(sample_dict: dict[str, float], precision: str = ".5") -> pd.DataFrame:
        df = pd.DataFrame(sample_dict.items())
        df.columns = ["metric", "value"]
        df.value = df.value.apply(lambda x: f"%{precision}f" % x)
        return df

    @rank_zero_only
    def get_system_profiler_df(self, logger_name: str, logger: pl.loggers.Logger) -> pd.DataFrame:
        if logger_name == "wandb":
            system_metrics_df = self.to_df(WandBSystemSummarizer(logger).summarize_system_metrics())
        elif logger_name == "mlflow":
            system_metrics_df = MLFlowSystemSummarizer(logger).summarize_mlflow_system_metrics()
        elif logger_name == "tensorboard":
            LOGGER.info("No system profiler data available for Tensorboard")
            system_metrics_df = None

        self.system_report_fname = self.dirpath / "system_profiler.csv"
        self._save_report(system_metrics_df, self.system_report_fname)
        return system_metrics_df

    def _save_report(self, df: pd.DataFrame, fname: Path) -> None:
        df.to_csv(fname)

    def _save_model_summary(self, model_summary: str, fname: Path) -> None:
        with fname.open("w") as f:
            f.write(model_summary)
            f.close()

    def get_model_summary(self, model: BaseGraphModule, example_input_array: np.ndarray) -> str:

        from torchinfo import summary

        # when using flash attention model, we need to convert the input and model to float16 and cuda
        # since FlashAttention only supports fp16 and bf16 data type
        example_input_array = example_input_array.to(dtype=torch.float16)
        example_input_array = example_input_array.to("cuda")
        model.half()
        model = model.to("cuda")

        summary_str = str(
            summary(
                model,
                input_data=example_input_array,
                depth=20,
                col_width=16,
                col_names=["trainable", "input_size", "output_size", "num_params", "params_percent", "mult_adds"],
                row_settings=["var_names"],
                verbose=0,
            ),
        )
        self.model_summary_fname = self.dirpath / "model_summary.txt"
        self._save_model_summary(summary_str, self.model_summary_fname)
        return summary_str

    @rank_zero_only
    def get_speed_profiler_df(self, progressbar: _tqdm) -> pd.DataFrame:
        """Computes the speed metrics based on training and validation rates."""
        speed_metrics = {}

        batch_size_tr = self.config.dataloader.batch_size.training
        batch_size_val = self.config.dataloader.batch_size.validation

        training_rates_array = np.array(progressbar.training_rates)
        speed_metrics["training_avg_throughput"] = training_rates_array.mean()
        speed_metrics["training_avg_throughput_per_sample"] = training_rates_array.mean() / batch_size_tr

        validation_rates_array = np.array(progressbar.validation_rates)
        speed_metrics["validation_avg_throughput"] = validation_rates_array.mean()
        speed_metrics["validation_avg_throughput_per_sample"] = validation_rates_array.mean() / batch_size_val

        # Calculate per_sample metrics
        speed_metrics["avg_training_dataloader_throughput"] = (
            1 / np.array(self.time_profiler.recorded_durations["[_TrainingEpochLoop].train_dataloader_next"]).mean()
        )
        speed_metrics["avg_training_dataloader_throughput_per_sample"] = (
            speed_metrics["avg_training_dataloader_throughput"] / batch_size_tr
        )

        speed_metrics["avg_validation_dataloader_throughput"] = (
            1 / np.array(self.time_profiler.recorded_durations["[_EvaluationLoop].val_next"]).mean()
        )
        speed_metrics["avg_validation_dataloader_throughput_per_sample"] = (
            speed_metrics["avg_validation_dataloader_throughput"] / batch_size_val
        )

        if self.time_rows_dict:
            speed_metrics.update(self.time_rows_dict)

        speed_profile_df = self.to_df(speed_metrics)

        self.speed_report_fname = self.dirpath / "speed_profiler.csv"
        self._save_report(speed_profile_df, self.speed_report_fname)

        return speed_profile_df

    def _save_extra_plots(self) -> None:
        if check_torch_version():
            # !it's available for  torch >= 2.1
            from torch.cuda._memory_viz import profile_plot

            self.memory_trace_fname = Path(self.dirpath, "memory_trace.html")
            with self.memory_trace_fname.open("w") as f:
                f.write(profile_plot(self.memory_profiler.profiler))

            # !it's available for  torch >= 2.1
            self.memory_timeline_fname = str(Path(self.dirpath, "memory_timelines.html"))
            self.memory_profiler.profiler.export_memory_timeline(self.memory_timeline_fname)

    @rank_zero_only
    def get_memory_profiler_df(self) -> pd.DataFrame:
        """Retrieves the memory profiler data as a DataFrame.

        Aggregates the results coming from multiple nodes/processes.

        Returns
        -------
        pd.DataFrame
            Memory profiler data.
        """
        if self.config.diagnostics.benchmark_profiler.memory.extra_plots:
            self._save_extra_plots()

        self.memory_profiler._delete_profilers()

        if not self.memory_profiler.function_events:
            return ""

        data = self.memory_profiler.function_events.key_averages(
            group_by_input_shapes=self.memory_profiler._group_by_input_shapes,
        )
        table = data.table(
            sort_by=self.memory_profiler._sort_by_key,
            row_limit=self.memory_profiler._row_limit,
            **self.memory_profiler._table_kwargs,
        )  # this is a string

        from io import StringIO

        table_main_body = table.split("\n")[:-3]  # Remove the last rows
        columns = [
            "Name",
            "Self CPU %",
            "Self CPU",
            "CPU total %",
            "CPU total",
            "CPU time avg",
            "Self CUDA",
            "Self CUDA %",
            "CUDA total",
            "CUDA time avg",
            "CPU Mem",
            "Self CPU Mem",
            "CUDA Mem",
            "Self CUDA Mem",
            "# of Calls",
            "Input Shapes",
        ]
        table_main_body = "\n".join(table_main_body)
        memory_df = pd.read_fwf(StringIO(table_main_body), names=columns, skiprows=2)
        flag = ["--" not in row for row in memory_df["Name"]]
        memory_df = memory_df[flag]
        time_rows = [row for row in table.split("\n")[-3:] if row != ""]
        try:
            if time_rows:
                time_rows_dict = {}
                for row in time_rows:
                    key, val = row.split(":")
                    val = convert_to_seconds(val.strip())
                    time_rows_dict[key] = val
                self.time_rows_dict = time_rows_dict

                memory_df = memory_df[~memory_df["Name"].isin(time_rows)]
        except ValueError as e:
            LOGGER.info("Error saving memory df: %s", e)

        self.memory_report_fname = self.dirpath / "memory_profiler.csv"
        self._save_report(memory_df, self.memory_report_fname)
        return memory_df


class ProfilerProgressBar(TQDMProgressBar):
    """Custom PyTorch Lightning progress bar with profiling functionality.

    Attributes
    ----------
    validation_rates : list[float]
        List to store validation rates (it/s).
    training_rates : list[float]
        List to store training rates (it/s).
    """

    def __init__(self):
        super().__init__()
        self.validation_rates = []
        self.training_rates = []

    def _extract_rate(self, pbar: _tqdm) -> float:
        """Extracts the iteration rate from the progress bar.

        Parameters
        ----------
        pbar : tqdm
            The progress bar.

        Returns
        -------
        float
            The iteration rate.
        """
        return (pbar.format_dict["n"] - pbar.format_dict["initial"]) / pbar.format_dict["elapsed"]

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Appends the rate from the progress bar to the list of 'training_rates'."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.train_progress_bar.format_dict["n"] != 0:
            self.training_rates.append(self._extract_rate(self.train_progress_bar))
            for logger in self.trainer.loggers:
                if isinstance(logger, AnemoiMLflowLogger):
                    logger.log_metrics({"training_rate": self.training_rates[-1]}, step=trainer.global_step)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Append rate from the progress bar to the list of 'validation_rates'."""
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.val_progress_bar.format_dict["n"] != 0:
            self.validation_rates.append(self._extract_rate(self.val_progress_bar))
            for logger in self.trainer.loggers:
                if isinstance(logger, AnemoiMLflowLogger):
                    logger.log_metrics({"validation_rate": self.validation_rates[-1]}, step=trainer.global_step)
