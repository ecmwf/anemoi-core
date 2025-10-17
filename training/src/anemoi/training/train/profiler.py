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
import warnings
from datetime import datetime
from datetime import timezone
from functools import cached_property
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from rich.console import Console

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.diagnostics.profilers import BenchmarkProfiler
from anemoi.training.diagnostics.profilers import ProfilerProgressBar
from anemoi.training.train.train import AnemoiTrainer

LOGGER = logging.getLogger(__name__)
console = Console(record=True, width=200)


class AnemoiProfiler(AnemoiTrainer):
    """Profiling for Anemoi."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def print_report(self, title: str, dataframe: pd.DataFrame, color: str = "white", emoji: str = "") -> None:
        if title == "Model Summary":
            console.print(f"[bold {color}]{title}[/bold {color}]", f":{emoji}:")
            console.print(dataframe, end="\n\n")
        else:
            console.print(f"[bold {color}]{title}[/bold {color}]", f":{emoji}:")
            console.print(dataframe.to_markdown(headers="keys", tablefmt="psql"), end="\n\n")

    @staticmethod
    def print_title() -> None:
        console.print("[bold magenta] Benchmark Profiler Summary [/bold magenta]!", ":book:")

    @staticmethod
    def print_metadata() -> None:
        console.print(f"[bold blue] SLURM NODE(s) {os.getenv('SLURM_JOB_NODELIST', '')} [/bold blue]!")
        console.print(f"[bold blue] SLURM JOB ID {os.getenv('SLURM_JOB_ID', '')} [/bold blue]!")
        console.print(f"[bold blue] TIMESTAMP {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M:%S')} [/bold blue]!")

    @rank_zero_only
    def print_benchmark_profiler_report(
        self,
        speed_metrics_df: pd.DataFrame | None = None,
        time_metrics_df: pd.DataFrame | None = None,
        memory_metrics_df: pd.DataFrame | None = None,
        system_metrics_df: pd.DataFrame | None = None,
        model_summary: str | None = None,
    ) -> None:
        self.print_title()
        self.print_metadata()

        if time_metrics_df is not None:
            warnings.warn(
                "INFO: Time Report metrics represent single-node metrics (not multi-node aggregated metrics)",
            )
            warnings.warn("INFO: Metrics with a * symbol, represent the value after aggregating all steps")
            self.print_report("Time Profiling", time_metrics_df, color="green", emoji="alarm_clock")

        if speed_metrics_df is not None:
            warnings.warn(
                "INFO: Speed Report metrics are single-node metrics (not multi-node aggregated metrics)",
            )
            self.print_report("Speed Profiling", speed_metrics_df, color="yellow", emoji="racing_car")

        if memory_metrics_df is not None:
            warnings.warn("INFO: Memory Report metrics represent metrics aggregated across all nodes")
            self.print_report("Memory Profiling", memory_metrics_df, color="purple", emoji="floppy_disk")

        if system_metrics_df is not None:
            self.print_report("System Profiling", system_metrics_df, color="Red", emoji="desktop_computer")

        if model_summary is not None:
            self.print_report("Model Summary", model_summary, color="Orange", emoji="robot")
            num_gpus = self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes
            if num_gpus > 1:
                LOGGER.info("Model Summary displays the stats for a single GPU.")

    @staticmethod
    def write_benchmark_profiler_report() -> None:
        console.save_html("report.html")

    @staticmethod
    def to_df(sample_dict: dict[str, float], precision: str = ".5") -> pd.DataFrame:
        df = pd.DataFrame(sample_dict.items())
        df.columns = ["metric", "value"]
        df.value = df.value.apply(lambda x: f"%{precision}f" % x)
        return df

    @cached_property
    @rank_zero_only
    def speed_profile(self) -> None:
        """Speed profiler Report.

        Get speed metrics from Progress Bar for training and validation.
        """
        if self.config.diagnostics.benchmark_profiler.speed.enabled:
            # Find the first ProfilerProgressBar callback.
            for callback in self.callbacks:
                if isinstance(callback, ProfilerProgressBar):
                    return self.profiler.get_speed_profiler_df(callback)
            else:
                error_msg = "No ProfilerProgressBar callback found."
                raise ValueError(error_msg)
        else:
            return None

    def _get_logger(self) -> dict[str, Logger]:
        if (self.config.diagnostics.log.wandb.enabled) and (not self.config.diagnostics.log.wandb.offline):
            logger_info = {"logger_name": "wandb", "logger": self.wandb_logger}
        elif self.config.diagnostics.log.tensorboard.enabled:
            logger_info = {"logger_name": "tensorboard", "logger": self.tensorboard_logger}
        elif self.config.diagnostics.log.mlflow.enabled:
            logger_info = {"logger_name": "mlflow", "logger": self.mlflow_logger}
        else:
            LOGGER.warning("No logger enabled for system profiler")
            logger_info = None
        return logger_info

    @cached_property
    @rank_zero_only
    def system_profile(self) -> None:
        """System Profiler Report."""
        if self.config.diagnostics.benchmark_profiler.system.enabled:
            logger_info = self._get_logger()
            if logger_info:
                return self.profiler.get_system_profiler_df(
                    logger_name=logger_info["logger_name"],
                    logger=logger_info["logger"],
                )
            LOGGER.warning("System Profiler Report is not available")
            return None
        return None

    @cached_property
    @rank_zero_only
    def memory_profile(self) -> None:
        """Memory Profiler Report."""
        if self.config.diagnostics.benchmark_profiler.memory.enabled:
            return self.profiler.get_memory_profiler_df()
        return None

    @cached_property
    @rank_zero_only
    def time_profile(self) -> None:
        """Time Profiler Report."""
        if self.config.diagnostics.benchmark_profiler.time.enabled:
            return self.profiler.get_time_profiler_df()
        return None

    @cached_property
    def model_summary(self) -> str:
        if self.config.diagnostics.benchmark_profiler.model_summary.enabled:
            model = self.model
            return self.profiler.get_model_summary(model=model, example_input_array=self.example_input_array)
        return None

    @rank_zero_only
    def export_to_logger(self) -> None:
        if (self.config.diagnostics.log.wandb.enabled) and (not self.config.diagnostics.log.wandb.offline):
            self.to_wandb()

        elif self.config.diagnostics.log.mlflow.enabled:
            self.to_mlflow()

    def report(self) -> str:
        """Print report to console."""
        LOGGER.info("Generating Profiler reports")
        self.print_benchmark_profiler_report(
            memory_metrics_df=self.memory_profile,
            time_metrics_df=self.time_profile,
            speed_metrics_df=self.speed_profile,  # speed profile needs to be generated after time and memory reports
            system_metrics_df=self.system_profile,
            model_summary=self.model_summary,
        )

    def _get_extra_files(self) -> None:
        extra_files = []
        extra_files.extend(self.profiler.dirpath.glob("*.pickle"))
        # These trace files are too big to push to MLFlow so
        # we won't push them as artifacts extra_files.extend(self.profiler.dirpath.glob("*.json"))
        return extra_files

    def _log_reports_to_mlflow(self, run_id: str, data: pd.DataFrame, artifact_file: str, report_fname: str) -> None:
        self.mlflow_logger.experiment.log_table(
            run_id=run_id,
            data=data,
            artifact_file=artifact_file,
        )

        self.mlflow_logger.experiment.log_artifact(run_id, report_fname)

    @rank_zero_only
    def to_mlflow(self) -> None:
        """Log report into MLFlow."""
        LOGGER.info("logging to MLFlow Profiler report")
        self.write_benchmark_profiler_report()
        # check this https://stackoverflow.com/questions/71151054/how-to-log- d da-table-of-metrics-into-mlflow

        run_id = self.mlflow_logger.run_id
        if self.config.diagnostics.benchmark_profiler.system.enabled:
            self._log_reports_to_mlflow(
                run_id=run_id,
                data=self.system_profile,
                artifact_file="system_metrics_report.json",
                report_fname=self.profiler.system_report_fname,
            )

        if self.config.diagnostics.benchmark_profiler.time.enabled:
            self._log_reports_to_mlflow(
                run_id=run_id,
                data=self.time_profile,
                artifact_file="time_metrics_reports.json",
                report_fname=self.profiler.time_report_fname,
            )

        if self.config.diagnostics.benchmark_profiler.speed.enabled:
            self._log_reports_to_mlflow(
                run_id=run_id,
                data=self.speed_profile,
                artifact_file="speed_metrics_reports.json",
                report_fname=self.profiler.speed_report_fname,
            )

        if self.config.diagnostics.benchmark_profiler.memory.enabled:
            self._log_reports_to_mlflow(
                run_id=run_id,
                data=self.memory_profile,
                artifact_file="memory_metrics_reports.json",
                report_fname=self.profiler.memory_report_fname,
            )

            extra_files = self._get_extra_files()
            for file in extra_files:
                artifact_path = self.profiler.dirpath / file
                if artifact_path.is_file():
                    self.mlflow_logger.experiment.log_artifact(run_id, artifact_path)

        if self.config.diagnostics.benchmark_profiler.model_summary.enabled:
            self.mlflow_logger.experiment.log_artifact(run_id, self.profiler.model_summary_fname)

    @rank_zero_only
    def to_wandb(self) -> None:
        """Log report into  W&B."""
        LOGGER.info("logging to W&B Profiler report")
        self.write_benchmark_profiler_report()
        import wandb
        from pytorch_lightning.loggers.wandb import WandbLogger

        logger = WandbLogger(
            project=self.run_dict["project"],
            entity=self.run_dict["entity"],
            id=self.run_dict["id"],
            offline=self.config.diagnostics.log.wandb.offline,
            resume=self.run_dict["id"],
        )

        logger.experiment.log({"speed_metrics_report": wandb.Table(dataframe=self.speed_profile)})
        logger.experiment.log({"memory_metrics_report": wandb.Table(dataframe=self.system_profile)})
        logger.experiment.log({"time_metrics_report": wandb.Table(dataframe=self.time_profile)})
        logger.experiment.log({"memory_metrics_report": wandb.Table(dataframe=self.memory_profile)})
        logger.experiment.log({"model_summary_report": wandb.Table(dataframe=self.model_summary)})
        with Path("report.html").open("w") as f:
            logger.experiment.log({"reports_benchmark_profiler": wandb.Html(f)})
        logger.experiment.finish()

    @cached_property
    def callbacks(self) -> list[pl.callbacks.Callback]:
        callbacks = super().callbacks
        callbacks.append(ProfilerProgressBar())
        if self.config.diagnostics.benchmark_profiler.snapshot.enabled:
            from anemoi.training.diagnostics.callbacks.profiler import MemorySnapshotRecorder
            from anemoi.training.diagnostics.profilers import check_torch_version

            available = check_torch_version()

            if available:  # if torch is below 2.1.0, the callback will not be added
                callbacks.append(MemorySnapshotRecorder(self.config))
        return callbacks

    @cached_property
    def datamodule(self) -> AnemoiDatasetsDataModule:
        datamodule = super().datamodule
        # to generate a model summary with shapes we need a sample input array
        batch = next(iter(datamodule.train_dataloader()))
        if type(batch) in [list, tuple]:
            batch = batch[0]
        self.example_input_array = batch[
            :,
            0 : self.config.training.multistep_input,
            ...,
            self.data_indices.data.input.full,
        ]
        # If the input batch is sharded, replicate it to its full size
        if self.config.dataloader.read_group_size > 1:
            self.example_input_array = self.example_input_array.repeat(
                1,
                1,
                1,
                self.config.dataloader.read_group_size,
                1,
            )
        return datamodule

    @cached_property
    def profiler(self) -> BenchmarkProfiler:
        return BenchmarkProfiler(self.config)

    def _update_paths(self) -> None:
        """Update the paths in the configuration."""
        super()._update_paths()

        if self.run_id:  # when using mlflow only rank0 will have a run_id except when resuming runs
            # Multi-gpu new runs or forked runs - only rank 0
            # Multi-gpu resumed runs - all ranks
            self.config.hardware.paths.profiler = Path(self.config.hardware.paths.profiler, self.run_id)
        elif self.config.training.fork_run_id:
            parent_run = self.config.training.fork_run_id
            self.config.hardware.paths.profiler = Path(self.config.hardware.paths.profiler, parent_run)
        LOGGER.info("Profiler path: %s", self.config.hardware.paths.profiler)

    def _close_logger(self) -> None:
        if (self.config.diagnostics.log.wandb.enabled) and (not self.config.diagnostics.log.wandb.offline):
            # We need to close the W&B logger to be able to read the System Metrics
            self.wandb_logger.experiment.finish()

    def profile(self) -> None:
        """Profile the model."""
        self.train()
        self.report()
        self.export_to_logger()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    AnemoiProfiler(config).profile()


if __name__ == "__main__":
    main()
