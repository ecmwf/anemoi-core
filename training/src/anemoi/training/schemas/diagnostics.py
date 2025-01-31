# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from typing import Any
from typing import Literal
from typing import Union

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from .utils import BaseModel

LOGGER = logging.getLogger(__name__)


class LongRolloutPlotsSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.LongRolloutPlots"] = Field(alias="_target_")
    "LongRolloutPlots object from anemoi training diagnostics callbacks."
    rollout: list[int]
    "Rollout steps to plot at."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    video_rollout: int = Field(default=0)
    "Number of rollout steps for video, by default 0 (no video)."
    accumulation_levels_plot: list[float] | None = Field(default=None)
    "Accumulation levels to plot, by default None."
    cmap_accumulation: list[str] | None = Field(default=None)
    "Colors of the accumulation levels, by default None."
    per_sample: int = Field(default=6)
    "Number of plots per sample, by default 6."
    every_n_epochs: int = Field(default=1)
    "Epoch frequency to plot at, by default 1."
    animation_interval: int = Field(default=400)
    "Delay between frames in the animation in milliseconds, by default 400."


class GraphTrainableFeaturesPlotSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.GraphTrainableFeaturesPlot"] = Field(alias="_target_")
    "GraphTrainableFeaturesPlot object from anemoi training diagnostics callbacks."
    every_n_epochs: int | None
    "Epoch frequency to plot at."


class PlotLossSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotLoss"] = Field(alias="_target_")
    "PlotLoss object from anemoi training diagnostics callbacks."
    parameter_groups: dict[str, list[str]]
    "Dictionary with parameter groups with parameter names as key."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at."


class PlotSampleSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotSample"] = Field(alias="_target_")
    "PlotSample object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    accumulation_levels_plot: list[float]
    "Accumulation levels to plot."
    cmap_accumulation: list[str]
    "Colors of the accumulation levels."
    precip_and_related_fields: list[str] | None = Field(default=None)
    "List of precipitation related fields, by default None."
    per_sample: int = Field(default=6)
    "Number of plots per sample, by default 6."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."


class PlotSpectrumSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotSpectrum"] = Field(alias="_target_")
    "PlotSpectrum object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."


class PlotHistogramSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotHistogram"] = Field(alias="_target_")
    "PlotHistogram object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    precip_and_related_fields: list[str] | None = Field(default=None)
    "List of precipitation related fields, by default None."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."


PlotCallbacks = Union[
    LongRolloutPlotsSchema
    | GraphTrainableFeaturesPlotSchema
    | PlotLossSchema
    | PlotSampleSchema
    | PlotSpectrumSchema
    | PlotHistogramSchema
]


class PlotSchema(BaseModel):
    asynchronous: bool
    "Handle plotting tasks without blocking the model training."
    datashader: bool
    "Use Datashader to plot."
    frequency: PlottingFrequency
    "Frequency of the plotting."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    precip_and_related_fields: list[str]
    "List of precipitation related fields from the parameters list."
    callbacks: list[PlotCallbacks] = Field(default=[])
    "List of plotting functions to call."


class PlottingFrequency(BaseModel):
    batch: PositiveInt = Field(default=750)
    "Frequency of the plotting in number of batches."
    epoch: PositiveInt = Field(default=5)
    "Frequency of the plotting in number of epochs."


class Debug(BaseModel):
    anomaly_detection: bool
    "Activate anomaly detection. This will detect and trace back NaNs/Infs, but slow down training."


class CheckpointSchema(BaseModel):
    save_frequency: int | None
    "Frequency at which to save the checkpoints."
    num_models_saved: int
    "Number of model checkpoint to save. Only the last num_models_saved checkpoints will be kept. \
            If set to -1, all checkpoints are kept"


class WandbSchema(BaseModel):
    enabled: bool
    "Use Weights & Biases logger."
    offline: bool
    "Run W&B offline."
    log_model: bool | Literal["all"]
    "Log checkpoints created by ModelCheckpoint as W&B artifacts. \
            If True, checkpoints are logged at the end of training. If 'all', checkpoints are logged during training."
    project: str
    "The name of the project to which this run will belong."
    gradients: bool
    "Whether to log the gradients."
    parameters: bool
    "Whether to log the hyper parameters."
    entity: str | None = None
    "Username or team name where to send runs. This entity must exist before you can send runs there."


class MlflowSchema(BaseModel):
    enabled: bool
    "Use MLflow logger."
    offline: bool
    "Run MLflow offline. Necessary if no internet access available."
    authentication: bool
    "Whether to authenticate with server or not"
    log_model: bool | Literal["all"]
    "Log checkpoints created by ModelCheckpoint as MLFlow artifacts. \
            If True, checkpoints are logged at the end of training. If 'all', checkpoints are logged during training."
    tracking_uri: str | None
    "Address of local or remote tracking server."
    experiment_name: str
    "Name of experiment."
    project_name: str
    "Name of project."
    system: bool
    "Activate system metrics."
    terminal: bool
    "Log terminal logs to MLflow."
    run_name: str | None
    "Name of run."
    on_resume_create_child: bool
    "Whether to create a child run when resuming a run."
    expand_hyperparams: list[str] = Field(default_factory=lambda: ["config"])
    "Keys to expand within params. Any key being expanded will have lists converted according to `expand_iterables`."
    http_max_retries: PositiveInt = Field(default=35)
    "Specifies the maximum number of retries for MLflow HTTP requests, default 35."


class TensorboardSchema(BaseModel):
    enabled: bool
    "Use TensorBoard logger."


class LoggingSchema(BaseModel):
    wandb: WandbSchema
    "W&B logging schema."
    tensorboard: TensorboardSchema
    "TensorBorad logging schema."
    mlflow: MlflowSchema
    "MLflow logging schema."
    interval: PositiveInt
    "Logging frequency in batches."


class MemorySchema(BaseModel):
    enabled: bool = Field(default=False)
    "Enable memory report. Default to false."
    steps: PositiveInt = Field(default=5)
    "Frequency of memory profiling. Default to 5."
    warmup: NonNegativeInt = Field(default=2)
    "Number of step to discard before the profiler starts to record traces. Default to 2."
    extra_plots: bool = Field(default=False)
    "Save plots produced with torch.cuda._memory_viz.profile_plot if available. Default to false."
    trace_rank0_only: bool = Field(default=False)
    "Trace only rank 0 from SLURM_PROC_ID. Default to false."


class Snapshot(BaseModel):
    enabled: bool = Field(default=False)
    "Enable memory snapshot recording. Default to false."
    steps: PositiveInt = Field(default=4)
    "Frequency of snapshot. Default to 4."
    warmup: NonNegativeInt = Field(default=0)
    "Number of step to discard before the profiler starts to record traces. Default to 0."


class Profiling(BaseModel):
    enabled: bool = Field(default=False)
    "Enable component profiler. Default to false."
    verbose: bool | None = None
    "Set to true to include the full list of profiled action or false to keep it concise."


class BenchmarkProfilerSchema(BaseModel):
    memory: MemorySchema = Field(default_factory=lambda: MemorySchema())
    "Schema for memory report containing metrics associated with CPU and GPU memory allocation."
    time: Profiling = Field(default_factory=lambda: Profiling(True))
    "Report with metrics of execution time for certain steps across the code."
    speed: Profiling = Field(default_factory=lambda: Profiling(True))
    "Report with metrics of execution speed at training and validation time."
    system: Profiling = Field(default_factory=lambda: Profiling())
    "Report with metrics of GPU/CPU usage, memory and disk usage and total execution time."
    model_summary: Profiling = Field(default_factory=lambda: Profiling())
    "Table summary of layers and parameters of the model."
    snapshot: Snapshot = Field(default_factory=lambda: Snapshot())
    "Memory snapshot if torch.cuda._record_memory_history is available."


class DiagnosticsSchema(BaseModel):
    plot: PlotSchema | None = None
    "Plot schema."
    callbacks: Any = Field(default=[])
    "Callbacks schema."
    benchmark_profiler: BenchmarkProfilerSchema
    "Benchmark profiler schema for `profile` command."
    debug: Debug
    "Debug schema."
    profiler: bool
    "Activate the pytorch profiler and tensorboard logger."
    log: LoggingSchema
    "Log schema."
    enable_progress_bar: bool
    "Activate progress bar."
    print_memory_summary: bool
    "Print the memory summary."
    enable_checkpointing: bool
    "Allow model to save checkpoints."
    checkpoint: dict[str, CheckpointSchema] = Field(default_factory=dict)
    "Checkpoint schema for defined frequency (every_n_minutes, every_n_epochs, ...)."
