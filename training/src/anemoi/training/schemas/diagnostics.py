# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from typing import Annotated
from typing import Any
from typing import Literal

from omegaconf import OmegaConf
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import model_validator
from pydantic import root_validator

from anemoi.training.diagnostics.mlflow import MAX_PARAMS_LENGTH
from anemoi.utils.schemas import BaseModel

LOGGER = logging.getLogger(__name__)


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


class MatplotlibColormapSchema(BaseModel):
    target_: Literal["anemoi.training.utils.custom_colormaps.MatplotlibColormap"] = Field(..., alias="_target_")
    "CustomColormap object from anemoi training utils."
    name: str
    "Name of the Matplotlib colormap."
    variables: list[str] | None = Field(default=None)
    "A list of strings representing the variables for which the colormap is used, by default None."


class MatplotlibColormapClevelsSchema(BaseModel):
    target_: Literal["anemoi.training.utils.custom_colormaps.MatplotlibColormapClevels"] = Field(..., alias="_target_")
    "CustomColormap object from anemoi training utils."
    clevels: list
    "The custom color levels for the colormap."
    variables: list[str] | None = Field(default=None)
    "A list of strings representing the variables for which the colormap is used, by default None."


class DistinctipyColormapSchema(BaseModel):
    target_: Literal["anemoi.training.utils.custom_colormaps.DistinctipyColormap"] = Field(..., alias="_target_")
    "CustomColormap object from anemoi training utils."
    n_colors: int
    "The number of colors in the colormap."
    variables: list[str] | None = Field(default=None)
    "A list of strings representing the variables for which the colormap is used, by default None."
    colorblind_type: str | None = Field(default=None)
    "The type of colorblindness to simulate. If None, the default colorblindness from distinctipy is applied."


ColormapSchema = Annotated[
    MatplotlibColormapSchema | MatplotlibColormapClevelsSchema | DistinctipyColormapSchema,
    Field(discriminator="target_"),
]


class LongRolloutPlotsSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.LongRolloutPlots"] = Field(alias="_target_")
    "LongRolloutPlots object from anemoi training diagnostics callbacks."
    rollout: list[int]
    "Rollout steps to plot at."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    video_rollout: int = Field(example=0)
    "Number of rollout steps for video, by default 0 (no video)."
    accumulation_levels_plot: list[float] | None = Field(default=None)
    "Accumulation levels to plot, by default None."
    cmap_accumulation: list[str] | None = Field(default=None)
    "Colors of the accumulation levels. Default to None. Kept for backward compatibility."
    per_sample: int | None = Field(default=None)
    "Number of plots per sample, by default 6."
    every_n_epochs: int = Field(example=1)
    "Epoch frequency to plot at, by default 1."
    animation_interval: int | None = Field(default=None)
    "Delay between frames in the animation in milliseconds, by default 400."
    colormaps: dict[str, ColormapSchema] | None = Field(default=None)
    "List of colormaps to use, by default None."


class PlotSampleSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotSample"] = Field(alias="_target_")
    "PlotSample object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    accumulation_levels_plot: list[float]
    "Accumulation levels to plot."
    cmap_accumulation: list[str] | None = Field(default=None)
    "Colors of the accumulation levels. Default to None. Kept for backward compatibility."
    precip_and_related_fields: list[str] | None = Field(default=None)
    "List of precipitation related fields, by default None."
    per_sample: int = Field(example=6)
    "Number of plots per sample, by default 6."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."
    colormaps: dict[str, ColormapSchema] | None = Field(default=None)
    "List of colormaps to use, by default None."


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


class PlotEnsSampleSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot_ens.PlotEnsSample"] = Field(alias="_target_")
    "PlotEnsSample object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    accumulation_levels_plot: list[float]
    "Accumulation levels to plot."
    cmap_accumulation: list[str] | None = Field(default=None)
    "Colors of the accumulation levels. Default to None. Kept for backward compatibility."
    precip_and_related_fields: list[str] | None = Field(default=None)
    "List of precipitation related fields, by default None."
    per_sample: int = Field(example=6)
    "Number of plots per sample, by default 6."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."
    colormaps: dict[str, ColormapSchema] | None = Field(default=None)
    "List of colormaps to use, by default None."
    members: list[int] | None = Field(default=None)
    "List of ensemble members to plot. If None, plots all members."


class PlotEnsLossSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot_ens.PlotLoss"] = Field(alias="_target_")
    "PlotLoss object from anemoi training diagnostics callbacks."
    parameter_groups: dict[str, list[str]]
    "Dictionary with parameter groups with parameter names as key."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at."


class PlotEnsSpectrumSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot_ens.PlotSpectrum"] = Field(alias="_target_")
    "PlotSpectrum object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."


class PlotEnsHistogramSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot_ens.PlotHistogram"] = Field(alias="_target_")
    "PlotHistogram object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    precip_and_related_fields: list[str] | None = Field(default=None)
    "List of precipitation related fields, by default None."
    every_n_batches: int | None = Field(default=None)
    "Batch frequency to plot at, by default None."


class GraphTrainableFeaturesPlotEnsSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot_ens.GraphTrainableFeaturesPlot"] = Field(
        alias="_target_",
    )
    "GraphTrainableFeaturesPlot object from anemoi training diagnostics callbacks."
    every_n_epochs: int | None
    "Epoch frequency to plot at."


PlotCallbacks = Annotated[
    LongRolloutPlotsSchema
    | GraphTrainableFeaturesPlotSchema
    | PlotLossSchema
    | PlotSampleSchema
    | PlotSpectrumSchema
    | PlotHistogramSchema
    | PlotEnsSampleSchema
    | PlotEnsLossSchema
    | PlotEnsSpectrumSchema
    | PlotEnsHistogramSchema
    | GraphTrainableFeaturesPlotEnsSchema,
    Field(discriminator="target_"),
]


class PlottingFrequency(BaseModel):
    batch: PositiveInt = Field(example=750)
    "Frequency of the plotting in number of batches."
    epoch: PositiveInt = Field(example=5)
    "Frequency of the plotting in number of epochs."


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
    colormaps: dict[str, ColormapSchema] = Field(default_factory=dict)
    "List of colormaps to use."
    callbacks: list[PlotCallbacks] = Field(example=[])
    "List of plotting functions to call."


class TimeLimitSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.stopping.TimeLimit"] = Field(alias="_target_")
    "TimeLimit object from anemoi training diagnostics callbacks."
    limit: int | str
    "Time limit, if int, assumed to be hours, otherwise must be a string with units (e.g. '1h', '30m')."
    record_file: str | None = Field(default=None)
    "File to record the last checkpoint to on exit, if set."


class EarlyStoppingSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.stopping.EarlyStopping"] = Field(alias="_target_")
    monitor: str = Field(examples=["val_wmse_epoch", "val_wmse/sfc_2t/1"])
    "Metric to monitor"
    min_delta: float = 0.0
    "Minimum change in the monitored quantity to qualify as an improvement."
    patience: int = 3
    "Number of epochs with no improvement after which training will be stopped."
    verbose: bool = False
    "If True, prints a message for each improvement."
    mode: Literal["min", "max"] = "min"
    "One of {'min', 'max'}, changes if minimisation or maximimisation of the metric is 'good'."
    strict: bool = True
    "Whether to crash the training if the monitored quantity is not found."
    check_finite: bool = True
    "Whether to check for NaNs and Infs in the monitored quantity."
    stopping_threshold: float | None = None
    "Stop training immediately once the monitored quantity reaches this threshold."
    divergence_threshold: float | None = None
    "Stop training as soon as the monitored quantity becomes worse than this threshold.."
    check_on_train_epoch_end: bool | None = None
    "Whether to check the stopping criteria at the end of each training epoch."


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

    @root_validator(pre=True)
    def clean_entity(cls: type["WandbSchema"], values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        if values["enabled"] is False:
            values["entity"] = None
        return values


class MlflowSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"] = Field(
        default="anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger",
        alias="_target_",
    )
    enabled: bool
    "Use MLflow logger."
    offline: bool
    "Run MLflow offline. Necessary if no internet access available."
    authentication: bool
    "Whether to authenticate with server or not"
    log_model: bool | Literal["all"] | None = None
    "Log checkpoints created by ModelCheckpoint as MLFlow artifacts. \
            If True, checkpoints are logged at the end of training. If 'all', checkpoints are logged during training."
    tracking_uri: str | None = None
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
    http_max_retries: PositiveInt = Field(example=35)
    "Specifies the maximum number of retries for MLflow HTTP requests, default 35."
    max_params_length: int = MAX_PARAMS_LENGTH
    "Maximum number of hpParams to be logged with mlflow"
    save_dir: str | None = None
    "Directory to save logs to when offline=True, default=config.hardware.paths.logs.mlflow"

    @root_validator(pre=True)
    def clean_entity(cls: type["MlflowSchema"], values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        if values["enabled"] is False:
            values["tracking_uri"] = None
        return values


class AzureMlflowSchema(MlflowSchema):
    target_: Literal["anemoi.training.diagnostics.mlflow.azureml.AnemoiAzureMLflowLogger"] = Field(
        ...,
        alias="_target_",
    )

    # These options are inherited, but either don't't make sense or don't work for Azure
    # so we enforce the required value
    offline: Literal[False]
    terminal: Literal[False]
    # These are specific to Azure
    identity: str | None = None
    "Type of identity to use for logging in with Azure ML."
    resource_group: str | None = None
    "Name of the AzureML resource group"
    workspace_name: str | None = None
    "Name of the AzureML workspace"
    subscription_id: str | None = None
    "AzureML subscription ID"
    azure_log_level: str = "WARNING"
    "Log level for all azure packages (azure-identity, azure-core, etc)"


class TensorboardSchema(BaseModel):
    enabled: bool
    "Use TensorBoard logger."


class LoggingSchema(BaseModel):
    wandb: WandbSchema
    "W&B logging schema."
    tensorboard: TensorboardSchema
    "TensorBorad logging schema."
    mlflow: Annotated[MlflowSchema | AzureMlflowSchema, Field(discriminator="target_")]
    "MLflow logging schema."
    interval: PositiveInt
    "Logging frequency in batches."

    @model_validator(mode="before")
    def inject_default_target(cls: type["MlflowSchema"], values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        mlflow_cfg = OmegaConf.to_container(values.get("mlflow"), resolve=True)
        if mlflow_cfg is not None and isinstance(mlflow_cfg, dict) and "_target_" not in mlflow_cfg:
            mlflow_cfg["_target_"] = "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"
            values["mlflow"] = mlflow_cfg
        return values


class MemorySchema(BaseModel):
    enabled: bool = Field(example=False)
    "Enable memory report. Default to false."
    steps: PositiveInt = Field(example=5)
    "Frequency of memory profiling. Default to 5."
    warmup: NonNegativeInt = Field(example=2)
    "Number of step to discard before the profiler starts to record traces. Default to 2."
    extra_plots: bool = Field(example=False)
    "Save plots produced with torch.cuda._memory_viz.profile_plot if available. Default to false."
    trace_rank0_only: bool = Field(example=False)
    "Trace only rank 0 from SLURM_PROC_ID. Default to false."


class Snapshot(BaseModel):
    enabled: bool = Field(example=False)
    "Enable memory snapshot recording. Default to false."
    steps: PositiveInt = Field(example=4)
    "Frequency of snapshot. Default to 4."
    warmup: NonNegativeInt = Field(example=0)
    "Number of step to discard before the profiler starts to record traces. Default to 0."


class Profiling(BaseModel):
    enabled: bool = Field(example=False)
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
    callbacks: list = Field(default_factory=list, example=[])
    "Callbacks schema."
    benchmark_profiler: BenchmarkProfilerSchema
    "Benchmark profiler schema for `profile` command."
    debug: Debug
    "Debug schema."
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
    check_val_every_n_epoch: PositiveInt = Field(default=1, example=1)
    "Run validation every n epochs."
