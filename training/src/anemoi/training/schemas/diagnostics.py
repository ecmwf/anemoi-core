# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import ValidationError
from pydantic import model_validator
from pydantic import root_validator

from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas import PydanticBaseModel

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
    video_rollout: int = Field(example=0)
    "Number of rollout steps for video, by default 0 (no video)."
    accumulation_levels_plot: Union[list[float], None] = Field(default=None)
    "Accumulation levels to plot, by default None."
    cmap_accumulation: Union[list[str], None] = Field(default=None)
    "Colors of the accumulation levels. Default to None. Kept for backward compatibility."
    per_sample: Union[int, None] = Field(default=None)
    "Number of plots per sample, by default 6."
    every_n_epochs: int = Field(example=1)
    "Epoch frequency to plot at, by default 1."
    animation_interval: Union[int, None] = Field(default=None)
    "Delay between frames in the animation in milliseconds, by default 400."
    colormaps: Union[dict[str, ColormapSchema], None] = Field(default=None)
    "List of colormaps to use, by default None."


class GraphTrainableFeaturesPlotSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.GraphTrainableFeaturesPlot"] = Field(alias="_target_")
    "GraphTrainableFeaturesPlot object from anemoi training diagnostics callbacks."
    every_n_epochs: Union[int, None]
    "Epoch frequency to plot at."


class PlotLossSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotLoss"] = Field(alias="_target_")
    "PlotLoss object from anemoi training diagnostics callbacks."
    parameter_groups: dict[str, list[str]]
    "Dictionary with parameter groups with parameter names as key."
    every_n_batches: Union[int, None] = Field(default=None)
    "Batch frequency to plot at."


class MatplotlibColormapSchema(BaseModel):
    target_: Literal["anemoi.training.utils.custom_colormaps.MatplotlibColormap"] = Field(..., alias="_target_")
    "CustomColormap object from anemoi training utils."
    name: str
    "Name of the Matplotlib colormap."
    variables: Union[list[str], None] = Field(default=None)
    "A list of strings representing the variables for which the colormap is used, by default None."


class MatplotlibColormapClevelsSchema(BaseModel):
    target_: Literal["anemoi.training.utils.custom_colormaps.MatplotlibColormapClevels"] = Field(..., alias="_target_")
    "CustomColormap object from anemoi training utils."
    clevels: list
    "The custom color levels for the colormap."
    variables: Union[list[str], None] = Field(default=None)
    "A list of strings representing the variables for which the colormap is used, by default None."


class DistinctipyColormapSchema(BaseModel):
    target_: Literal["anemoi.training.utils.custom_colormaps.DistinctipyColormap"] = Field(..., alias="_target_")
    "CustomColormap object from anemoi training utils."
    n_colors: int
    "The number of colors in the colormap."
    variables: Union[list[str], None] = Field(default=None)
    "A list of strings representing the variables for which the colormap is used, by default None."
    colorblind_type: Union[str, None] = Field(default=None)
    "The type of colorblindness to simulate. If None, the default colorblindness from distinctipy is applied."


ColormapSchema = Annotated[
    Union[MatplotlibColormapSchema, MatplotlibColormapClevelsSchema, DistinctipyColormapSchema],
    Field(discriminator="target_"),
]


class PlotSampleSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotSample"] = Field(alias="_target_")
    "PlotSample object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    accumulation_levels_plot: list[float]
    "Accumulation levels to plot."
    cmap_accumulation: Union[list[str], None] = Field(default=None)
    "Colors of the accumulation levels. Default to None. Kept for backward compatibility."
    precip_and_related_fields: Union[list[str], None] = Field(default=None)
    "List of precipitation related fields, by default None."
    per_sample: int = Field(example=6)
    "Number of plots per sample, by default 6."
    every_n_batches: Union[int, None] = Field(default=None)
    "Batch frequency to plot at, by default None."
    colormaps: Union[dict[str, ColormapSchema], None] = Field(default=None)
    "List of colormaps to use, by default None."


class PlotSpectrumSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotSpectrum"] = Field(alias="_target_")
    "PlotSpectrum object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    every_n_batches: Union[int, None] = Field(default=None)
    "Batch frequency to plot at, by default None."


class PlotHistogramSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.plot.PlotHistogram"] = Field(alias="_target_")
    "PlotHistogram object from anemoi training diagnostics callbacks."
    sample_idx: int
    "Index of sample to plot, must be inside batch size."
    parameters: list[str]
    "List of parameters to plot."
    precip_and_related_fields: Union[list[str], None] = Field(default=None)
    "List of precipitation related fields, by default None."
    every_n_batches: Union[int, None] = Field(default=None)
    "Batch frequency to plot at, by default None."


PlotCallbacks = Annotated[
    Union[
        LongRolloutPlotsSchema,
        GraphTrainableFeaturesPlotSchema,
        PlotLossSchema,
        PlotSampleSchema,
        PlotSpectrumSchema,
        PlotHistogramSchema,
    ],
    Field(discriminator="target_"),
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
    colormaps: dict[str, ColormapSchema] = Field(default_factory=dict)
    "List of colormaps to use."
    callbacks: list[PlotCallbacks] = Field(example=[])
    "List of plotting functions to call."


class PlottingFrequency(BaseModel):
    batch: PositiveInt = Field(example=750)
    "Frequency of the plotting in number of batches."
    epoch: PositiveInt = Field(example=5)
    "Frequency of the plotting in number of epochs."


class TimeLimitSchema(BaseModel):
    target_: Literal["anemoi.training.diagnostics.callbacks.stopping.TimeLimit"] = Field(alias="_target_")
    "TimeLimit object from anemoi training diagnostics callbacks."
    limit: Union[int, str]
    "Time limit, if int, assumed to be hours, otherwise must be a string with units (e.g. '1h', '30m')."
    record_file: Union[str, None] = Field(default=None)
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
    stopping_threshold: Optional[float] = None
    "Stop training immediately once the monitored quantity reaches this threshold."
    divergence_threshold: Optional[float] = None
    "Stop training as soon as the monitored quantity becomes worse than this threshold.."
    check_on_train_epoch_end: Optional[bool] = None
    "Whether to check the stopping criteria at the end of each training epoch."


class Debug(BaseModel):
    anomaly_detection: bool
    "Activate anomaly detection. This will detect and trace back NaNs/Infs, but slow down training."


class CheckpointSchema(BaseModel):
    save_frequency: Union[int, None]
    "Frequency at which to save the checkpoints."
    num_models_saved: int
    "Number of model checkpoint to save. Only the last num_models_saved checkpoints will be kept. \
            If set to -1, all checkpoints are kept"


class WandbSchema(BaseModel):
    enabled: bool
    "Use Weights & Biases logger."
    offline: bool
    "Run W&B offline."
    log_model: Union[bool, Literal["all"]]
    "Log checkpoints created by ModelCheckpoint as W&B artifacts. \
            If True, checkpoints are logged at the end of training. If 'all', checkpoints are logged during training."
    project: str
    "The name of the project to which this run will belong."
    gradients: bool
    "Whether to log the gradients."
    parameters: bool
    "Whether to log the hyper parameters."
    entity: Union[str, None] = None
    "Username or team name where to send runs. This entity must exist before you can send runs there."

    @root_validator(pre=True)
    def clean_entity(self, values: dict) -> dict:
        if values["enabled"] is False:
            values["entity"] = None
        return values


class MlflowSchema(PydanticBaseModel):
    class Config:
        """Pydantic BaseModel configuration."""

        use_attribute_docstrings = True
        use_enum_values = True
        validate_assignment = True
        validate_default = True
        extra = "allow"  # Beware this allows extra fields in the config, typos are less likely to be spotted

    enabled: bool
    "Use MLflow logger."
    offline: bool
    "Run MLflow offline. Necessary if no internet access available."
    authentication: bool
    "Whether to authenticate with server or not"
    log_model: Union[bool, Literal["all"]]
    "Log checkpoints created by ModelCheckpoint as MLFlow artifacts. \
            If True, checkpoints are logged at the end of training. If 'all', checkpoints are logged during training."
    tracking_uri: Union[str, None]
    "Address of local or remote tracking server."
    experiment_name: str
    "Name of experiment."
    project_name: str
    "Name of project."
    system: bool
    "Activate system metrics."
    terminal: bool
    "Log terminal logs to MLflow."
    run_name: Union[str, None]
    "Name of run."
    on_resume_create_child: bool
    "Whether to create a child run when resuming a run."
    expand_hyperparams: list[str] = Field(default_factory=lambda: ["config"])
    "Keys to expand within params. Any key being expanded will have lists converted according to `expand_iterables`."
    http_max_retries: PositiveInt = Field(example=35)
    "Specifies the maximum number of retries for MLflow HTTP requests, default 35."

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        allowed_extras = {"max_params_length": int}
        # "Specifies the maximum number of params to be logged to the MLflow server."
        for extra_field in self.__pydantic_extra__:
            if extra_field not in allowed_extras:
                msg = f"Extra field {extra_field} not allowed for MlflowSchema."
                raise ValidationError(msg)
            if isinstance(extra_field, allowed_extras[extra_field]):
                msg = f"Extra field {extra_field} should be of type {allowed_extras[extra_field]}."
                raise ValidationError(msg)

        return self


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
    verbose: Union[bool, None] = None
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
    plot: Union[PlotSchema, None] = None
    "Plot schema."
    callbacks: list = Field(default_factory=list, example=[])
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
