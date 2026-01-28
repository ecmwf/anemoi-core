# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from pathlib import Path  # noqa: TC003 - used in runtime type resolution for Pydantic
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ConfigBase(BaseModel, Mapping[str, Any]):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_assignment=True)

    def __getattr__(self, item: str) -> Any:
        extra = self.__pydantic_extra__ or {}
        if item in extra:
            value = self._wrap_value(extra[item])
            extra[item] = value
            return value
        raise AttributeError(item)

    def __getitem__(self, key: str) -> Any:
        if key in self.model_fields:
            return self._wrap_value(getattr(self, key))
        extra = self.__pydantic_extra__ or {}
        if key in extra:
            return self._wrap_value(extra[key])
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw_items())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._raw_items())

    def _raw_items(self) -> dict[str, Any]:
        data: dict[str, Any] = {name: getattr(self, name) for name in self.model_fields}
        data.update(self.__pydantic_extra__ or {})
        return data

    def keys(self) -> list[str]:
        return list(self._raw_items().keys())

    def items(self) -> list[tuple[str, Any]]:
        return [(key, self[key]) for key in self._raw_items()]

    def values(self) -> list[Any]:
        return [self[key] for key in self._raw_items()]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    @classmethod
    def _wrap_value(cls, value: Any) -> Any:
        if isinstance(value, ConfigBase):
            return value
        if isinstance(value, Mapping):
            if all(isinstance(key, str) for key in value):
                return ConfigNode(**value)
            return {key: cls._wrap_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._wrap_value(item) for item in value]
        return value


class ConfigNode(ConfigBase):
    """Generic config node with attribute access."""


class HardwareConfig(ConfigBase):
    accelerator: str | None = None
    num_nodes: int | None = None
    num_gpus_per_node: int | None = None
    num_gpus_per_model: int | None = None
    num_gpus_per_ensemble: int | None = None


class InputConfig(ConfigBase):
    dataset: str | Path | None = None
    graph: str | Path | None = None
    warm_start: str | Path | None = None


class LogsConfig(ConfigBase):
    root: str | Path | None = None
    wandb: str | Path | None = None
    mlflow: str | Path | None = None
    tensorboard: str | Path | None = None


class CheckpointsConfig(ConfigBase):
    root: str | Path | None = None
    every_n_epochs: str | None = None
    every_n_train_steps: str | None = None
    every_n_minutes: str | None = None


class OutputConfig(ConfigBase):
    root: str | Path | None = None
    plots: str | Path | None = None
    profiler: str | Path | None = None
    logs: LogsConfig | None = None
    checkpoints: CheckpointsConfig | None = None


class SystemConfig(ConfigBase):
    input: InputConfig
    output: OutputConfig
    hardware: HardwareConfig


class DataConfig(ConfigBase):
    format: str | None = None
    timestep: str | None = None
    frequency: str | None = None
    datasets: ConfigNode | None = None
    resolution: str | None = None


class DataloaderConfig(ConfigBase):
    dataset: ConfigNode | str | Path | None = None
    training: ConfigNode
    validation: ConfigNode
    test: ConfigNode
    grid_indices: ConfigNode
    batch_size: DataloaderBatchSizeConfig
    num_workers: DataloaderNumWorkersConfig
    limit_batches: DataloaderLimitBatchesConfig
    pin_memory: bool
    prefetch_factor: int
    read_group_size: int | None = None
    validation_rollout: int | None = None
    model_run_info: DataloaderModelRunInfoConfig | None = None


class GraphConfig(ConfigBase):
    data: str | None = None
    hidden: str | list[str] | None = None
    overwrite: bool | None = None
    nodes: ConfigNode | None = None
    edges: list[ConfigNode] | None = None
    post_processors: list[ConfigNode] | None = None


class ModelConfig(ConfigBase):
    model: ConfigNode | None = None
    output_mask: ConfigNode | None = None
    keep_batch_sharded: bool | None = None
    num_channels: int | None = None
    cpu_offload: bool | None = None
    trainable_parameters: ConfigNode | None = None
    compile: list[ConfigNode] | ConfigNode | None = None
    bounding: list[ConfigNode] | None = None
    encoder: ConfigNode | None = None
    decoder: ConfigNode | None = None
    processor: ConfigNode | None = None
    noise_injector: ConfigNode | None = None
    condition_on_residual: bool | None = None
    residual: ConfigNode | None = None
    layer_kernels: ConfigNode | None = None


class LrConfig(ConfigBase):
    rate: float | None = None
    iterations: int | None = None
    warmup: int | None = None
    min: float | None = None


class RolloutConfig(ConfigBase):
    start: int | None = None
    epoch_increment: int | None = None
    max: int | None = None


class SwaConfig(ConfigBase):
    enabled: bool | None = None
    lr: float | None = None


class ExplicitTimesConfig(ConfigBase):
    input: list[int] | None = None
    target: list[int] | None = None


class TrainingConfig(ConfigBase):
    run_id: str | None = None
    fork_run_id: str | None = None
    transfer_learning: bool
    load_weights_only: bool
    deterministic: bool
    num_sanity_val_steps: int
    max_epochs: int | None = None
    max_steps: int | None = None
    precision: str | None = None
    model_task: str | None = None
    multistep_input: int | None = None
    accum_grad_batches: int | None = None
    loss_gradient_scaling: bool | None = None
    ensemble_size_per_device: int | None = None
    gradient_clip: GradientClipConfig
    training_loss: ConfigNode | None = None
    validation_metrics: ConfigNode | None = None
    scalers: ConfigNode | None = None
    metrics: ConfigNode | None = None
    variable_groups: ConfigNode | None = None
    target_forcing: ConfigNode | None = None
    strategy: ConfigNode | None = None
    optimizer: ConfigNode | None = None
    lr: LrConfig
    rollout: RolloutConfig
    swa: SwaConfig
    explicit_times: ExplicitTimesConfig | None = None
    submodules_to_freeze: list[str] = Field(default_factory=list)
    recompile_limit: int | None = None


class GradientClipConfig(ConfigBase):
    val: float | None = None
    algorithm: str | None = None


class DataloaderBatchSizeConfig(ConfigBase):
    training: int
    validation: int
    test: int


class DataloaderNumWorkersConfig(ConfigBase):
    training: int
    validation: int
    test: int


class DataloaderLimitBatchesConfig(ConfigBase):
    training: int | None = None
    validation: int | None = None
    test: int | None = None


class DataloaderModelRunInfoConfig(ConfigBase):
    start: str | None = None
    length: int | None = None


class DiagnosticsDebugConfig(ConfigBase):
    anomaly_detection: bool | None = None


class DiagnosticsWandbConfig(ConfigBase):
    enabled: bool
    offline: bool
    log_model: bool
    project: str | None = None
    entity: str | None = None
    gradients: bool
    parameters: bool


class DiagnosticsTensorboardConfig(ConfigBase):
    enabled: bool


class DiagnosticsMlflowConfig(ConfigBase):
    enabled: bool
    _target_: str | None = None
    offline: bool
    authentication: bool
    tracking_uri: str | None = None
    experiment_name: str | None = None
    project_name: str | None = None
    system: bool
    terminal: bool
    run_name: str | None = None
    on_resume_create_child: bool
    expand_hyperparams: list[str] | None = None
    http_max_retries: int
    max_params_length: int
    save_dir: str | Path | None = None


class DiagnosticsLogConfig(ConfigBase):
    wandb: DiagnosticsWandbConfig
    tensorboard: DiagnosticsTensorboardConfig
    mlflow: DiagnosticsMlflowConfig
    interval: int


class DiagnosticsProgressBarConfig(ConfigBase):
    _target_: str | None = None
    refresh_rate: int | None = None


class DiagnosticsCheckpointPolicyConfig(ConfigBase):
    save_frequency: int | None = None
    num_models_saved: int | None = None


class DiagnosticsCheckpointConfig(ConfigBase):
    every_n_minutes: DiagnosticsCheckpointPolicyConfig | None = None
    every_n_epochs: DiagnosticsCheckpointPolicyConfig | None = None
    every_n_train_steps: DiagnosticsCheckpointPolicyConfig | None = None


class DiagnosticsPlotFrequencyConfig(ConfigBase):
    batch: int
    epoch: int


class DiagnosticsPlotConfig(ConfigBase):
    asynchronous: bool
    datashader: bool
    frequency: DiagnosticsPlotFrequencyConfig
    parameters: list[str]
    sample_idx: int
    precip_and_related_fields: list[str]
    colormaps: ConfigNode | None = None
    datasets_to_plot: list[str]
    callbacks: list[ConfigNode]


class DiagnosticsBenchmarkMemoryConfig(ConfigBase):
    enabled: bool
    steps: int
    warmup: int
    extra_plots: bool
    trace_rank0_only: bool


class DiagnosticsBenchmarkTimeConfig(ConfigBase):
    enabled: bool
    verbose: bool


class DiagnosticsBenchmarkSpeedConfig(ConfigBase):
    enabled: bool


class DiagnosticsBenchmarkSystemConfig(ConfigBase):
    enabled: bool


class DiagnosticsBenchmarkModelSummaryConfig(ConfigBase):
    enabled: bool


class DiagnosticsBenchmarkSnapshotConfig(ConfigBase):
    enabled: bool
    steps: int
    warmup: int


class DiagnosticsConfig(ConfigBase):
    enable_checkpointing: bool
    enable_progress_bar: bool
    checkpoint: DiagnosticsCheckpointConfig
    log: DiagnosticsLogConfig
    progress_bar: DiagnosticsProgressBarConfig
    debug: DiagnosticsDebugConfig
    plot: DiagnosticsPlotConfig
    callbacks: list[ConfigNode] = Field(default_factory=list)
    benchmark_profiler: DiagnosticsBenchmarkProfilerConfig
    check_val_every_n_epoch: int | None = None
    print_memory_summary: bool | None = None


class DiagnosticsBenchmarkProfilerConfig(ConfigBase):
    memory: DiagnosticsBenchmarkMemoryConfig
    time: DiagnosticsBenchmarkTimeConfig
    speed: DiagnosticsBenchmarkSpeedConfig
    system: DiagnosticsBenchmarkSystemConfig
    model_summary: DiagnosticsBenchmarkModelSummaryConfig
    snapshot: DiagnosticsBenchmarkSnapshotConfig


class Settings(ConfigBase):
    system: SystemConfig
    data: DataConfig
    dataloader: DataloaderConfig
    graph: GraphConfig
    model: ModelConfig
    training: TrainingConfig
    diagnostics: DiagnosticsConfig


def to_container(value: Any) -> Any:
    if isinstance(value, ConfigBase):
        return {key: to_container(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: to_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_container(item) for item in value]
    return value


def get_path(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return default
        current = current.get(part, default) if isinstance(current, Mapping) else getattr(current, part, default)
    return current


__all__ = [
    "CheckpointsConfig",
    "ConfigBase",
    "ConfigNode",
    "DataConfig",
    "DataloaderBatchSizeConfig",
    "DataloaderConfig",
    "DataloaderLimitBatchesConfig",
    "DataloaderModelRunInfoConfig",
    "DataloaderNumWorkersConfig",
    "DiagnosticsBenchmarkMemoryConfig",
    "DiagnosticsBenchmarkModelSummaryConfig",
    "DiagnosticsBenchmarkProfilerConfig",
    "DiagnosticsBenchmarkSnapshotConfig",
    "DiagnosticsBenchmarkSpeedConfig",
    "DiagnosticsBenchmarkSystemConfig",
    "DiagnosticsBenchmarkTimeConfig",
    "DiagnosticsCheckpointConfig",
    "DiagnosticsCheckpointPolicyConfig",
    "DiagnosticsConfig",
    "DiagnosticsDebugConfig",
    "DiagnosticsLogConfig",
    "DiagnosticsMlflowConfig",
    "DiagnosticsPlotConfig",
    "DiagnosticsPlotFrequencyConfig",
    "DiagnosticsProgressBarConfig",
    "DiagnosticsTensorboardConfig",
    "DiagnosticsWandbConfig",
    "ExplicitTimesConfig",
    "GradientClipConfig",
    "GraphConfig",
    "HardwareConfig",
    "InputConfig",
    "LogsConfig",
    "LrConfig",
    "ModelConfig",
    "OutputConfig",
    "RolloutConfig",
    "Settings",
    "SwaConfig",
    "SystemConfig",
    "TrainingConfig",
    "get_path",
    "to_container",
]
