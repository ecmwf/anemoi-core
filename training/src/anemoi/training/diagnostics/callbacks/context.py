from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class _CheckpointPaths:
    root: Any
    by_frequency: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.by_frequency[key]


@dataclass(frozen=True)
class _OutputPaths:
    checkpoints: _CheckpointPaths
    plots: Any
    profiler: Any


@dataclass(frozen=True)
class _SystemContext:
    output: _OutputPaths


@dataclass(frozen=True)
class _SWASettings:
    enabled: bool
    lr: Any


@dataclass(frozen=True)
class _TrainingContext:
    swa: _SWASettings
    max_epochs: int


@dataclass(frozen=True)
class PlotRuntimeSettings:
    """Runtime settings consumed by plotting callbacks."""

    asynchronous: bool
    projection_kind: str
    backend: str
    datashader: bool
    frequency_batch: int | None
    frequency_epoch: int | None
    async_with_read_group_risk: bool
    save_basedir: Any
    log_wandb_enabled: bool
    log_mlflow_enabled: bool
    global_diagnostic: list[str]
    dataset_diagnostics: dict[str, list[str]]

    @classmethod
    def from_config(cls, config: Any) -> PlotRuntimeSettings:
        plot_cfg = config.diagnostics.plot
        settings = getattr(plot_cfg, "settings", None)
        schedule = getattr(settings, "schedule", None) if settings is not None else None
        frequency = getattr(plot_cfg, "frequency", None)

        asynchronous = getattr(settings, "asynchronous", None)
        if asynchronous is None:
            asynchronous = getattr(plot_cfg, "asynchronous", False)

        projection_kind = getattr(settings, "projection_kind", None)
        if projection_kind is None:
            projection_kind = getattr(plot_cfg, "projection_kind", "equirectangular")

        backend = getattr(settings, "backend", None)
        if backend is None:
            backend = "datashader" if getattr(plot_cfg, "datashader", False) else "matplotlib"
        datashader = backend == "datashader"

        freq_batch = getattr(schedule, "batch", None)
        if freq_batch is None:
            freq_batch = getattr(frequency, "batch", None)

        freq_epoch = getattr(schedule, "epoch", None)
        if freq_epoch is None:
            freq_epoch = getattr(frequency, "epoch", None)

        data_cfg = config.data
        global_diag = list(data_cfg.get("diagnostic", []))

        dataset_diags: dict[str, list[str]] = {}
        for dataset_name, dataset_cfg in data_cfg.datasets.items():
            diagnostics = getattr(dataset_cfg, "diagnostic", None)
            dataset_diags[dataset_name] = [] if diagnostics is None else list(diagnostics)

        read_group_size = config.dataloader.read_group_size
        return cls(
            asynchronous=bool(asynchronous),
            projection_kind=str(projection_kind),
            backend=str(backend),
            datashader=datashader,
            frequency_batch=freq_batch,
            frequency_epoch=freq_epoch,
            async_with_read_group_risk=bool(asynchronous) and read_group_size > 1,
            save_basedir=config.system.output.plots,
            log_wandb_enabled=config.diagnostics.log.wandb.enabled,
            log_mlflow_enabled=config.diagnostics.log.mlflow.enabled,
            global_diagnostic=global_diag,
            dataset_diagnostics=dataset_diags,
        )

    @classmethod
    def from_context(cls, context: Any) -> PlotRuntimeSettings:
        if hasattr(context, "plot_runtime"):
            return context.plot_runtime
        return cls.from_config(context)


@dataclass(frozen=True)
class ProfilerRuntimeSettings:
    """Runtime settings consumed by profiler callbacks."""

    profiler_dirpath: Path
    snapshot_warmup: int
    snapshot_steps: int

    @classmethod
    def from_config(cls, config: Any) -> ProfilerRuntimeSettings:
        return cls(
            profiler_dirpath=Path(config.system.output.profiler),
            snapshot_warmup=config.diagnostics.benchmark_profiler.snapshot.warmup or 0,
            snapshot_steps=config.diagnostics.benchmark_profiler.snapshot.steps,
        )

    @classmethod
    def from_context(cls, context: Any) -> ProfilerRuntimeSettings:
        if hasattr(context, "profiler_runtime"):
            return context.profiler_runtime
        return cls.from_config(context)


@dataclass(frozen=True)
class CallbackContext:
    """Minimal callback configuration view used by callback wiring."""

    diagnostics: Any
    training: _TrainingContext
    system: _SystemContext
    plot_runtime: PlotRuntimeSettings
    profiler_runtime: ProfilerRuntimeSettings
    _source: DictConfig

    @classmethod
    def from_config(cls, config: DictConfig) -> CallbackContext:
        checkpoints = _CheckpointPaths(
            root=config.system.output.checkpoints.root,
            by_frequency={
                "every_n_minutes": config.system.output.checkpoints.every_n_minutes,
                "every_n_epochs": config.system.output.checkpoints.every_n_epochs,
                "every_n_train_steps": config.system.output.checkpoints.every_n_train_steps,
            },
        )
        return cls(
            diagnostics=config.diagnostics,
            training=_TrainingContext(
                swa=_SWASettings(
                    enabled=config.training.swa.enabled,
                    lr=config.training.swa.lr,
                ),
                max_epochs=config.training.max_epochs,
            ),
            system=_SystemContext(
                output=_OutputPaths(
                    checkpoints=checkpoints,
                    plots=config.system.output.plots,
                    profiler=config.system.output.profiler,
                ),
            ),
            plot_runtime=PlotRuntimeSettings.from_config(config),
            profiler_runtime=ProfilerRuntimeSettings.from_config(config),
            _source=config,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibility passthrough during callback migration."""
        return self._source.get(key, default)
