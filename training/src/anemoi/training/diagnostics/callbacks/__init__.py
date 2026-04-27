# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import timedelta

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar

from anemoi.training.diagnostics.callbacks.checkpoint import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks.optimiser import LearningRateMonitor
from anemoi.training.diagnostics.callbacks.plot import PlottingSettings
from anemoi.training.diagnostics.callbacks.provenance import ParentUUIDCallback
from anemoi.training.diagnostics.callbacks.sanity import CheckVariableOrder
from anemoi.training.diagnostics.callbacks.weight_averaging import _get_weight_averaging_callback
from anemoi.training.utils.checkpoint import RegisterMigrations

LOGGER = logging.getLogger(__name__)


def nestedget(config: DictConfig, key: str, default: object) -> object:
    """Get a nested key from a DictConfig object.

    E.g.
    >>> nestedget(config, "diagnostics.log.wandb.enabled", False)
    """
    keys = key.split(".")
    for k in keys:
        config = getattr(config, k, default)
        if not isinstance(config, dict | DictConfig):
            break
    return config


@dataclass
class CallbacksContext:
    """Context containing configuration for callbacks initialisation.

    Parameters
    ----------
    diagnostics : DictConfig
        Diagnostics configuration (config.diagnostics)
    checkpoints_output : DictConfig
        Checkpoint output paths configuration (config.system.output.checkpoints)
    plots_output : str | None
        Base directory for plot outputs (config.system.output.plots)
    wandb_enabled : bool
        Whether Weights & Biases logging is enabled
    mlflow_enabled : bool
        Whether MLflow logging is enabled
    weight_averaging_config : DictConfig | None
        Weight averaging configuration (config.training.weight_averaging), or None if not set
    """

    diagnostics: DictConfig
    checkpoints_output: DictConfig
    plots_output: str | None
    wandb_enabled: bool
    mlflow_enabled: bool
    weight_averaging_config: DictConfig | None = field(default=None)


def _get_checkpoint_callback(diagnostics_cfg: DictConfig, checkpoint_paths_cfg: DictConfig) -> list[AnemoiCheckpoint]:
    """Get checkpointing callbacks.

    Parameters
    ----------
    diagnostics_cfg : DictConfig
        Diagnostics configuration (``config.diagnostics``).
    checkpoint_paths_cfg : DictConfig
        Checkpoint paths configuration (``config.system.output.checkpoints``).
    """
    if not diagnostics_cfg.enable_checkpointing:
        return []

    checkpoint_settings = {
        "dirpath": checkpoint_paths_cfg.root,
        "verbose": False,
        # save weights, optimizer states, LR-schedule states, hyperparameters etc.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
        "save_weights_only": False,
        "auto_insert_metric_name": False,
        # save after every validation epoch, if we've improved
        "save_on_train_epoch_end": False,
        "enable_version_counter": False,
    }

    ckpt_frequency_save_dict = {}

    for key, frequency_dict in diagnostics_cfg.checkpoint.items():
        frequency = frequency_dict.save_frequency
        n_saved = frequency_dict.num_models_saved
        if key == "every_n_minutes" and frequency_dict.save_frequency is not None:
            target = "train_time_interval"
            frequency = timedelta(minutes=frequency_dict.save_frequency)
        else:
            target = key
        ckpt_frequency_save_dict[target] = (
            checkpoint_paths_cfg[key],
            frequency,
            n_saved,
        )

    checkpoint_callbacks = []
    for save_key, (
        name,
        save_frequency,
        save_n_models,
    ) in ckpt_frequency_save_dict.items():
        if save_frequency is not None:
            LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
            checkpoint_callbacks.append(
                # save_top_k: the save_top_k flag can either save the best or the last k checkpoints
                # depending on the monitor flag on ModelCheckpoint.
                # See https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html for reference
                AnemoiCheckpoint(
                    filename=name,
                    save_last=True,
                    **{save_key: save_frequency},
                    # if save_top_k == k, last k models saved; if save_top_k == -1, all models are saved
                    save_top_k=save_n_models,
                    monitor="step",
                    mode="max",
                    **checkpoint_settings,
                ),
            )
        LOGGER.debug("Not setting up a checkpoint callback with %s", save_key)

    return checkpoint_callbacks


def _check_plotting_dependencies(diagnostics_cfg: DictConfig) -> None:
    """Check that all plotting dependencies required by the current config are installed."""
    try:
        import matplotlib as mpl  # noqa: F401
    except ImportError as err:
        msg = (
            "Plotting callbacks are configured but matplotlib is not installed. "
            "Install it with: pip install anemoi-training[plotting]"
        )
        raise ImportError(msg) from err

    if diagnostics_cfg.plot.datashader:
        try:
            import datashader  # noqa: F401
        except ImportError as err:
            msg = (
                "datashader=True is set but datashader is not installed. "
                "Install it with: pip install anemoi-training[plotting]"
            )
            raise ImportError(msg) from err

    spectrum_targets = {"PlotSpectrum"}
    has_spectrum = any(
        any(t in str(getattr(cb, "_target_", "")) for t in spectrum_targets) for cb in diagnostics_cfg.plot.callbacks
    )
    if has_spectrum:
        try:
            import pyshtools  # noqa: F401
        except ImportError as err:
            msg = (
                "PlotSpectrum is configured but pyshtools is not installed. "
                "Install it with: pip install anemoi-training[plotting]"
            )
            raise ImportError(msg) from err

    if diagnostics_cfg.plot.projection_kind == "lambert_conformal":
        try:
            import cartopy  # noqa: F401
        except ImportError as err:
            msg = (
                "projection_kind='lambert_conformal' requires cartopy, which is not installed. "
                "Install it with: pip install anemoi-training[plotting]"
            )
            raise ImportError(msg) from err


def _get_progress_bar_callback(diagnostics_cfg: DictConfig) -> list[Callback]:
    """Get progress bar callback.

    Instantiated from `config.diagnostics.progress_bar`. If not set, defaults to TQDMProgressBar.

    Example config:
        progress_bar:
          _target_: pytorch_lightning.callbacks.TQDMProgressBar
          refresh_rate: 1
          process_position: 0

    Parameters
    ----------
    diagnostics_cfg : DictConfig
        Diagnostics configuration (``config.diagnostics``).

    Returns
    -------
    list[Callback]
        List containing the progress bar callback, or empty list if disabled.
    """
    if not diagnostics_cfg.enable_progress_bar:
        LOGGER.info("Progress bar disabled.")
        return []

    progress_bar_cfg = getattr(diagnostics_cfg, "progress_bar", None)
    if progress_bar_cfg is not None:
        try:
            progress_bar = instantiate(progress_bar_cfg)
            LOGGER.info("Using progress bar: %s", type(progress_bar))
        except InstantiationException:
            LOGGER.warning("Failed to instantiate progress bar callback from config: %s", progress_bar_cfg)
            progress_bar = TQDMProgressBar(refresh_rate=1, process_position=0)
    else:
        LOGGER.info("Using default progress bar: TQDMProgressBar.")
        progress_bar = TQDMProgressBar(refresh_rate=1, process_position=0)

    return [progress_bar]


def get_callbacks(context: CallbacksContext) -> list[Callback]:
    """Setup callbacks for PyTorch Lightning trainer.

    Set `context.diagnostics.callbacks` to a list of callback configurations
    in hydra form.

    E.g.:
    ```
    callbacks:
        - _target_: anemoi.training.diagnostics.callbacks.RolloutEval
          rollout: 1
          frequency: 12
    ```

    Set `context.diagnostics.plot.callbacks` to a list of plot callback configurations
    will only be added if `context.diagnostics.plot.enabled` is set to True.

    Plotting callbacks automatically receive global plotting settings from `context.diagnostics.plot`
    (datashader, projection_kind, asynchronous, save_basedir, colormaps, precip_and_related_fields,
    focus_areas, dataset_names) via the `plotting_settings` parameter.

    User-configurable callbacks (under ``diagnostics.callbacks``) are instantiated verbatim
    via ``hydra.utils.instantiate``. They receive only what is in the config tree — use
    Hydra interpolation (e.g. ``${system.output.plots}``) for config-derived values.
    Runtime-computed values (resolved paths, loggers) are not available here; if a callback
    needs them, add a typed field to :class:`CallbacksContext` and a dedicated helper function.

    Parameters
    ----------
    context : CallbacksContext
        Callbacks context containing diagnostics, output paths, and runtime-extracted settings.

    Returns
    -------
    list[Callback]
        A list of PyTorch Lightning callbacks

    """
    trainer_callbacks: list[Callback] = []
    diagnostics_cfg = context.diagnostics

    # Get Checkpoint callback
    trainer_callbacks.extend(_get_checkpoint_callback(diagnostics_cfg, context.checkpoints_output))

    # User-configurable callbacks — instantiated verbatim from their YAML config.
    # These receive only what is in the config tree. Use Hydra interpolation
    # (e.g. ${system.output.plots}) for values defined elsewhere in the config.
    # If a callback needs a runtime-computed value (resolved path, logger handle, etc.),
    # add a typed field to CallbacksContext and a dedicated helper function here instead.
    trainer_callbacks.extend(instantiate(callback) for callback in diagnostics_cfg.callbacks)

    # Plotting callbacks — instantiated with global plotting settings from diagnostics.plot
    if diagnostics_cfg.plot.callbacks:
        _check_plotting_dependencies(diagnostics_cfg)
        plotting_settings = PlottingSettings.from_plot_config(diagnostics_cfg.plot, context.plots_output)
        for callback_cfg in diagnostics_cfg.plot.callbacks:
            callback_cfg_dict = dict(callback_cfg)
            callback_cfg_dict["plotting_settings"] = plotting_settings
            trainer_callbacks.append(instantiate(callback_cfg_dict))

    # LearningRateMonitor when any experiment logger is active
    if context.wandb_enabled or context.mlflow_enabled:
        trainer_callbacks.append(LearningRateMonitor())

    # Weight averaging callback (SWA, EMA, etc.)
    trainer_callbacks.extend(_get_weight_averaging_callback(context.weight_averaging_config))

    # Progress bar callback
    trainer_callbacks.extend(_get_progress_bar_callback(diagnostics_cfg))

    # Parent UUID callback
    # Check variable order callback
    # Register Migrations callback
    trainer_callbacks.extend(
        (
            ParentUUIDCallback(),
            CheckVariableOrder(),
            RegisterMigrations(),
        ),
    )

    return trainer_callbacks


__all__ = ["CallbacksContext", "get_callbacks", "nestedget"]
