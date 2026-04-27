# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar

from anemoi.training.diagnostics.callbacks.checkpoint import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks.optimiser import LearningRateMonitor
from anemoi.training.diagnostics.callbacks.optimiser import StochasticWeightAveraging
from anemoi.training.diagnostics.callbacks.plot import PlottingSettings
from anemoi.training.diagnostics.callbacks.provenance import ParentUUIDCallback
from anemoi.training.diagnostics.callbacks.sanity import CheckVariableOrder
from anemoi.training.utils.checkpoint import RegisterMigrations

LOGGER = logging.getLogger(__name__)


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
    full_config : DictConfig
        Full configuration object (used for CONFIG_ENABLED_CALLBACKS)
    """

    diagnostics: DictConfig
    checkpoints_output: DictConfig
    plots_output: str | None
    full_config: DictConfig


def nestedget(config: DictConfig, key: str, default: Any) -> Any:
    """Get a nested key from a DictConfig object.

    E.g.
    >>> nestedget(config, "diagnostics.log.wandb.enabled", False)
    """
    keys = key.split(".")
    for k in keys:
        config = getattr(config, k, default)
        if not isinstance(config, BaseModel | dict | DictConfig):
            break
    return config


# Callbacks to add according to flags in the config.
# Each entry: (enable_condition, factory_fn) where factory_fn receives the full config
# and returns a Callback instance. This allows per-callback config extraction.
CONFIG_ENABLED_CALLBACKS: list[
    tuple[list[str] | str | Callable[[DictConfig], bool], Callable[[DictConfig], Callback]]
] = [
    (
        "training.swa.enabled",
        lambda config: StochasticWeightAveraging(
            swa_lrs=config.training.swa.lr,
            max_epochs=config.training.max_epochs,
        ),
    ),
    (
        lambda config: nestedget(config, "diagnostics.log.wandb.enabled", False)
        or nestedget(config, "diagnostics.log.mlflow.enabled", False),
        lambda _config: LearningRateMonitor(),
    ),
]


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


def _get_config_enabled_callbacks(config: DictConfig) -> list[Callback]:
    """Get callbacks that are enabled in the config as according to CONFIG_ENABLED_CALLBACKS."""
    callbacks = []

    def check_key(config: dict, key: str | Iterable[str] | Callable[[DictConfig], bool]) -> bool:
        """Check key in config."""
        if isinstance(key, Callable):
            return key(config)
        if isinstance(key, str):
            return nestedget(config, key, False)
        if isinstance(key, Iterable):
            return all(nestedget(config, k, False) for k in key)
        return nestedget(config, key, False)

    for enable_key, factory in CONFIG_ENABLED_CALLBACKS:
        if check_key(config, enable_key):
            callbacks.append(factory(config))

    return callbacks


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

    User-configurable callbacks (under `diagnostics.callbacks`) are instantiated as-is from YAML.
    Callbacks added programmatically (see CONFIG_ENABLED_CALLBACKS and the fixed callbacks at
    the bottom of this function) receive only the specific values they need.

    Some callbacks are added by default, depending on the configuration.
    See CONFIG_ENABLED_CALLBACKS for more information.

    Parameters
    ----------
    context : CallbacksContext
        Callbacks context containing diagnostics, output paths, and full config

    Returns
    -------
    list[Callback]
        A list of PyTorch Lightning callbacks

    """
    trainer_callbacks: list[Callback] = []
    diagnostics_cfg = context.diagnostics

    # Get Checkpoint callback
    trainer_callbacks.extend(_get_checkpoint_callback(diagnostics_cfg, context.checkpoints_output))

    # Base callbacks — instantiated directly from their own YAML-specified parameters
    trainer_callbacks.extend(instantiate(callback) for callback in diagnostics_cfg.callbacks)

    # Plotting callbacks — instantiated with global plotting settings from diagnostics.plot
    plotting_settings = PlottingSettings(
        datashader=diagnostics_cfg.plot.datashader,
        projection_kind=diagnostics_cfg.plot.projection_kind,
        asynchronous=diagnostics_cfg.plot.asynchronous,
        save_basedir=context.plots_output,
        colormaps=getattr(diagnostics_cfg.plot, "colormaps", None),
        precip_and_related_fields=getattr(diagnostics_cfg.plot, "precip_and_related_fields", None),
        focus_areas=getattr(diagnostics_cfg.plot, "focus_areas", None),
        dataset_names=getattr(diagnostics_cfg.plot, "datasets_to_plot", None),
    )
    for callback_cfg in diagnostics_cfg.plot.callbacks:
        # Merge global plotting settings into callback config
        callback_cfg_dict = dict(callback_cfg)
        callback_cfg_dict["plotting_settings"] = plotting_settings
        trainer_callbacks.append(instantiate(callback_cfg_dict))

    # Extend with config enabled callbacks
    trainer_callbacks.extend(_get_config_enabled_callbacks(context.full_config))

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


__all__ = ["CallbacksContext", "get_callbacks"]
