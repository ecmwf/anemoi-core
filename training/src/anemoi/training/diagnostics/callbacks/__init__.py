# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from datetime import timedelta
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar

from anemoi.training.diagnostics.callbacks.checkpoint import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks.context import CallbackContext
from anemoi.training.diagnostics.callbacks.optimiser import LearningRateMonitor
from anemoi.training.diagnostics.callbacks.optimiser import StochasticWeightAveraging
from anemoi.training.diagnostics.callbacks.provenance import ParentUUIDCallback
from anemoi.training.diagnostics.callbacks.sanity import CheckVariableOrder
from anemoi.training.utils.checkpoint import RegisterMigrations

LOGGER = logging.getLogger(__name__)


def _get_checkpoint_callback(
    enable_checkpointing: bool,
    checkpoint_config: Any,
    checkpoint_paths: Any,
) -> list[AnemoiCheckpoint]:
    """Get checkpointing callbacks."""
    if not enable_checkpointing:
        return []

    checkpoint_settings = {
        "dirpath": checkpoint_paths.root,
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

    for key, frequency_dict in checkpoint_config.items():
        frequency = frequency_dict.save_frequency
        n_saved = frequency_dict.num_models_saved
        if key == "every_n_minutes" and frequency_dict.save_frequency is not None:
            target = "train_time_interval"
            frequency = timedelta(minutes=frequency_dict.save_frequency)
        else:
            target = key
        ckpt_frequency_save_dict[target] = (
            checkpoint_paths[key],
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


def _get_config_enabled_callbacks(
    training_config: Any,
    log_config: Any,
) -> list[Callback]:
    """Get callbacks that are enabled in the callback context."""
    callbacks = []

    if getattr(training_config.swa, "enabled", False):
        callbacks.append(
            StochasticWeightAveraging(
                max_epochs=training_config.max_epochs,
                default_swa_lr=training_config.swa.lr,
            ),
        )

    if getattr(log_config.wandb, "enabled", False) or getattr(log_config.mlflow, "enabled", False):
        callbacks.append(LearningRateMonitor())

    return callbacks


def _get_progress_bar_callback(enable_progress_bar: bool, progress_bar_cfg: Any) -> list[Callback]:
    """Get progress bar callback.

    Instantiated from `config.diagnostics.progress_bar`. If not set, defaults to TQDMProgressBar.

    Example config:
        progress_bar:
          _target_: pytorch_lightning.callbacks.TQDMProgressBar
          refresh_rate: 1
          process_position: 0

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    list[Callback]
        List containing the progress bar callback, or empty list if disabled.
    """
    if not enable_progress_bar:
        LOGGER.info("Progress bar disabled.")
        return []

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


def get_callbacks(context: CallbackContext | DictConfig) -> list[Callback]:
    """Setup callbacks for PyTorch Lightning trainer.

    Set `config.diagnostics.callbacks` to a list of callback configurations
    in hydra form.

    E.g.:
    ```
    callbacks:
        - _target_: anemoi.training.diagnostics.callbacks.RolloutEval
          rollout: 1
          frequency: 12
    ```

    Set `config.diagnostics.plot.callbacks` to a list of plot callback configurations
    will only be added if `config.diagnostics.plot.enabled` is set to True.

    A callback should take a callback context in its `__init__` method as the first argument.
    During migration, passing a full `DictConfig` is still supported.

    Some callbacks are added automatically depending on SWA and logger settings.

    Parameters
    ----------
    context : CallbackContext | DictConfig
        Callback context. A DictConfig is accepted for backward compatibility.

    Returns
    -------
    list[Callback]
        A list of PyTorch Lightning callbacks

    """
    if isinstance(context, DictConfig):
        context = CallbackContext.from_config(context)

    trainer_callbacks: list[Callback] = []

    # Get checkpoint callback
    trainer_callbacks.extend(
        _get_checkpoint_callback(
            enable_checkpointing=context.diagnostics.enable_checkpointing,
            checkpoint_config=context.diagnostics.checkpoint,
            checkpoint_paths=context.system.output.checkpoints,
        ),
    )

    # Base callbacks
    trainer_callbacks.extend(instantiate(callback, context) for callback in context.diagnostics.callbacks)

    # Plotting callbacks
    trainer_callbacks.extend(instantiate(callback, context) for callback in context.diagnostics.plot.callbacks)

    # Extend with config enabled callbacks
    trainer_callbacks.extend(
        _get_config_enabled_callbacks(
            training_config=context.training,
            log_config=context.diagnostics.log,
        ),
    )

    # Progress bar callback
    trainer_callbacks.extend(
        _get_progress_bar_callback(
            enable_progress_bar=context.diagnostics.enable_progress_bar,
            progress_bar_cfg=context.diagnostics.progress_bar,
        ),
    )

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


__all__ = ["get_callbacks"]
