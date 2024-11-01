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
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Callable
from typing import Iterable

from hydra.utils import instantiate
from omegaconf import DictConfig

from anemoi.training.diagnostics.callbacks.checkpoint import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks.optimiser import LearningRateMonitor
from anemoi.training.diagnostics.callbacks.optimiser import StochasticWeightAveraging
from anemoi.training.diagnostics.callbacks.provenance import ParentUUIDCallback

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)


def nestedget(conf: DictConfig, key, default):
    """Get a nested key from a DictConfig object

    E.g.
    >>> nestedget(config, "diagnostics.log.wandb.enabled", False)
    """
    keys = key.split(".")
    for k in keys:
        conf = conf.get(k, default)
        if not isinstance(conf, (dict, DictConfig)):
            break
    return conf


# Callbacks to add according to flags in the config
# Can be function to check status from config
CONFIG_ENABLED_CALLBACKS: list[tuple[list[str] | str | Callable[[DictConfig], bool], type[Callback]]] = [
    ("training.swa.enabled", StochasticWeightAveraging),
    (
        lambda config: nestedget(config, "diagnostics.log.wandb.enabled", False)
        or nestedget(config, "diagnostics.log.mlflow.enabled", False),
        LearningRateMonitor,
    ),
]


def _get_checkpoint_callback(config: DictConfig) -> list[AnemoiCheckpoint] | None:
    """Get checkpointing callback"""
    if not config.diagnostics.get("enable_checkpointing", True):
        return []

    checkpoint_settings = {
        "dirpath": config.hardware.paths.checkpoints,
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

    for key, frequency_dict in config.diagnostics.checkpoint.items():
        frequency = frequency_dict["save_frequency"]
        n_saved = frequency_dict["num_models_saved"]
        if key == "every_n_minutes" and frequency_dict["save_frequency"] is not None:
            target = "train_time_interval"
            frequency = timedelta(minutes=frequency_dict["save_frequency"])
        else:
            target = key
        ckpt_frequency_save_dict[target] = (
            config.hardware.files.checkpoint[key],
            frequency,
            n_saved,
        )

    if not config.diagnostics.profiler:
        for save_key, (
            name,
            save_frequency,
            save_n_models,
        ) in ckpt_frequency_save_dict.items():
            if save_frequency is not None:
                LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
                return (
                    # save_top_k: the save_top_k flag can either save the best or the last k checkpoints
                    # depending on the monitor flag on ModelCheckpoint.
                    # See https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html for reference
                    [
                        AnemoiCheckpoint(
                            config=config,
                            filename=name,
                            save_last=True,
                            **{save_key: save_frequency},
                            # if save_top_k == k, last k models saved; if save_top_k == -1, all models are saved
                            save_top_k=save_n_models,
                            monitor="step",
                            mode="max",
                            **checkpoint_settings,
                        ),
                    ]
                )
            else:
                LOGGER.debug("Not setting up a checkpoint callback with %s", save_key)
    else:
        # the tensorboard logger + pytorch profiler cause pickling errors when writing checkpoints
        LOGGER.warning("Profiling is enabled - will not write any training or inference model checkpoints!")
    return None


def _get_config_enabled_callbacks(config: DictConfig) -> list[Callback]:
    """Get callbacks that are enabled in the config as according to CONFIG_ENABLED_CALLBACKS

    Provides backwards compatibility
    """
    callbacks = []

    def check_key(config, key: str | Iterable[str] | Callable[[DictConfig], bool]):
        """Check key in config."""
        if isinstance(key, Callable):
            return key(config)
        elif isinstance(key, str):
            return nestedget(config, key, False)
        elif isinstance(key, Iterable):
            return all(nestedget(config, k, False) for k in key)
        return nestedget(config, key, False)

    for enable_key, callback_list in CONFIG_ENABLED_CALLBACKS:
        if check_key(config, enable_key):
            callbacks.append(callback_list(config))

    return callbacks


def get_callbacks(config: DictConfig) -> list[Callback]:  # noqa: C901
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

    A callback must take a `DictConfig` in its `__init__` method as the first argument,
    which will be the complete configuration object.

    Some callbacks are added by default, depending on the configuration.
    See CONFIG_ENABLED_CALLBACKS for more information.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    List[Callback]
        A list of PyTorch Lightning callbacks

    """

    trainer_callbacks: list[Callback] = []

    # Get Checkpoint callback
    checkpoint_callback = _get_checkpoint_callback(config)
    if checkpoint_callback is not None:
        trainer_callbacks.extend(checkpoint_callback)

    # Base callbacks
    for callback in config.diagnostics.get("callbacks", None) or []:
        # Instantiate new callbacks
        trainer_callbacks.append(instantiate(callback, config))

    # Plotting callbacks
    for callback in config.diagnostics.plot.get("callbacks", None) or []:
        # Instantiate new callbacks
        trainer_callbacks.append(instantiate(callback, config))

    # Extend with config enabled callbacks
    trainer_callbacks.extend(_get_config_enabled_callbacks(config))

    # Parent UUID callback
    trainer_callbacks.append(ParentUUIDCallback(config))

    return trainer_callbacks


__all__ = ["get_callbacks"]
