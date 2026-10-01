# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from typing import Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from packaging.version import Version
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import WeightAveraging as _PLWeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

LOGGER = logging.getLogger(__name__)

MIN_PL_VERSION = "2.6.0"


class WeightAveraging(_PLWeightAveraging):
    """Base class that averages parameters and synchronises fixed buffers.

    Adds the update schedule shared by the EMA and SWA variants: the averaged model is updated
    every ``update_every_n_steps`` optimizer steps once ``update_starting_at_step`` is reached, and
    additionally at the end of every epoch from ``update_starting_at_epoch`` onwards.
    """

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = False,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, use_buffers=use_buffers, **kwargs)
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch

    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        """Decide whether to update the averaged model after the given step or epoch.

        Parameters
        ----------
        step_idx : int | None
            Index of the last optimizer step, or None when called at the end of an epoch.
        epoch_idx : int | None
            Index of the last epoch, or None when called after an optimizer step.

        Returns
        -------
        bool
            True if the averaged model should be updated.
        """
        if step_idx is not None:
            meets_step_requirement = self.update_starting_at_step is None or step_idx >= self.update_starting_at_step
            meets_step_frequency = self.update_every_n_steps > 0 and step_idx % self.update_every_n_steps == 0
            if meets_step_requirement and meets_step_frequency:
                return True

        if epoch_idx is not None:
            return self.update_starting_at_epoch is not None and epoch_idx >= self.update_starting_at_epoch

        return False


class EMAWeightAveraging(WeightAveraging):
    """Exponential Moving Average weight averaging."""

    def __init__(self, decay: float = 0.999, **kwargs: Any) -> None:
        super().__init__(**kwargs, avg_fn=get_ema_avg_fn(decay=decay))


class SWAWeightAveraging(WeightAveraging):
    """Stochastic Weight Averaging (running mean).

    Uses the default running-mean function from PyTorch's ``AveragedModel``.
    """


@contextmanager
def averaged_weights(trainer: "pl.Trainer") -> Iterator[None]:
    """Temporarily swap the averaged weights into the live model.

    The model holds the raw training weights outside of validation, so anything written from the
    live model (the inference checkpoint) would otherwise disagree with the averaged weights that
    validation metrics and the Lightning checkpoint's ``state_dict`` are based on. Does nothing when
    weight averaging is not configured, or before training starts.

    Parameters
    ----------
    trainer : pl.Trainer
        The current trainer, used to find the weight averaging callback.
    """
    callback = next(
        (cb for cb in trainer.callbacks if isinstance(cb, _PLWeightAveraging) and cb._average_model is not None),
        None,
    )
    if callback is None:
        yield
        return

    # _swap_models is pytorch-lightning internal, but it is the same swap the callback performs
    # around every validation epoch, and it is an involution: swapping twice restores the model.
    callback._swap_models(trainer.lightning_module)
    try:
        yield
    finally:
        callback._swap_models(trainer.lightning_module)


def _get_weight_averaging_callback(weight_averaging_config: DictConfig | None) -> list[Callback]:
    """Get weight averaging callback from the config.

    Example config (recommended):
        weight_averaging:
            _target_: anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging
            decay: 0.999

    Stock ``pytorch_lightning.callbacks.*WeightAveraging`` classes can also be used, but they
    default to ``use_buffers=True``, which fails on non-floating-point buffers.

    Parameters
    ----------
    weight_averaging_config : DictConfig | None
        Weight averaging configuration (``config.training.weight_averaging``),
        or ``None`` if not configured.

    Returns
    -------
    list[Callback]
        List containing the weight averaging callback, or empty list if not configured.
    """
    if weight_averaging_config is None:
        LOGGER.debug("No weight averaging configured. Skipping.")
        return []
    if not isinstance(weight_averaging_config, dict | DictConfig):
        LOGGER.warning(
            "training.weight_averaging has unexpected type %s; expected a dict with '_target_'. Skipping.",
            type(weight_averaging_config).__name__,
        )
        return []
    if "_target_" not in weight_averaging_config:
        LOGGER.warning("training.weight_averaging is set but has no '_target_' field. Skipping.")
        return []

    if Version(pl.__version__) < Version(MIN_PL_VERSION):
        msg = (
            f"Weight averaging callback {weight_averaging_config['_target_']!r} requires "
            f"pytorch_lightning>={MIN_PL_VERSION}, but found {pl.__version__}. "
            f"Please upgrade pytorch_lightning to use this callback."
        )
        raise RuntimeError(msg)

    callback = instantiate(weight_averaging_config)
    LOGGER.info("Loaded weight averaging callback: %s", weight_averaging_config["_target_"])

    return [callback]
