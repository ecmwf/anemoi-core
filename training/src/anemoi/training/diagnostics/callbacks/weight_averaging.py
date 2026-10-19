# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
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
    """Base class that averages parameters and synchronises fixed buffers."""

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, use_buffers=use_buffers, **kwargs)


class EMAWeightAveraging(WeightAveraging):
    """Exponential Moving Average weight averaging."""

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = False,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            **kwargs,
            avg_fn=get_ema_avg_fn(decay=decay),
        )
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch


class SWAWeightAveraging(WeightAveraging):
    """Stochastic Weight Averaging (running mean).

    Uses the default running-mean function from PyTorch's ``AveragedModel``.
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


def _get_weight_averaging_callback(weight_averaging_config: DictConfig | None) -> list[Callback]:
    """Get weight averaging callback from the config.

    Example config (recommended):
        weight_averaging:
            _target_: anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging
            decay: 0.999

    Stock ``pytorch_lightning.callbacks.*WeightAveraging`` classes can also be used.
    Set ``use_buffers=False`` when the model contains non-floating-point buffers.

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
