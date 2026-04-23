# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import types
from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from packaging.version import Version
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)

MIN_PL_VERSION = "2.6.0"


def _safe_swap_models(self: Any, pl_module: Any) -> None:
    """Swap buffers between the averaged model and the current model.

    Uses name-based matching to allow buffer reordering as can happen with dynamic scalers (e.g.
    NaNMaskScaler).

    Args:
        pl_module : The PyTorch Lightning module
    """
    assert self._average_model is not None

    avg_params = dict(self._average_model.module.named_parameters())
    for name, current_param in pl_module.named_parameters():
        avg_param = avg_params[name]
        tmp = avg_param.data.clone()
        avg_param.data.copy_(current_param.data)
        current_param.data.copy_(tmp)

    avg_buffers = dict(self._average_model.module.named_buffers())
    for name, current_buf in pl_module.named_buffers():
        if name not in avg_buffers:
            continue
        avg_buf = avg_buffers[name]
        if avg_buf.shape != current_buf.shape:
            continue
        tmp = avg_buf.data.clone()
        avg_buf.data.copy_(current_buf.data)
        current_buf.data.copy_(tmp)


def _safe_copy_average_to_current(self: Any, pl_module: Any) -> None:
    """Copy averaged buffers to the current model.

    Same as ``_safe_swap_models`` but for the copy performed at the end of training.
    """
    assert self._average_model is not None

    avg_params = dict(self._average_model.module.named_parameters())
    for name, current_param in pl_module.named_parameters():
        current_param.data.copy_(avg_params[name].data)

    avg_buffers = dict(self._average_model.module.named_buffers())
    for name, current_buf in pl_module.named_buffers():
        if name not in avg_buffers:
            continue
        avg_buf = avg_buffers[name]
        if avg_buf.shape != current_buf.shape:
            continue
        current_buf.data.copy_(avg_buf.data)


def _get_weight_averaging_callback(config: DictConfig) -> list[Callback]:
    """Get weight averaging callback from the config.

    Example config for EMA weight averaging:
        weight_averaging:
            _target_: pytorch_lightning.callbacks.EMAWeightAveraging
            decay: 0.999

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    list[Callback]
        List containing the weight averaging callback, or empty list if not configured.
    """
    from anemoi.training.diagnostics.callbacks import nestedget

    weight_averaging_config = nestedget(config, "training.weight_averaging", None)
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

    # Patch swap/copy methods to use name-based matching. Needed for dynamic scalers like NaNMaskScaler.
    if hasattr(callback, "_swap_models"):
        callback._swap_models = types.MethodType(_safe_swap_models, callback)
    if hasattr(callback, "_copy_average_to_current"):
        callback._copy_average_to_current = types.MethodType(_safe_copy_average_to_current, callback)

    return [callback]
