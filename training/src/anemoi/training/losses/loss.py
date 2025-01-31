# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses.weightedloss import BaseLoss

LOGGER = logging.getLogger(__name__)


# Future import breaks other type hints TODO Harrison Cook
def get_loss_function(
    config: DictConfig,
    scalers: dict[str, tuple[int | tuple[int, ...] | torch.Tensor]] | None = None,
    **kwargs,
) -> BaseLoss | torch.nn.ModuleList:
    """Get loss functions from config.

    Can be ModuleList if multiple losses are specified.

    Parameters
    ----------
    config : DictConfig
        Loss function configuration, should include `scalers` if scalers are to be added to the loss function.
    scalers : dict[str, tuple[int | tuple[int, ...] | torch.Tensor]], optional
        Scalers which can be added to the loss function. Defaults to None., by default None
        If a scaler is to be added to the loss, ensure it is in `scalers` in the loss config
        E.g.
            If `scalers: ['variable']` is set in the config, and `variable` in `scalers`
            `variable` will be added to the scaler of the loss function.
    kwargs : Any
        Additional arguments to pass to the loss function

    Returns
    -------
    Union[BaseLoss, torch.nn.ModuleList]
        Loss function, or list of metrics

    Raises
    ------
    TypeError
        If not a subclass of `BaseLoss`
    ValueError
        If scaler is not found in valid scalers
    """
    config_container = OmegaConf.to_container(config, resolve=False)
    if isinstance(config_container, list):
        return torch.nn.ModuleList(
            [get_loss_function(OmegaConf.create(loss_config), scalers=scalers, **kwargs) for loss_config in config],
        )

    loss_config = OmegaConf.to_container(config, resolve=True)
    scalers_to_include = loss_config.pop("scalers", [])

    if "*" in scalers_to_include:
        scalers_to_include = [s for s in list(scalers.keys()) if f"!{s}" not in scalers_to_include]

    # Instantiate the loss function with the loss_init_config
    loss_function = instantiate(loss_config, **kwargs)

    if not isinstance(loss_function, BaseLoss):
        error_msg = f"Loss must be a subclass of 'BaseLoss', not {type(loss_function)}"
        raise TypeError(error_msg)

    for key in scalers_to_include:
        if key not in scalers or []:
            error_msg = f"Scaler {key!r} not found in valid scalers: {list(scalers.keys())}"
            raise ValueError(error_msg)
        loss_function.add_scaler(*scalers[key], name=key)

    return loss_function
