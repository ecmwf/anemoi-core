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
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

if TYPE_CHECKING:
    import numpy as np

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.models.data_indices.tensor import OutputTensorIndex

METRIC_RANGE_DTYPE = dict[str, list[int]]
LOGGER = logging.getLogger(__name__)


# Future import breaks other type hints TODO Harrison Cook
def get_loss_function(
    config: DictConfig,
    scalers: dict[str, tuple[tuple[int] | np.ndarray]] | None = None,
    **kwargs,
) -> BaseLoss:
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
    Union[BaseLoss, torch.nn.ModuleDict]
        Loss function, or dict of metrics

    Raises
    ------
    TypeError
        If not a subclass of `BaseLoss`
    ValueError
        If scaler is not found in valid scalers
    """
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


def _get_metric_ranges(
    extract_variable_group_and_level: ExtractVariableGroupAndLevel,
    output_data_indices: OutputTensorIndex,
    metrics_to_log: list | None = None,
) -> METRIC_RANGE_DTYPE:
    metric_ranges = defaultdict(list)

    for key, idx in output_data_indices.name_to_index.items():
        variable_group, variable_ref, _ = extract_variable_group_and_level.get_group_and_level(key)

        # Add metrics for grouped variables and variables in default group
        metric_ranges[f"{variable_group}_{variable_ref}"].append(idx)

        # Specific metrics from hydra to log in logger
        if metrics_to_log is not None and key in metrics_to_log:
            metric_ranges[key] = [idx]

    # Add the full list of output indices
    metric_ranges["all"] = output_data_indices.full.tolist()
    return metric_ranges


def get_metric_ranges(
    config: DictConfig,
    data_indices: IndexCollection,
    metadata_variables: dict | None = None,
) -> tuple[METRIC_RANGE_DTYPE, METRIC_RANGE_DTYPE]:

    metric_ranges = defaultdict(list)
    metric_ranges_validation = defaultdict(list)
    variable_groups = config.training.scalers.variable_groups
    metrics_to_log = config.training.metrics

    extract_variable_group_and_level = ExtractVariableGroupAndLevel(variable_groups, metadata_variables)
    metric_ranges = _get_metric_ranges(
        extract_variable_group_and_level,
        data_indices.internal_model.output,
        metrics_to_log=metrics_to_log,
    )
    metric_ranges_validation = _get_metric_ranges(
        extract_variable_group_and_level,
        data_indices.model.output,
        metrics_to_log=metrics_to_log,
    )
    return metric_ranges, metric_ranges_validation
