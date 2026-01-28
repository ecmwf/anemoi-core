# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Callable

from anemoi.models.data_indices.tensor import OutputTensorIndex
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

METRIC_RANGE_DTYPE = dict[str, list[int]]

LOGGER = logging.getLogger(__name__)


# Future import breaks other type hints TODO Harrison Cook
def get_loss_function(
    loss: BaseLoss | Callable[..., BaseLoss],
    scalers: dict[str, TENSOR_SPEC] | None = None,
    data_indices: dict | None = None,
    scalers_to_include: list[str] | None = None,
    **kwargs,
) -> BaseLoss:
    """Attach scalers to a loss function or loss factory.

    Parameters
    ----------
    loss : BaseLoss | Callable[..., BaseLoss]
        Loss instance or explicit factory returning a BaseLoss.
    scalers : dict[str, TENSOR_SPEC] | None
        Scalers which can be added to the loss function.
    data_indices : dict, optional
        Indices of the training data
    scalers_to_include : list[str] | None
        Scalers to attach to the loss. Use ["*"] to include all available scalers.
    kwargs : Any
        Additional arguments to pass to the loss factory

    Returns
    -------
    BaseLoss
        The loss function to use for training/validation.

    Raises
    ------
    TypeError
        If not a subclass of `BaseLoss`.
    ValueError
        If scaler is not found in valid scalers
    """
    if scalers is None:
        scalers = {}

    if scalers_to_include is None:
        scalers_to_include = []

    if "*" in scalers_to_include:
        scalers_to_include = [s for s in list(scalers.keys()) if f"!{s}" not in scalers_to_include]

    if isinstance(loss, BaseLoss):
        loss_function = loss
    elif callable(loss):
        loss_function = loss(**kwargs)
    else:
        error_msg = f"Loss must be a BaseLoss or a callable factory, not {type(loss)}"
        raise TypeError(error_msg)

    if not isinstance(loss_function, BaseLoss):
        error_msg = f"Loss must be a subclass of 'BaseLoss', not {type(loss_function)}"
        raise TypeError(error_msg)
    _apply_scalers(loss_function, scalers_to_include, scalers, data_indices)

    return loss_function


def _apply_scalers(
    loss_function: BaseLoss,
    scalers_to_include: list,
    scalers: dict[str, TENSOR_SPEC] | None,
    data_indices: dict | None,
) -> None:
    """Attach scalers to a loss function and set data indices if needed."""
    for key in scalers_to_include:
        if key not in scalers or []:
            error_msg = f"Scaler {key!r} not found in valid scalers: {list(scalers.keys())}"
            raise ValueError(error_msg)
        if key in ["stdev_tendency", "var_tendency"]:
            for var_key, idx in data_indices.model.output.name_to_index.items():
                if idx in data_indices.model.output.prognostic and data_indices.data.output.name_to_index.get(
                    var_key,
                ):
                    scaling = scalers[key][1][idx]
                    LOGGER.info("Parameter %s is being scaled by statistic_tendencies by %.2f", var_key, scaling)
        loss_function.add_scaler(*scalers[key], name=key)

        if hasattr(loss_function, "set_data_indices"):
            loss_function.set_data_indices(data_indices)


def get_metric_ranges(
    extract_variable_group_and_level: ExtractVariableGroupAndLevel,
    output_data_indices: OutputTensorIndex,
    metrics_to_log: list,
) -> METRIC_RANGE_DTYPE:
    metric_ranges = defaultdict(list)

    for key, idx in output_data_indices.name_to_index.items():
        variable_group, variable_ref, _ = extract_variable_group_and_level.get_group_and_level(key)

        # Add metrics for grouped variables and variables in default group
        metric_ranges[f"{variable_group}_{variable_ref}"].append(idx)

        # Specific metrics from hydra to log in logger
        if key in metrics_to_log:
            metric_ranges[key] = [idx]

    # Add the full list of output indices
    metric_ranges["all"] = output_data_indices.full.tolist()
    return metric_ranges
