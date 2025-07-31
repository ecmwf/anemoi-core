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

from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.data_indices.tensor import OutputTensorIndex
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

METRIC_RANGE_DTYPE = dict[str, list[int]]
LOGGER = logging.getLogger(__name__)


# Future import breaks other type hints TODO Harrison Cook
def get_loss_function(
    config: DictConfig,
    scalers: dict[str, TENSOR_SPEC] | None = None,
    data_indices: dict | None = None,
    **kwargs,
) -> BaseLoss:
    """Get loss functions from config.

    Can be ModuleList if multiple losses are specified.

    Parameters
    ----------
    config : DictConfig
        Loss function configuration, should include `scalers` if scalers are to be added to the loss function.
    scalers : TENSOR_SPEC, optional,
        Scalers which can be added to the loss function. Defaults to None., by default None
        If a scaler is to be added to the loss, ensure it is in `scalers` in the loss config.
        For instance, if `scalers: ['variable']` is set in the config, and `variable` in `scalers`
        `variable` will be added to the scaler of the loss function.
    data_indices : dict, optional
        Indices of the training data
    kwargs : Any
        Additional arguments to pass to the loss function

    Returns
    -------
    BaseLoss | torch.nn.ModuleDict
        The loss function, or dict of metrics, to use for training/validation.

    Raises
    ------
    TypeError
        If not a subclass of `BaseLoss`.
    ValueError
        If scaler is not found in valid scalers
    """
    loss_config = OmegaConf.to_container(config, resolve=True)
    scalers_to_include = loss_config.pop("scalers", [])

    if scalers is None:
        scalers = {}

    if "*" in scalers_to_include:
        scalers_to_include = [s for s in list(scalers.keys()) if f"!{s}" not in scalers_to_include]

    # Instantiate the loss function with the loss_init_config
    loss_function = instantiate(loss_config, **kwargs, _recursive_=False)

    if not isinstance(loss_function, BaseLoss):
        error_msg = f"Loss must be a subclass of 'BaseLoss', not {type(loss_function)}"
        raise TypeError(error_msg)
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

    return loss_function


def _get_metric_ranges(
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


def get_metric_ranges(
    config: DictConfig,
    data_indices: IndexCollection,
    metadata_extractor: ExtractVariableGroupAndLevel,
) -> tuple[METRIC_RANGE_DTYPE, METRIC_RANGE_DTYPE]:

    metrics_to_log = config.training.metrics or []

    return _get_metric_ranges(
        metadata_extractor,
        data_indices.model.output,
        metrics_to_log,
    )
