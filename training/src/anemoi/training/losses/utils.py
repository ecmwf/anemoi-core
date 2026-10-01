# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.training.losses.base import BaseLossWrapper
from anemoi.training.losses.combined import CombinedLoss
from anemoi.training.losses.scaler_tensor import as_dimension_tuple
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.variables_metadata import check_loss_variable_units_compatibility

if TYPE_CHECKING:
    import numpy as np
    import torch

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.base import BaseLoss
    from anemoi.training.losses.scaler_tensor import ScaleTensor

LOGGER = logging.getLogger(__name__)


def reduce_to_last_dim(x: np.ndarray) -> np.ndarray:
    """Sum all dimensions of *x* except the last one."""
    if x.ndim > 1:
        return x.sum(axis=tuple(range(x.ndim - 1)))
    return x


def variable_axis_scaling(scaler: ScaleTensor) -> torch.Tensor | None:
    """Return the product of every scaler that acts on the variable axis alone.

    Scalers record the axes they apply to by `TensorDim` name, so the per-variable weight of a loss
    is the product of the one-dimensional VARIABLE scalers it carries (`general_variable`,
    `pressure_level`, the tendency scalers, ...). Scalers that span the variable axis *together with*
    another axis have no single value per variable and are skipped.

    Parameters
    ----------
    scaler : ScaleTensor
        Scalers attached to the loss.

    Returns
    -------
    torch.Tensor | None
        A 1-D tensor of per-variable weights, or None when the loss carries no variable scaler.
    """
    combined = None
    for name, (dims, values) in scaler.tensors.items():
        dims = as_dimension_tuple(dims)
        if TensorDim.VARIABLE not in dims:
            continue
        if len(dims) > 1:
            LOGGER.debug("Scaler %r spans %s, so it has no single value per variable; skipping.", name, dims)
            continue
        flat = values.detach().reshape(-1).float()
        combined = flat if combined is None else combined * flat
    return combined


def print_variable_scaling(loss: BaseLoss, data_indices: IndexCollection) -> dict[str, float]:
    """Log the final variable scaling for each variable in the model and return the scaling values.

    Parameters
    ----------
    loss : BaseLoss
        Loss function to get the variable scaling from.
    data_indices : IndexCollection
        Index collection to get the variable names from.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping variable names to their scaling values. If max_variables is specified,
        only the top N variables plus 'total_sum' will be included.
    """
    # TODO(frazane,anaprietonem): handle special cases more gracefully
    # https://github.com/ecmwf/anemoi-core/pull/723#discussion_r2597831922
    if isinstance(loss, CombinedLoss):
        variable_scaling = {}
        key_count_by_name: dict[str, int] = {}
        for sub_loss in loss.losses:
            base_key = sub_loss.__class__.__name__
            key_count_by_name[base_key] = key_count_by_name.get(base_key, 0) + 1
            suffix = "" if key_count_by_name[base_key] == 1 else f"_{key_count_by_name[base_key]}"
            variable_scaling[f"{base_key}{suffix}"] = print_variable_scaling(sub_loss, data_indices)
        return variable_scaling

    if isinstance(loss, LossVariableMapper):
        subset_vars = enumerate(loss.predicted_variables)
        # LossVariableMapper forwards scalers to its inner loss, so get scaling from there
        scaler_source = loss.loss.scaler
    elif isinstance(loss, BaseLossWrapper):
        return print_variable_scaling(loss.loss, data_indices)
    else:
        subset_vars = enumerate(data_indices.model.output.name_to_index.keys())
        scaler_source = loss.scaler

    variable_scaling = variable_axis_scaling(scaler_source)
    log_text = f"Final Variable Scaling in {loss.__class__.__name__}: "
    scaling_values, scaling_sum = {}, 0.0
    for idx, name in subset_vars:
        if variable_scaling is None:
            value = 1.0
        elif variable_scaling.numel() == 1:
            # A broadcast scaler applies the same weight to every variable.
            value = float(variable_scaling[0])
        else:
            value = float(variable_scaling[idx]) if idx < variable_scaling.numel() else 1.0
        log_text += f"{name}: {value:.4g}, "
        scaling_values[name] = value
        scaling_sum += value

    log_text += f"Total scaling sum: {scaling_sum:.4g}, "
    scaling_values["total_sum"] = scaling_sum
    LOGGER.debug(log_text)

    return scaling_values


def check_loss_tree_variable_units(
    loss: object,
    variables_metadata: dict[str, dict] | None,
    **options: object,
) -> None:
    """Walk a loss tree and check unit compatibility for any variable-mapped losses.

    Recurses into composite losses (e.g. ``CombinedLoss``) to find all
    ``LossVariableMapper`` instances and validate their predicted/target pairs.

    Parameters
    ----------
    loss : object
        The loss (or composite loss) to inspect.
    variables_metadata : dict[str, dict] | None
        Per-variable metadata dict keyed by variable name.
    **options : object
        Additional keyword arguments forwarded to ``Variable.compatible``
        (e.g. ``ignore_units``, ``ignore_processing_period``).
    """
    predicted_variables = getattr(loss, "predicted_variables", None)
    target_variables = getattr(loss, "target_variables", None)

    if predicted_variables is not None and target_variables is not None:
        # Per-loss options (from check_variables_compatibility on this loss config entry)
        # take precedence over any inherited options passed from the caller.
        node_options = {**options, **getattr(loss, "compatibility_options", {})}
        check_loss_variable_units_compatibility(
            predicted_variables,
            target_variables,
            variables_metadata,
            **node_options,
        )

    # Recurse into composite losses (e.g. CombinedLoss.losses)
    for sub_loss in getattr(loss, "losses", []):
        check_loss_tree_variable_units(sub_loss, variables_metadata, **options)
