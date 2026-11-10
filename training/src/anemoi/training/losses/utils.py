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
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.variables_metadata import check_loss_variable_units_compatibility

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import numpy as np

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


def reduce_to_last_dim(x: np.ndarray) -> np.ndarray:
    """Sum all dimensions of *x* except the last one."""
    if x.ndim > 1:
        return x.sum(axis=tuple(range(x.ndim - 1)))
    return x


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

    variable_scaling = scaler_source.subset_by_dim(TensorDim.VARIABLE).get_scaler(len(TensorDim)).reshape(-1)
    log_text = f"Final Variable Scaling in {loss.__class__.__name__}: "
    scaling_values, scaling_sum = {}, 0.0
    for idx, name in subset_vars:
        value = float(variable_scaling[idx]) if idx < variable_scaling.shape[0] else 1.0
        log_text += f"{name}: {value:.4g}, "
        scaling_values[name] = value
        scaling_sum += value

    log_text += f"Total scaling sum: {scaling_sum:.4g}, "
    scaling_values["total_sum"] = scaling_sum
    LOGGER.debug(log_text)

    return scaling_values


def check_loss_tree_ignores_nans(loss: BaseLoss, allow_nans: Sequence[str], description: str) -> None:
    """Check that a loss ignores NaNs for all variables that are allowed to contain NaNs.

    Walks the loss tree and raises if a loss that computes on one of the ``allow_nans`` variables
    does not have ``ignore_nans`` set. Losses that cannot ignore NaNs (e.g. spectral losses) always fail.
    A ``LossVariableMapper`` restricts the check to the variables it computes on (its target variables).

    Parameters
    ----------
    loss : BaseLoss
        The loss (or composite loss) to inspect.
    allow_nans : Sequence[str]
        Names of the variables that are allowed to contain NaNs.
    description : str
        Description of the loss used in the error message, e.g. "training loss of dataset 'data'".

    Raises
    ------
    ValueError
        If a loss that sees any of the variables does not ignore NaNs.
    """
    if not allow_nans:
        return

    if isinstance(loss, LossVariableMapper):
        # Without data indices the mapper has no variable selection, so it sees all variables.
        seen = allow_nans if loss.target_variables is None else [v for v in allow_nans if v in loss.target_variables]
        if seen:
            check_loss_tree_ignores_nans(loss.loss, seen, description)
        return

    leaves = list(loss.iter_leaf_losses())
    if not (len(leaves) == 1 and leaves[0] is loss):
        # Composite loss (e.g. CombinedLoss) or transparent wrapper (e.g. TimeAggregateLossWrapper): descend into the
        # children so that variable selections of nested LossVariableMappers are respected.
        children = getattr(loss, "losses", None)
        if children is None:
            children = [loss.loss] if isinstance(loss, BaseLossWrapper) else leaves
        for child in children:
            check_loss_tree_ignores_nans(child, allow_nans, description)
        return

    if not loss.ignore_nans:
        msg = (
            f"The {description} ({type(loss).__name__}) does not ignore NaNs, but the variables "
            f"{allow_nans} are listed under `allow_nans` in the data config and can contain NaNs. "
            "Set `ignore_nans: True` for this loss, restrict it to other variables with `predicted_variables`/"
            "`target_variables`, or remove the variables from `allow_nans`. "
            "Losses that cannot ignore NaNs (e.g. spectral losses) cannot be used with these variables."
        )
        raise ValueError(msg)


def check_loss_and_metrics_ignore_nans(
    loss: BaseLoss,
    metrics: Mapping[str, BaseLoss],
    allow_nans: Sequence[str],
    dataset_name: str,
) -> None:
    """Check that the training loss and all validation metrics of a dataset ignore NaNs where required.

    Parameters
    ----------
    loss : BaseLoss
        Training loss of the dataset.
    metrics : Mapping[str, BaseLoss]
        Validation metrics of the dataset, by name.
    allow_nans : Sequence[str]
        Names of the variables that are allowed to contain NaNs.
    dataset_name : str
        Name of the dataset, used in the error messages.

    Raises
    ------
    ValueError
        If the training loss or a validation metric that sees any of the variables does not ignore NaNs.
    """
    allow_nans = list(allow_nans)
    check_loss_tree_ignores_nans(loss, allow_nans, f"training loss of dataset {dataset_name!r}")
    for metric_name, metric in metrics.items():
        check_loss_tree_ignores_nans(
            metric,
            allow_nans,
            f"validation metric {metric_name!r} of dataset {dataset_name!r}",
        )


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
