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
from typing import TYPE_CHECKING

import numpy as np
from hydra.utils import instantiate

from anemoi.training.losses.scalers.base_scaler import BaseDelayedScaler

if TYPE_CHECKING:
    import torch

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.utils.masks import BaseMask
    from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def create_scalers(
    scalers_config: DotDict,
    data_indices: IndexCollection,
    output_mask: BaseMask,
    **kwargs,
) -> tuple[dict, dict]:
    scalers, delayed_scaler_builders = {}, {}
    for name, config in scalers_config.items():
        scaler_builder = instantiate(config, data_indices=data_indices, **kwargs)

        if isinstance(scaler_builder, BaseDelayedScaler):
            delayed_scaler_builders[name] = scaler_builder
            scalers[name] = (scaler_builder.scale_dims, np.ones(tuple([1] * len(scaler_builder.scale_dims))))
            continue

        scaler_values = scaler_builder.get_scaling()

        # If a scaler needs to apply the output mask (LAM) after its creation,
        # it must include the apply_output_mask attribue.
        if scaler_builder.is_spatial_dim_scaled and getattr(scaler_builder, "apply_output_mask", False):
            scaler_values = output_mask.apply(scaler_values, dim=0, fill_value=0.0)

        scaler_values = scaler_builder.normalise(scaler_values)
        scalers[name] = (scaler_builder.scale_dims, scaler_values)

    print_final_variable_scaling(scalers, data_indices)

    return scalers, delayed_scaler_builders


def get_final_variable_scaling(scalers: dict[str, tuple[tuple[int, ...] | torch.Tensor]]) -> torch.Tensor:
    """Get the final variable scaling.

    All variable scalings have scale_dim -1, so we can get the right scalar for printing across the variable dimension.

    Parameters
    ----------
    scalers : dict
        Dictionary of scalers.

    Returns
    -------
    torch.Tensor
        Final variable scaling.
    """
    # Subsetting over -1 to get the right scalar for printing across the variable dimension
    final_variable_scaling = 1.0
    for scale_dims, scaling in scalers.values():
        if -1 in scale_dims or 3 in scale_dims:
            final_variable_scaling = final_variable_scaling * scaling.squeeze()

    return final_variable_scaling


def print_final_variable_scaling(
    scalers: dict[str, tuple[tuple[int] | torch.Tensor]],
    data_indices: IndexCollection,
) -> None:
    final_variable_scaling = get_final_variable_scaling(scalers)
    log_text = "Final Variable Scaling: "
    for idx, name in enumerate(data_indices.internal_model.output.name_to_index.keys()):
        log_text += f"{name}: {final_variable_scaling[idx]:.4g}, "
    LOGGER.debug(log_text)
