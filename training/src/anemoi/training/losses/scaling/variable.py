# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.training.losses.scaling import BaseScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig

    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BaseVariableLossScaler(BaseScaler):
    """Base class for all variable loss scalers."""

    def __init__(
        self,
        group_config: DictConfig,
        data_indices: IndexCollection,
        scale_dim: int,
        metadata_variables: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        group_config: DictConfig
            Configuration of groups for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        scale_dim : int
            Dimension to scale
        metadata_variables : dict, optional
            Dictionary with variable names as keys and metadata as values, by default None

        """
        super().__init__(data_indices, scale_dim)
        del kwargs
        self.variable_groups = group_config
        self.metadata_variables = metadata_variables

        self.extract_variable_group_and_level = ExtractVariableGroupAndLevel(
            self.variable_groups,
            self.metadata_variables,
        )

    def get_variable_group(self, variable_name: str) -> tuple[str, str, int]:
        """Get the group of a variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        str
            Group of the variable given in the training-config file.
        str
            Variable reference which corresponds to the variable name without the variable level
        str
            Variable level, i.e. pressure level or model level

        """
        return self.extract_variable_group_and_level.get_group_and_level(
            variable_name,
        )


class GeneralVariableLossScaler(BaseVariableLossScaler):
    """Scaling per variable defined in config file."""

    def __init__(
        self,
        group_config: DictConfig,
        data_indices: IndexCollection,
        weights: DictConfig,
        scale_dim: int,
        metadata_variables: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialise GeneralVariableLossScaler.

        Parameters
        ----------
        group_config : DictConfig
            Configuration of groups for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        weights_config : DictConfig
            Configuration for variable loss scaling.
        scale_dim : int
            Dimension to scale
        metadata_variables : dict, optional
            Dictionary with variable names as keys and metadata as values, by default None

        """
        super().__init__(group_config, data_indices, scale_dim, metadata_variables)
        self.weights = weights
        del kwargs

    def get_scaling(self) -> np.ndarray:
        """Get loss scaling.

        Retrieve the loss scaling for each variable from the config file.
        """
        variable_loss_scaling = (
            np.ones((len(self.data_indices.internal_data.output.full),), dtype=np.float32) * self.weights.default
        )

        for variable_name, idx in self.data_indices.internal_model.output.name_to_index.items():
            _, variable_ref, _ = self.get_variable_group(variable_name)
            # Apply variable scaling by variable name
            # or base variable name (variable_ref: variable name without variable level)
            variable_loss_scaling[idx] = self.weights.get(
                variable_ref,
                1.0,
            )
            if variable_ref != variable_name:
                assert (
                    self.weights.get(
                        variable_name,
                        None,
                    )
                    is None
                ), f"Variable {variable_name} is not allowed to have a separate scaling besides {variable_ref}."

        return variable_loss_scaling


def get_final_variable_scaling(scalers: dict[str, tuple[int | tuple[int, ...] | torch.Tensor]]) -> float:
    """Get the final variable scaling.

    All variable scalings have scale_dim -1, so we can get the right scalar for printing across the variable dimension.

    Parameters
    ----------
    scalers : dict
        Dictionary of scalers.

    Returns
    -------
    float
        Final variable scaling.
    """
    # Subsetting over -1 to get the right scalar for printing across the variable dimension
    final_variable_scaling = 1.0
    for scale_dim, scaling in scalers.values():
        if scale_dim == -1:
            final_variable_scaling = final_variable_scaling * scaling

    return final_variable_scaling


def print_final_variable_scaling(
    scalers: dict[str, tuple[int | tuple[int, ...] | torch.Tensor]], data_indices
) -> None:
    final_variable_scaling = get_final_variable_scaling(scalers)
    log_text = "Final Variable Scaling: "
    for idx, name in enumerate(data_indices.internal_model.output.name_to_index.keys()):
        log_text += f"{name}: {final_variable_scaling[idx]:.4g}, "
    LOGGER.debug(log_text)
