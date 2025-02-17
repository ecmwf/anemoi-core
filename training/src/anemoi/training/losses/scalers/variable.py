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

from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BaseVariableLossScaler(BaseScaler):
    """Base class for all variable loss scalers."""

    scale_dims: int = 3

    def __init__(
        self,
        group_config: DictConfig,
        data_indices: IndexCollection,
        metadata_variables: dict | None = None,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        group_config: DictConfig
            Configuration of groups for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        metadata_variables : dict, optional
            Dictionary with variable names as keys and metadata as values, by default None
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(data_indices, norm=norm)
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
        metadata_variables: dict | None = None,
        norm: str | None = None,
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
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(group_config, data_indices, metadata_variables=metadata_variables, norm=norm)
        self.weights = weights
        del kwargs

    def get_scaling(self, **_kwargs) -> np.ndarray:
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
