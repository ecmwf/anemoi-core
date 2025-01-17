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
import warnings
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from anemoi.models.data_indices.collection import IndexCollection

import numpy as np

from anemoi.training.utils.variables_metadata import get_variable_group_and_level

LOGGER = logging.getLogger(__name__)


class BaseVariableLossScaler(ABC):
    """Configurable method converting variable to loss scaling."""

    def __init__(
        self,
        scaling_config: DictConfig,
        data_indices: IndexCollection,
        metadata_variables: dict | None = None,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        scaling_config : DictConfig
            Configuration for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        metadata_variables : dict, optional
            Dictionary with variable names as keys and metadata as values, by default None

        """
        self.scaling_config = scaling_config
        self.data_indices = data_indices
        self.variable_groups = self.scaling_config.variable_groups
        # turn dictionary around
        self.group_variables = {}
        for group, variables in self.variable_groups.items():
            if isinstance(variables, str):
                variables = [variables]
            for variable in variables:
                self.group_variables[variable] = group
        self.default_group = self.scaling_config.variable_groups.default
        self.metadata_variables = metadata_variables

    @abstractmethod
    def get_variable_scaling(self) -> np.ndarray: ...

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
        return get_variable_group_and_level(
            variable_name,
            self.group_variables,
            self.metadata_variables,
            self.default_group,
        )


class GeneralVariableLossScaler(BaseVariableLossScaler):
    """General scaling of variables to loss scaling."""

    def get_variable_scaling(self) -> np.ndarray:
        variable_loss_scaling = (
            np.ones((len(self.data_indices.internal_data.output.full),), dtype=np.float32) * self.scaling_config.default
        )

        for variable_name, idx in self.data_indices.internal_model.output.name_to_index.items():
            _, variable_ref, _ = self.get_variable_group(variable_name)
            # Apply variable scaling by base variable name (variable_ref: variable name without variable level)
            variable_loss_scaling[idx] = self.scaling_config.get(
                variable_ref,
                1.0,
            )
            # TODO(all): do we want to allow scaling by variable_ref and variable_name?
            # i.e. scale q_50 by value for q_50 AND q
            if variable_ref != variable_name:
                variable_loss_scaling[idx] *= self.scaling_config.get(
                    variable_name,
                    1.0,
                )

        return variable_loss_scaling


class BaseVariableLevelScaler(BaseVariableLossScaler):
    """Configurable method converting variable level to scaling."""

    def __init__(
        self,
        scaling_config: DictConfig,
        data_indices: IndexCollection,
        metadata_variables: dict,
        group: str,
        y_intercept: float,
        slope: float,
        name: str,
        scale_dim: int,
    ) -> None:
        """Initialise variable level scaler.

        Parameters
        ----------
        scaling_config : DictConfig
            Configuration for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        metadata_variables : dict
            Metadata of the dataset.
        group : str
            Group of variables to scale.
        y_intercept : float
            Y-axis shift of scaling function.
        slope : float
            Slope of scaling function.
        """
        super().__init__(scaling_config, data_indices, metadata_variables)
        self.scaling_group = group
        self.y_intercept = y_intercept
        self.slope = slope
        self.name = name
        self.scale_dim = scale_dim

    @abstractmethod
    def get_level_scaling(self, variable_level: int) -> float: ...

    def get_variable_scaling(self) -> np.ndarray:
        variable_level_scaling = np.ones((len(self.data_indices.internal_data.output.full),), dtype=np.float32)

        LOGGER.info(
            "Variable Level Scaling: Applying %s scaling to %s variables (%s)",
            self.name,
            self.scaling_group,
            self.variable_groups[self.scaling_group],
        )
        LOGGER.info("with slope = %s and y-intercept/minimum = %s.", self.slope, self.y_intercept)

        for variable_name, idx in self.data_indices.internal_model.output.name_to_index.items():
            variable_group, _, variable_level = self.get_variable_group(variable_name)
            if variable_group != self.scaling_group:
                continue
            # Apply variable level scaling
            assert variable_level is not None, f"Variable {variable_name} has no level to scale."
            variable_level_scaling[idx] = self.get_level_scaling(float(variable_level))

        return variable_level_scaling


class LinearVariableLevelScaler(BaseVariableLevelScaler):
    """Linear with slope self.slope, yaxis shift by self.y_intercept."""

    def get_level_scaling(self, variable_level: float) -> np.ndarray:
        return variable_level * self.slope + self.y_intercept


class ReluVariableLevelScaler(BaseVariableLevelScaler):
    """Linear above self.y_intercept, taking constant value self.y_intercept below."""

    def get_level_scaling(self, variable_level: float) -> np.ndarray:
        return max(self.y_intercept, variable_level * self.slope)


class PolynomialVariableLevelScaler(BaseVariableLevelScaler):
    """Polynomial scaling, (slope * variable_level)^2, yaxis shift by self.y_intercept."""

    def get_level_scaling(self, variable_level: float) -> np.ndarray:
        return (self.slope * variable_level) ** 2 + self.y_intercept


class NoVariableLevelScaler(BaseVariableLevelScaler):
    """Constant scaling by 1.0."""

    def __init__(
        self,
        scaling_config: DictConfig,
        data_indices: IndexCollection,
        metadata_variables: dict,
        group: str,
        y_intercept: float = 1.0,
        slope: float = 0.0,
        name: str | None = None,
        scale_dim: int | None = None,
    ) -> None:
        """Initialise Scaler with constant scaling of 1."""
        assert (
            y_intercept == 1.0 and slope == 0
        ), "self.y_intercept must be 1.0 and self.slope 0.0 for no scaling to fit with definition of linear function."
        super().__init__(
            scaling_config,
            data_indices,
            metadata_variables,
            group,
            y_intercept=1.0,
            slope=0.0,
            name=name,
            scale_dim=scale_dim,
        )

    @staticmethod
    def get_level_scaling(variable_level: float) -> np.ndarray:
        del variable_level  # unused
        # no scaling, always return 1.0
        return 1.0


class BaseTendencyScaler(BaseVariableLossScaler):
    """Configurable method to scale prognostic variables based on data statistics and statistics_tendencies."""

    def __init__(
        self,
        scaling_config: DictConfig,
        data_indices: IndexCollection,
        statistics: dict,
        statistics_tendencies: dict,
        name: str,
        scale_dim: int,
    ) -> None:
        """Initialise variable level scaler.

        Parameters
        ----------
        scaling_config : DictConfig
            Configuration for variable loss scaling.
        data_indices : IndexCollection
            Collection of data indices.
        statistics : dict
            Data statistics dictionary
        statistics_tendencies : dict
            Data statistics dictionary for tendencies
        """
        super().__init__(scaling_config, data_indices)
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.name = name
        self.scale_dim = scale_dim

        if not self.statistics_tendencies:
            warnings.warn("Dataset has no tendency statistics! Are you sure you want to use a tendency scaler?")

    @abstractmethod
    def get_level_scaling(self, variable_level: int) -> float: ...

    def get_variable_scaling(self) -> np.ndarray:
        variable_level_scaling = np.ones((len(self.data_indices.internal_data.output.full),), dtype=np.float32)

        LOGGER.info("Variable Level Scaling: Applying %s scaling to prognostic variables", self.name)

        for key, idx in self.data_indices.internal_model.output.name_to_index.items():
            if idx in self.data_indices.internal_model.output.prognostic:
                prog_idx = self.data_indices.data.output.name_to_index[key]
                variable_stdev = self.statistics["stdev"][prog_idx] if self.statistics_tendencies else 1
                variable_tendency_stdev = (
                    self.statistics_tendencies["stdev"][prog_idx] if self.statistics_tendencies else 1
                )
                scaling = self.get_level_scaling(variable_stdev, variable_tendency_stdev)
                LOGGER.info("Parameter %s is being scaled by statistic_tendencies by %.2f", key, scaling)
                variable_level_scaling[idx] *= scaling

        return variable_level_scaling


class NoTendencyScaler(BaseTendencyScaler):
    """No scaling by tendency statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        del variable_stdev, variable_tendency_stdev
        return 1.0


class StdevTendencyScaler(BaseTendencyScaler):
    """Scale loses by standard deviation of tendency statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev / variable_tendency_stdev


class VarTendencyScaler(BaseTendencyScaler):
    """Scale loses by variance of tendency statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev**2 / variable_tendency_stdev**2
