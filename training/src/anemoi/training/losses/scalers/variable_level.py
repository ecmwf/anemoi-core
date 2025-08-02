# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.variable import BaseVariableLossScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LOGGER = logging.getLogger(__name__)


class BaseVariableLevelScaler(BaseVariableLossScaler):
    """Configurable method converting variable level to loss scalings."""

    def __init__(
        self,
        data_indices: IndexCollection,
        group: str,
        y_intercept: float,
        slope: float,
        metadata_extractor: ExtractVariableGroupAndLevel,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise variable level scaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        group : str
            Group of variables to scale.
        y_intercept : float
            Y-axis shift of scaling function.
        slope : float
            Slope of scaling function.
        metadata_extractor : ExtractVariableGroupAndLevel
            Metadata extractor for variable groups and levels.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(data_indices, metadata_extractor=metadata_extractor, norm=norm)
        del kwargs
        self.scaling_group = group
        self.y_intercept = y_intercept
        self.slope = slope

    @abstractmethod
    def get_level_scaling(self, variable_level: int) -> float:
        """Get the scaling of a variable level.

        Parameters
        ----------
        variable_level : int
            Variable level to scale.

        Returns
        -------
        float
            Scaling of the variable level.
        """
        ...

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        variable_level_scaling = torch.ones((len(self.data_indices.data.output.full),), dtype=torch.float32)

        LOGGER.info(
            "Variable Level Scaling: Applying %s scaling to %s variables (%s)",
            self.__class__.__name__,
            self.scaling_group,
            self.variable_metadata_extractor.get_group_specification(self.scaling_group),
        )
        LOGGER.info("with slope = %s and y-intercept/minimum = %s.", self.slope, self.y_intercept)

        for variable_name, idx in self.data_indices.model.output.name_to_index.items():
            variable_group, _, variable_level = self.variable_metadata_extractor.get_group_and_level(variable_name)
            if variable_group != self.scaling_group:
                continue
            # Apply variable level scaling
            assert variable_level is not None, f"Variable {variable_name} has no level to scale."
            variable_level_scaling[idx] = self.get_level_scaling(float(variable_level))

        return variable_level_scaling


class LinearVariableLevelScaler(BaseVariableLevelScaler):
    """Linear with slope self.slope, yaxis shift by self.y_intercept."""

    def get_level_scaling(self, variable_level: float) -> torch.Tensor:
        return variable_level * self.slope + self.y_intercept


class ReluVariableLevelScaler(BaseVariableLevelScaler):
    """Linear above self.y_intercept, taking constant value self.y_intercept below."""

    def get_level_scaling(self, variable_level: float) -> torch.Tensor:
        return max(self.y_intercept, variable_level * self.slope)


class PolynomialVariableLevelScaler(BaseVariableLevelScaler):
    """Polynomial scaling, (slope * variable_level)^2, yaxis shift by self.y_intercept."""

    def get_level_scaling(self, variable_level: float) -> torch.Tensor:
        return (self.slope * variable_level) ** 2 + self.y_intercept


class NoVariableLevelScaler(BaseVariableLevelScaler):
    """Constant scaling by 1.0."""

    def __init__(
        self,
        data_indices: IndexCollection,
        group: str,
        metadata_extractor: ExtractVariableGroupAndLevel,
        **kwargs,
    ) -> None:
        """Initialise Scaler with constant scaling of 1."""
        del kwargs
        super().__init__(
            data_indices,
            group,
            y_intercept=1.0,
            slope=0.0,
            metadata_extractor=metadata_extractor,
        )

    @staticmethod
    def get_level_scaling(variable_level: float) -> torch.Tensor:
        del variable_level  # unused
        # no scaling, always return 1.0
        return 1.0
