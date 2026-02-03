# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from abc import abstractmethod

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseTendencyScaler(BaseScaler):
    """Configurable method to scale prognostic variables based on data statistics and statistics_tendencies."""

    scale_dims: TensorDim = TensorDim.VARIABLE

    def __init__(
        self,
        data_indices: IndexCollection,
        statistics: dict,
        statistics_tendencies: dict,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise variable level scaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        statistics : dict
            Data statistics dictionary
        statistics_tendencies : dict
            Data statistics dictionary for tendencies
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        del kwargs
        self.data_indices = data_indices
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies

        if not self.statistics_tendencies:
            warnings.warn("Dataset has no tendency statistics! Are you sure you want to use a tendency scaler?")

    @abstractmethod
    def get_level_scaling(self, variable_level: int) -> float: ...

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        variable_level_scaling = torch.ones((len(self.data_indices.data.output.full),), dtype=torch.float32)

        for key, idx in self.data_indices.model.output.name_to_index.items():
            if idx in self.data_indices.model.output.prognostic and self.data_indices.data.output.name_to_index.get(
                key,
            ):
                prog_idx = self.data_indices.data.output.name_to_index[key]
                variable_stdev = self.statistics["stdev"][prog_idx] if self.statistics_tendencies else 1
                variable_tendency_stdev = (
                    self.statistics_tendencies["stdev"][prog_idx] if self.statistics_tendencies else 1
                )
                scaling = self.get_level_scaling(variable_stdev, variable_tendency_stdev)
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
