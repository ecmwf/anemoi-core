# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from collections.abc import Mapping

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseResidualScaler(BaseScaler):
    """Configurable method to scale prognostic variables based on data statistics and statistics_residuals.

    Residual normalisation is never step-dependent: the model learns
    ``target - interp(source)``, a single quantity with no lead-time or timestep structure, so
    unlike tendency statistics ``statistics_residuals`` is a flat per-dataset mapping (no
    ``lead_times`` key, no per-step selection).
    """

    scale_dims: TensorDim = TensorDim.VARIABLE

    #: Whether this scaler requires ``statistics_residuals`` to be provided. Set to ``False`` on
    #: subclasses that do not consume any statistics (mirrors ``NoTendencyScaler``, which tolerates
    #: a missing ``statistics_tendencies`` with a warning rather than an error).
    requires_statistics_residuals: bool = True

    def __init__(
        self,
        data_indices: IndexCollection,
        statistics: dict,
        statistics_residuals: Mapping | None = None,
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
        statistics_residuals : dict, optional
            Flat data statistics dictionary for residuals, keyed directly by statistic name
            (e.g. "stdev"). Never a fallback to ``statistics``: residual scalers that consume
            statistics require it to be given explicitly.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        del kwargs
        self.data_indices = data_indices
        self.statistics = statistics

        if self.requires_statistics_residuals and (
            not isinstance(statistics_residuals, Mapping) or "stdev" not in statistics_residuals
        ):
            error_msg = (
                f"{self.__class__.__name__} requires explicit statistics_residuals containing a 'stdev' "
                "array; state statistics are never used as a fallback for residual normalisation."
            )
            raise ValueError(error_msg)

        self.statistics_residuals = statistics_residuals

    @abstractmethod
    def get_level_scaling(self, variable_stdev: float, variable_residual_stdev: float) -> float: ...

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        variable_level_scaling = torch.ones((len(self.data_indices.data.output.full),), dtype=torch.float32)

        for key, idx in self.data_indices.model.output.name_to_index.items():
            if idx in self.data_indices.model.output.prognostic and self.data_indices.data.output.name_to_index.get(
                key,
            ):
                prog_idx = self.data_indices.data.output.name_to_index[key]
                variable_stdev = self.statistics["stdev"][prog_idx] if self.statistics_residuals else 1
                variable_residual_stdev = (
                    self.statistics_residuals["stdev"][prog_idx] if self.statistics_residuals else 1
                )
                scaling = self.get_level_scaling(variable_stdev, variable_residual_stdev)
                variable_level_scaling[idx] *= scaling

        return variable_level_scaling


class NoResidualScaler(BaseResidualScaler):
    """No scaling by residual statistics."""

    requires_statistics_residuals = False

    def get_level_scaling(self, variable_stdev: float, variable_residual_stdev: float) -> float:
        del variable_stdev, variable_residual_stdev
        return 1.0


class StdevResidualScaler(BaseResidualScaler):
    """Scale losses by standard deviation of residual statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_residual_stdev: float) -> float:
        return variable_stdev / variable_residual_stdev


class VarResidualScaler(BaseResidualScaler):
    """Scale losses by variance of residual statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_residual_stdev: float) -> float:
        return variable_stdev**2 / variable_residual_stdev**2
