# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn

from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.layers.activations import leaky_hardtanh


class BaseBounding(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    This class defines an interface for bounding strategies which are used to apply a specific
    restriction to the predictions of a model.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        """Initializes the bounding strategy.
        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict, optional
            A dictionary containing the statistics of the variables.
        name_to_index_stats : dict, optional
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """
        super().__init__()

        self.name_to_index = name_to_index
        self.variables = variables
        self.data_index = self._create_index(variables=self.variables)
        self.statistics = statistics
        self.name_to_index_stats = name_to_index_stats

    def _create_index(self, variables: list[str]) -> InputTensorIndex:
        return InputTensorIndex(includes=variables, excludes=[], name_to_index=self.name_to_index)._only

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the bounding to the predictions.

        Parameters
        ----------
        x : torch.Tensor
            The tensor containing the predictions that will be bounded.

        Returns
        -------
        torch.Tensor
        A tensor with the bounding applied.
        """
        pass


class ReluBounding(BaseBounding):
    """Initializes the bounding with a ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.relu(x[..., self.data_index])
        return x


class LeakyReluBounding(BaseBounding):
    """Initializes the bounding with a Leaky ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.leaky_relu(x[..., self.data_index])
        return x


class NormalizedReluBounding(BaseBounding):
    """Bounding variable with a ReLU activation and customizable normalized thresholds."""

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: list[float],
        normalizer: list[str],
        statistics: dict,
        name_to_index_stats: dict,
    ) -> None:
        """Initializes the NormalizedReluBounding with the specified parameters.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict
            A dictionary containing the statistics of the variables (mean, std, min, max, etc.).
        min_val : list[float]
            The minimum values for the ReLU activation. It should be given in the same order as the variables.
        normalizer : list[str]
            A list of normalization types to apply, one per variable. Options: 'mean-std', 'min-max', 'max', 'std'.
        name_to_index_stats : dict
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
        """
        super().__init__(
            variables=variables,
            name_to_index=name_to_index,
            statistics=statistics,
            name_to_index_stats=name_to_index_stats,
        )
        self.min_val = min_val
        self.normalizer = normalizer

        # Validate normalizer input
        if not all(norm in {"mean-std", "min-max", "max", "std"} for norm in self.normalizer):
            raise ValueError(
                "Each normalizer must be one of: 'mean-std', 'min-max', 'max', 'std' in NormalizedReluBounding."
            )
        if len(self.normalizer) != len(variables):
            raise ValueError(
                "The length of the normalizer list must match the number of variables in NormalizedReluBounding."
            )
        if len(self.min_val) != len(variables):
            raise ValueError(
                "The length of the min_val list must match the number of variables in NormalizedReluBounding."
            )

        # Compute normalized min values
        self.data_index = [name_to_index[var] for var in variables]
        self.norm_min_val = torch.zeros(len(variables))
        for ii, variable in enumerate(variables):
            stat_index = self.name_to_index_stats[variable]
            if self.normalizer[ii] == "mean-std":
                mean = self.statistics["mean"][stat_index]
                std = self.statistics["stdev"][stat_index]
                self.norm_min_val[ii] = (min_val[ii] - mean) / std
            elif self.normalizer[ii] == "min-max":
                min_stat = self.statistics["min"][stat_index]
                max_stat = self.statistics["max"][stat_index]
                self.norm_min_val[ii] = (min_val[ii] - min_stat) / (max_stat - min_stat)
            elif self.normalizer[ii] == "max":
                max_stat = self.statistics["max"][stat_index]
                self.norm_min_val[ii] = min_val[ii] / max_stat
            elif self.normalizer[ii] == "std":
                std = self.statistics["stdev"][stat_index]
                self.norm_min_val[ii] = min_val[ii] / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ReLU activation with the normalized minimum values to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to process.

        Returns
        -------
        torch.Tensor
            The processed tensor with bounding applied.
        """
        self.norm_min_val = self.norm_min_val.to(x.device)
        x[..., self.data_index] = (
            torch.nn.functional.relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class LeakyNormalizedReluBounding(NormalizedReluBounding):
    """Initializes the bounding with a Leaky ReLU activation and customizable normalized thresholds."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = (
            torch.nn.functional.leaky_relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class HardtanhBounding(BaseBounding):
    """Initializes the bounding with specified minimum and maximum values for bounding.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    statistics : dict, optional
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict, optional
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        return x


class LeakyHardtanhBounding(HardtanhBounding):
    """Initializes the bounding with a Leaky HardTanh activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = leaky_hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
        return x


class FractionBounding(BaseBounding):
    """Initializes the FractionBounding with specified parameters.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    total_var : str
        A string representing a variable from which a secondary variable is derived. For
        example, in the case of convective precipitation (Cp), total_var = Tp (total precipitation).
    statistics : dict, optional
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict, optional
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        total_var: str,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val
        self.total_variable = self._create_index(variables=[total_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the HardTanh bounding  to the data_index variables
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x


class LeakyFractionBounding(FractionBounding):
    """Initializes the bounding with a Leaky HardTanh activation and a fraction of the total variable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the LeakyHardTanh bounding  to the data_index variables
        x[..., self.data_index] = leaky_hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x
