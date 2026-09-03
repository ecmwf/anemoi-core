# (C) Copyright 2024-2026 Anemoi contributors.
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
from typing import Any
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.layers.activations import leaky_hardtanh


class BaseBounding(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    Stateless: resolves variable indices from the SourceView's
    ``name_to_index`` at forward time rather than storing them at init.
    """

    def __init__(self, *, variables: list[str], **kwargs) -> None:
        """Initializes the bounding strategy.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        """
        super().__init__()
        for var in variables:
            if var not in name_to_index:
                raise KeyError(
                    f"{self.__class__.__name__}: variable '{var}' is not present in the name_to_index mapping."
                )

        self.name_to_index = name_to_index
        self.variables = variables

    def _get_indices(self, name_to_index: dict[str, int]) -> torch.Tensor:
        """Resolve variable indices from name_to_index mapping."""
        return torch.tensor([name_to_index[name] for name in self.variables if name in name_to_index], dtype=torch.long)

    @abstractmethod
    def bound(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Apply bounding to the specified indices in the data tensor."""
        ...

    def forward(self, x: "SourceView") -> "SourceView":
        """Applies the bounding to the predictions.

        Parameters
        ----------
        x : SourceView
            The source view containing the predictions that will be bounded.

        Returns
        -------
        SourceView
            A source view with the bounding applied.
        """
        indices = self._get_indices(x.name_to_index).to(x.device)
        x = x.apply_func(self.bound, indices=indices)
        return x


class ReluBounding(BaseBounding):
    """Bounding with a ReLU activation / zero clamping."""

    def bound(self, data: torch.Tensor, indices: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.relu(data[..., indices])
        return data


class LeakyReluBounding(BaseBounding):
    """Bounding with a Leaky ReLU activation / zero clamping."""

    def bound(self, data: torch.Tensor, indices: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.leaky_relu(data[..., indices])
        return data


class NormalizedReluBounding(BaseBounding):
    """Bounding with a ReLU activation and customizable normalized thresholds."""

    def __init__(
        self,
        *,
        variables: list[str],
        min_val: list[float],
        normalizer: list[str],
        **kwargs,
    ) -> None:
        """Initializes the NormalizedReluBounding.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        min_val : list[float]
            The minimum values for the ReLU activation, in the same order as variables.
        normalizer : list[str]
            Normalization types per variable: 'mean-std', 'min-max', 'max', 'std'.
        """
        super().__init__(variables=variables)
        self.min_val = min_val
        self.normalizer = normalizer

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

    def _compute_norm_min_val(self, name_to_index: dict[str, int], statistics: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute normalized minimum values from statistics."""
        norm_min_val = torch.zeros(len(self.variables))
        for ii, variable in enumerate(self.variables):
            stat_index = name_to_index[variable]
            if self.normalizer[ii] == "mean-std":
                mean = statistics["mean"][stat_index]
                std = statistics["stdev"][stat_index]
                norm_min_val[ii] = (self.min_val[ii] - mean) / std
            elif self.normalizer[ii] == "min-max":
                min_stat = statistics["min"][stat_index]
                max_stat = statistics["max"][stat_index]
                norm_min_val[ii] = (self.min_val[ii] - min_stat) / (max_stat - min_stat)
            elif self.normalizer[ii] == "max":
                max_stat = statistics["max"][stat_index]
                norm_min_val[ii] = self.min_val[ii] / max_stat
            elif self.normalizer[ii] == "std":
                std = statistics["stdev"][stat_index]
                norm_min_val[ii] = self.min_val[ii] / std
        return norm_min_val

    def bound(self, data: torch.Tensor, indices: torch.Tensor, norm_min_val: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.relu(data[..., indices] - norm_min_val) + norm_min_val
        return data

    def forward(self, x: "SourceView") -> "SourceView":
        data_index = self._get_indices(x.name_to_index)
        norm_min_val = self._compute_norm_min_val(x.name_to_index, x.statistics).to(x.data.device)
        x = x.apply_func(self.bound, indices=data_index, norm_min_val=norm_min_val)
        return x


class NormalizedLeakyReluBounding(NormalizedReluBounding):
    """Bounding with a Leaky ReLU activation and customizable normalized thresholds."""

    def bound(self, data: torch.Tensor, indices: torch.Tensor, norm_min_val: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.leaky_relu(data[..., indices] - norm_min_val) + norm_min_val
        return data


class HardtanhBounding(BaseBounding):
    """Bounding with specified minimum and maximum values.

    Parameters
    ----------
    variables : list[str]
        Variables that will be bounded.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    """

    def __init__(self, *, variables: list[str], min_val: float, max_val: float, **kwargs) -> None:
        super().__init__(variables=variables)
        self.min_val = min_val
        self.max_val = max_val

    def bound(self, data: torch.Tensor, indices: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.hardtanh(
            data[..., indices], min_val=self.min_val, max_val=self.max_val
        )
        return data


class LeakyHardtanhBounding(HardtanhBounding):
    """Bounding with a Leaky HardTanh activation."""

    def bound(self, data: torch.Tensor, indices: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = leaky_hardtanh(data[..., indices], min_val=self.min_val, max_val=self.max_val)
        return data


class FractionBounding(BaseBounding):
    """Bounding as a fraction of a total variable.

    Parameters
    ----------
    variables : list[str]
        Variables that will be bounded.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    total_var : str
        Variable from which fractions are derived (e.g. total precipitation).
    """

    def __init__(self, *, variables: list[str], min_val: float, max_val: float, total_var: str, **kwargs) -> None:
        super().__init__(variables=variables)
        self.min_val = min_val
        self.max_val = max_val
        self.total_var = total_var

    def bound(
        self, data: torch.Tensor, indices: torch.Tensor, name_to_index: dict[str, int], **_kwargs
    ) -> torch.Tensor:
        total_index = torch.tensor([name_to_index[self.total_var]], dtype=torch.long)
        data[..., indices] = torch.nn.functional.hardtanh(
            data[..., indices], min_val=self.min_val, max_val=self.max_val
        )
        data[..., indices] *= data[..., total_index]
        return data


class LeakyFractionBounding(FractionBounding):
    """Bounding with a Leaky HardTanh activation and a fraction of the total variable."""

    def bound(
        self, data: torch.Tensor, indices: torch.Tensor, name_to_index: dict[str, int], **_kwargs
    ) -> torch.Tensor:
        total_index = torch.tensor([name_to_index[self.total_var]], dtype=torch.long)
        data[..., indices] = torch.nn.functional.leaky_hardtanh(
            data[..., indices], min_val=self.min_val, max_val=self.max_val
        )
        data[..., indices] *= data[..., total_index]
        return data


def _build_dataset_boundings(
    bounding_cfgs: Any,
    data_indices: IndexCollection,
    statistics: dict | None,
) -> nn.ModuleList:
    """Build the list of model-output bounding modules from configuration.

    This is a thin factory over Hydra's ``instantiate`` that reads the iterable
    ``model_config.model.bounding`` and instantiates each entry while injecting
    the common keyword arguments required by bounding modules:
    ``name_to_index``, ``statistics``, and ``name_to_index_stats``. The result
    is returned as an ``nn.ModuleList`` preserving the order of the config.

    Parameters
    ----------
    model_config : Any
        Object with a ``model`` attribute containing an iterable ``bounding``
        (e.g. a list of Hydra configs). If absent or empty, an empty
        ``nn.ModuleList`` is returned.
    data_indices : IndexCollection
        Object providing the mappings:
        ``data_indices.model.output.name_to_index`` and
        ``data_indices.data.input.name_to_index``. These are forwarded to each
        instantiated bounding module as ``name_to_index`` and
        ``name_to_index_stats`` respectively.
    statistics : dict[str, dict] | None
        Optional dataset/model statistics passed to each bounding module. Use
        ``None`` if not required by the configured classes.

    Returns
    -------
    nn.Sequential
        Sequence of bounding modules.
    """

    return nn.ModuleList(
        [
            instantiate(
                cfg,
                name_to_index=data_indices.model.output.name_to_index,
                statistics=statistics,
                name_to_index_stats=data_indices.data.input.name_to_index,
            )
            for cfg in bounding_cfgs
        ]
    )


def build_boundings(
    boundings_config: dict[str, list],
    data_indices: dict[str, IndexCollection],
    statistics: dict[str, dict] | None,
) -> nn.ModuleDict:
    """Build the model-output bounding modules from configuration.

    This is a thin factory that creates a ``nn.ModuleDict`` of bounding
    modules by invoking ``_build_dataset_boundings`` for each dataset
    specified in ``data_indices``.

    Parameters
    ----------
    boundings_config : dict[str, list]
        Dictionary containing iterables ``bounding`` (e.g. a list of Hydra configs).
        If absent or empty, an empty ``nn.ModuleDict`` is returned.
    data_indices : dict[str, IndexCollection]
        Dictionary mapping dataset names to data indices objects. Each
        data indices object must provide the mappings:
        ``data_indices.model.output.name_to_index`` and
        ``data_indices.data.input.name_to_index``. These are forwarded to each
        instantiated bounding module as ``name_to_index`` and
        ``name_to_index_stats`` respectively.
    statistics : dict[str, dict] | None
        Dictionary mapping dataset names to optional dataset/model statistics
        passed to each bounding module. Use ``None`` if not required by the
        configured classes.
    """
    bounding_modules = nn.ModuleDict()
    for dataset_name in data_indices.keys():
        bounding_modules[dataset_name] = _build_dataset_boundings(
            boundings_config.get(dataset_name, []),
            data_indices=data_indices[dataset_name],
            statistics=statistics[dataset_name],
        )

    return bounding_modules
