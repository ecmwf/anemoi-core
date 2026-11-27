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
from typing import TYPE_CHECKING
from typing import Any

import torch
from hydra.utils import instantiate
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.activations import leaky_hardtanh
from anemoi.models.layers.forcing_functions import physical_forcing
from anemoi.models.preprocessing.normalization import check_normalization_method
from anemoi.models.preprocessing.normalization import missing_statistics
from anemoi.models.preprocessing.normalization import normalize_value
from anemoi.models.preprocessing.normalization import remapped_statistics

if TYPE_CHECKING:
    from anemoi.models.data import Source


class BaseBounding(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    Stateless: resolves variable indices from the Source's
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
        self.variables = variables

    def _get_indices(self, name_to_index: dict[str, int]) -> torch.Tensor:
        """Resolve variable indices from name_to_index mapping."""
        return torch.tensor([name_to_index[name] for name in self.variables if name in name_to_index], dtype=torch.long)

    @abstractmethod
    def bound(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Apply bounding to the specified indices in the data tensor."""
        ...

    def forward(self, x: "Source") -> "Source":
        """Applies the bounding to the predictions.

        Parameters
        ----------
        x : Source
            The source view containing the predictions that will be bounded.

        Returns
        -------
        Source
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

    #: The model interface sets ``statistics_remap`` to the dataset's normaliser remap.
    uses_statistics_remap = True

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
            Normalization method per variable, as configured for the variable in the
            :class:`~anemoi.models.preprocessing.normalizer.InputNormalizer` (see
            :data:`~anemoi.models.preprocessing.normalization.NORMALIZATION_STATISTICS`).
        """
        super().__init__(variables=variables)
        self.min_val = min_val
        self.normalizer = normalizer

        for method in self.normalizer:
            check_normalization_method(method, self.__class__.__name__)
        if len(self.normalizer) != len(variables):
            raise ValueError(
                "The length of the normalizer list must match the number of variables in NormalizedReluBounding."
            )
        if len(self.min_val) != len(variables):
            raise ValueError(
                "The length of the min_val list must match the number of variables in NormalizedReluBounding."
            )

    def _compute_norm_min_val(self, name_to_index: dict[str, int], statistics: dict) -> torch.Tensor:
        """Normalise the physical minimum values with the view's statistics.

        Variables the normaliser remaps (e.g. ``cp: tp``) use the remapped statistics, as the
        normaliser does (``statistics_remap``, set by the model interface; see
        :attr:`~anemoi.models.preprocessing.BasePreprocessor.shares_statistics_remap`).
        """
        missing = missing_statistics(self.normalizer, statistics or {})
        if missing:
            msg = f"{self.__class__.__name__} with normalizers {self.normalizer} needs the statistics {missing}."
            raise ValueError(msg)
        remap = getattr(self, "statistics_remap", None) or {}
        if remap:
            statistics = remapped_statistics(statistics, name_to_index, remap, self.__class__.__name__)
        norm_min_val = torch.zeros(len(self.variables))
        for ii, (variable, method) in enumerate(zip(self.variables, self.normalizer)):
            norm_min_val[ii] = normalize_value(method, self.min_val[ii], statistics, name_to_index[variable])
        return norm_min_val

    def bound(self, data: torch.Tensor, indices: torch.Tensor, norm_min_val: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.relu(data[..., indices] - norm_min_val) + norm_min_val
        return data

    def forward(self, x: "Source") -> "Source":
        data_index = self._get_indices(x.name_to_index)
        norm_min_val = self._compute_norm_min_val(x.name_to_index, x.statistics).to(x.device)
        x = x.apply_func(self.bound, indices=data_index, norm_min_val=norm_min_val)
        return x


class NormalizedLeakyReluBounding(NormalizedReluBounding):
    """Bounding with a Leaky ReLU activation and customizable normalized thresholds."""

    def bound(self, data: torch.Tensor, indices: torch.Tensor, norm_min_val: torch.Tensor, **_kwargs) -> torch.Tensor:
        data[..., indices] = torch.nn.functional.leaky_relu(data[..., indices] - norm_min_val) + norm_min_val
        return data


class ForcingMaskBounding(NormalizedReluBounding):
    """Set variables to a fill value where a condition on the output-time forcings holds.

    The condition is ``mask <operator> threshold``, evaluated in physical units at each output time. ``mask`` is
    either the name of a forcing, read from the target forcings, or a callable computing a quantity from the
    target forcings, such as :class:`~anemoi.models.layers.forcing_functions.MaxInsolationOverWindow`.

    Example: zero accumulated radiation where the sun was down for the whole accumulation period::

        _target_: anemoi.models.layers.bounding.ForcingMaskBounding
        variables: [ssr, ssrd, fdir]
        normalizer: [mean-std, mean-std, mean-std]
        mask:
          _target_: anemoi.models.layers.forcing_functions.MaxInsolationOverWindow
          window: 6h  # the model's output timestep

    Parameters
    ----------
    variables : list[str]
        Variables to overwrite.
    normalizer : list[str]
        Normalisation method of each variable, to normalise the fill value.
    mask : str | Callable[[Source], torch.Tensor]
        Forcing name, or callable returning a ``(batch, time, ensemble, grid)`` tensor in physical units.
    operator : str, optional
        ``lt``, ``le``, ``gt`` or ``ge``, by default ``le``.
    threshold : float, optional
        Threshold in physical units, by default 0.
    fill_value : float, optional
        Value in physical units, by default 0.
    mask_normalizer : str, optional
        Normalisation method of the forcing when ``mask`` is a forcing name, by default ``none``.
    """

    #: The model passes the output-time forcings of the dataset to ``forward``.
    uses_target_forcings = True

    OPERATORS = {"lt": torch.lt, "le": torch.le, "gt": torch.gt, "ge": torch.ge}

    def __init__(
        self,
        *,
        variables: list[str],
        normalizer: list[str],
        mask: Any,
        operator: str = "le",
        threshold: float = 0.0,
        fill_value: float = 0.0,
        mask_normalizer: str = "none",
        **kwargs,
    ) -> None:
        super().__init__(variables=variables, min_val=[float(fill_value)] * len(variables), normalizer=normalizer)
        if operator not in self.OPERATORS:
            msg = f"{self.__class__.__name__}: unknown operator {operator!r}, expected one of {list(self.OPERATORS)}."
            raise ValueError(msg)
        if not (isinstance(mask, str) or callable(mask)):
            raise TypeError(f"{self.__class__.__name__}: mask must be a forcing name or a callable, got {mask!r}.")
        check_normalization_method(mask_normalizer, self.__class__.__name__)
        self.mask = mask
        self.operator = operator
        self.threshold = float(threshold)
        self.mask_normalizer = mask_normalizer

    def forward(self, x: "Source", target_forcings: "Source | None" = None) -> "Source":
        if target_forcings is None:
            raise ValueError(f"{self.__class__.__name__} needs the output-time forcings (target_forcings).")
        if x.is_tabular or target_forcings.is_tabular:
            raise ValueError(f"{self.__class__.__name__} supports gridded datasets only.")

        if isinstance(self.mask, str):
            value = physical_forcing(target_forcings, self.mask, self.mask_normalizer)
        else:
            value = self.mask(target_forcings)
        condition = self.OPERATORS[self.operator](value, self.threshold)  # (batch, time, ensemble, grid)
        if condition.shape[-1] != x.grid_size or condition.shape[1] != x.time_size:
            raise ValueError(
                f"{self.__class__.__name__}: the mask has shape {tuple(condition.shape)} but the predictions have "
                f"{x.time_size} times and {x.grid_size} grid points."
            )

        data_index = self._get_indices(x.name_to_index).to(x.device)
        fill = self._compute_norm_min_val(x.name_to_index, x.statistics).to(x.device)
        return x.apply_func(self.bound, indices=data_index, norm_min_val=fill, condition=condition)

    def bound(
        self, data: torch.Tensor, indices: torch.Tensor, norm_min_val: torch.Tensor, condition: torch.Tensor, **_kwargs
    ) -> torch.Tensor:
        fill = norm_min_val.to(device=data.device, dtype=data.dtype)
        data[..., indices] = torch.where(condition.unsqueeze(-1).to(data.device), fill, data[..., indices])
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
        data[..., indices] = leaky_hardtanh(data[..., indices], min_val=self.min_val, max_val=self.max_val)
        data[..., indices] *= data[..., total_index]
        return data


def apply_boundings(boundings: nn.Module, x: "Source", target_forcings: "Source | None" = None) -> "Source":
    """Apply the boundings of a dataset to its predictions, in order.

    ``boundings`` is the module built by :func:`build_boundings` (a ``nn.Sequential``), but a single module or a
    ``nn.ModuleList`` is accepted too. Boundings with ``uses_target_forcings`` also receive the output-time forcings.
    """
    modules = boundings if isinstance(boundings, (nn.Sequential, nn.ModuleList)) else [boundings]
    for bounding in modules:
        if getattr(bounding, "uses_target_forcings", False):
            x = bounding(x, target_forcings=target_forcings)
        else:
            x = bounding(x)
    return x


def _build_dataset_boundings(
    bounding_cfgs: Any,
    data_indices: IndexCollection,
    statistics: dict | None,
) -> nn.Module:
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
    nn.Module
        Sequence of bounding modules.
    """
    return nn.Sequential(
        *(
            instantiate(
                cfg,
                name_to_index=data_indices.model.output.name_to_index,
                statistics=statistics,
                name_to_index_stats=data_indices.data.input.name_to_index,
            )
            for cfg in bounding_cfgs
        )
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
    for dataset_name, data_indices_obj in data_indices.items():
        bounding_modules[dataset_name] = _build_dataset_boundings(
            boundings_config.get(dataset_name, []),
            data_indices=data_indices_obj,
            statistics=statistics[dataset_name],
        )

    return bounding_modules
