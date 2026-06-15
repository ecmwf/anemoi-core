# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn

from anemoi.models.data import SourceView
from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BasePreprocessor(nn.Module, ABC):
    """Base class for data pre- and post-processors."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the preprocessor.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        data_indices : dict
            Data indices for input and output variables

        Attributes
        ----------
        default : str
            Default method for variables not specified in the config
        method_config : dict
            Dictionary of the methods with lists of variables
        methods : dict
            Dictionary of the variables with methods
        data_indices : IndexCollection
            Data indices for input and output variables
        remap : dict
            Dictionary of the variables with remapped names in the config
        """

        super().__init__()

        self.default, self.remap, self.normalizer, self.method_config, self.method_kwargs = self._process_config(config)
        self.methods = self._invert_key_value_list(self.method_config)

        self.data_indices = data_indices

    @classmethod
    def _process_config(cls, config):
        _special_keys = [
            "default",
            "remap",
            "normalizer",
            "method_kwargs",
        ]  # Keys that do not contain a list of variables in a preprocessing method.
        default = config.get("default", "none")
        remap = config.get("remap", {})
        normalizer = config.get("normalizer", "none")
        method_kwargs = config.get("method_kwargs", {})
        method_config = {k: v for k, v in config.items() if k not in _special_keys and v is not None and v != "none"}

        if not method_config:
            LOGGER.warning(
                f"{cls.__name__}: Using default method {default} for all variables not specified in the config.",
            )
        for m in method_config:
            if isinstance(method_config[m], str):
                method_config[m] = {method_config[m]: f"{m}_{method_config[m]}"}
            elif isinstance(method_config[m], list):
                method_config[m] = {method: f"{m}_{method}" for method in method_config[m]}

        return default, remap, normalizer, method_config, method_kwargs

    @staticmethod
    def _invert_key_value_list(method_config: dict[str, list[str]]) -> dict[str, str]:
        """Invert a dictionary of methods with lists of variables.

        Parameters
        ----------
        method_config : dict[str, list[str]]
            dictionary of the methods with lists of variables

        Returns
        -------
        dict[str, str]
            dictionary of the variables with methods
        """
        return {
            variable: method
            for method, variables in method_config.items()
            if not isinstance(variables, str)
            for variable in variables
        }

    @abstractmethod
    def transform(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Transform the input tensor."""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse transform the input tensor."""
        raise NotImplementedError

    def forward(self, x: SourceView, in_place: bool = True, inverse: bool = False, **kwargs) -> SourceView:
        """Process the input tensor.

        Parameters
        ----------
        x : SourceView
            Input tensor
        in_place : bool
            Whether to process the tensor in place
        inverse : bool
            Whether to inverse transform the input
        **kwargs
            Additional keyword arguments to pass to transform/inverse_transform

        Returns
        -------
        SourceView
            Processed tensor
        """
        if "skip_imputation" in kwargs and not getattr(self, "supports_skip_imputation", False):
            kwargs = {key: value for key, value in kwargs.items() if key != "skip_imputation"}

        if inverse:
            return x.apply_func(self.inverse_transform, in_place=in_place, **kwargs)

        return x.apply_func(self.transform, in_place=in_place, **kwargs)


class Processors(nn.Module):
    """A collection of processors."""

    def __init__(self, processors: list, inverse: bool = False) -> None:
        """Initialize the processors.

        Parameters
        ----------
        processors : list
            List of processors
        """
        super().__init__()

        self.inverse = inverse
        self.first_run = True

        if inverse:
            # Reverse the order of processors for inverse transformation
            # e.g. first impute then normalise forward but denormalise then de-impute for inverse
            processors = processors[::-1]

        self.processors = nn.ModuleDict(processors)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [{'inverse' if self.inverse else 'forward'}]({self.processors})"

    def forward(self, x: SourceView, in_place: bool = True, **kwargs) -> SourceView:
        """Process the input tensor.

        Parameters
        ----------
        x : SourceView
            Input tensor
        in_place : bool
            Whether to process the tensor in place
        **kwargs
            Additional keyword arguments to pass to processors

        Returns
        -------
        SourceView
            Processed tensor
        """
        for processor in self.processors.values():
            x = processor(x, in_place=in_place, inverse=self.inverse, **kwargs)

        return x


class StepwiseProcessors(nn.Module):
    """Ordered container for per-step processors that can include missing steps."""

    def __init__(self, lead_times: list[str]) -> None:
        super().__init__()
        self._lead_times = list(lead_times)
        self._processors = nn.ModuleDict()

    def __len__(self) -> int:
        return len(self._lead_times)

    def __iter__(self):
        for lead_time in self._lead_times:
            key = str(lead_time)
            yield self._processors[key] if key in self._processors else None

    def __getitem__(self, index: int | str) -> Optional["Processors"]:
        if isinstance(index, int):
            lead_time = self._lead_times[index]
        else:
            lead_time = str(index)
        key = str(lead_time)
        return self._processors[key] if key in self._processors else None

    @property
    def lead_times(self) -> list[str]:
        return list(self._lead_times)

    def set(self, lead_time: str, processors: "Processors") -> None:
        self._processors[str(lead_time)] = processors
