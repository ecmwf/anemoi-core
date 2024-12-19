# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BasePreprocessor(nn.Module):
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

        self.default, self.remap, self.method_config = self._process_config(config)
        self.methods = self._invert_key_value_list(self.method_config)

        self.data_indices = data_indices

    @classmethod
    def _process_config(cls, config):
        _special_keys = ["default", "remap"]  # Keys that do not contain a list of variables in a preprocessing method.
        default = config.get("default", "none")
        remap = config.get("remap", {})
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

        return default, remap, method_config

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

    def forward(self, x, in_place: bool = True, inverse: bool = False) -> Tensor:
        """Process the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        in_place : bool
            Whether to process the tensor in place
        inverse : bool
            Whether to inverse transform the input

        Returns
        -------
        torch.Tensor
            Processed tensor
        """
        if inverse:
            return self.inverse_transform(x, in_place=in_place)
        return self.transform(x, in_place=in_place)

    def transform(self, x, in_place: bool = True) -> Tensor:
        """Process the input tensor."""
        if not in_place:
            x = x.clone()
        return x

    def inverse_transform(self, x, in_place: bool = True) -> Tensor:
        """Inverse process the input tensor."""
        if not in_place:
            x = x.clone()
        return x


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

    def forward(self, x, in_place: bool = True) -> Tensor:
        """Process the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        in_place : bool
            Whether to process the tensor in place

        Returns
        -------
        torch.Tensor
            Processed tensor
        """
        for processor in self.processors.values():
            x = processor(x, in_place=in_place, inverse=self.inverse)

        if self.first_run:
            self.first_run = False
            self._run_checks(x)
        return x

    def _run_checks(self, x):
        """Run checks on the processed tensor."""
        if not self.inverse:
            # Forward transformation checks:
            assert not torch.isnan(
                x
            ).any(), f"NaNs ({torch.isnan(x).sum()}) found in processed tensor after {self.__class__.__name__}."
