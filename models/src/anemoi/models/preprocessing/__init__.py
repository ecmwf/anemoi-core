# (C) Copyright 2024-2026 Anemoi contributors.
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
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn

from anemoi.models.data.sources.base import Source
from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


def resolve_variable_indices(
    processor: str,
    names: Sequence[str],
    width: int,
    name_to_index: Mapping[str, int] | None,
    layouts: Mapping[str, tuple[int, Sequence[int | None]]],
) -> list[int | None]:
    """Return where each of ``names`` sits along the variable axis of a tensor with ``width`` variables.

    With ``name_to_index``, positions come from the variable names;
    a name the tensor does not carry gets ``None``. Without it, the layout is
    inferred from the width among ``layouts``. 
    
    A width that matches no layout will raise a ValueError.

    Parameters
    ----------
    processor : str
        Processor name, for error messages.
    names : Sequence[str]
        Variables to locate.
    width : int
        Size of the tensor's variable axis.
    name_to_index : Mapping[str, int] | None
        The tensor's variable positions, if known.
    layouts : Mapping[str, tuple[int, Sequence[int | None]]]
        Width-inference fallback: ``{label: (number_of_variables, positions_of_names)}``.

    Returns
    -------
    list[int | None]
        One position (or ``None``) per name.
    """
    if name_to_index is not None:
        if len(name_to_index) != width:
            msg = f"{processor}: the tensor has {width} variables, but name_to_index lists {len(name_to_index)}."
            raise ValueError(msg)
        return [name_to_index.get(name) for name in names]

    matches = {label: list(indices) for label, (n_variables, indices) in layouts.items() if n_variables == width}
    if not matches:
        expected = {label: n_variables for label, (n_variables, _) in layouts.items()}
        msg = f"{processor}: a tensor with {width} variables matches none of the known layouts {expected}."
        raise ValueError(msg)

    if len({tuple(indices) for indices in matches.values()}) > 1:
        msg = (
            f"{processor}: a tensor with {width} variables matches the layouts {sorted(matches)}, which place "
            "the variables differently. Pass name_to_index to select the variables by name."
        )
        raise ValueError(msg)

    return next(iter(matches.values()))


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
            dictionary of the methods with lists of variables.

        Returns
        -------
        dict[str, str]
            dictionary of the variables with methods.
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

    def forward(
        self,
        x: Source,
        in_place: bool = True,
        inverse: bool = False,
        **kwargs,
    ) -> Source:
        """Process the input tensor.

        Parameters
        ----------
        x : Source
            Input tensor.
        in_place : bool
            Whether to process the tensor in place.
        inverse : bool
            Whether to inverse transform the input.
        **kwargs
            Additional keyword arguments to pass to transform/inverse_transform.

        Returns
        -------
        Source
            Processed tensor.
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
        # self.first_run = True

        if inverse:
            # Reverse the order of processors for inverse transformation
            # e.g. first impute then normalise forward but denormalise then de-impute for inverse
            processors = processors[::-1]

        self.processors = nn.ModuleDict(processors)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [{'inverse' if self.inverse else 'forward'}]({self.processors})"

    def forward(self, x: Source, in_place: bool = True, **kwargs) -> Source:
        """Process the input tensor.

        Parameters
        ----------
        x : Source
            Input tensor.
        in_place : bool
            Whether to process the tensor in place.
        **kwargs
            Additional keyword arguments to pass to processors.

        Returns
        -------
        Source
            Processed tensor.
        """
        for processor in self.processors.values():
            if self.inverse and getattr(processor, "supports_skip_imputation", False):
                continue
            x = processor(x, in_place=in_place, inverse=self.inverse, **kwargs)

        return x
