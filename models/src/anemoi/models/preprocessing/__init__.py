# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
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

    #: Whether the processor constrains model outputs (clipping, masking) rather than changing their
    #: representation. Constraints apply to predicted states only: a tendency is converted without
    #: them, see ``Processors(..., skip_output_constraints=True)``.
    constrains_output = False

    #: Whether the processor applies to tendencies (differences of states, normalised with tendency
    #: statistics computed on raw differences). ``False`` for output constraints and for nonlinear or
    #: shifted remappings of the state values: a tendency is converted without them, see
    #: ``Processors(..., tendency=True)``.
    applies_to_tendencies = True

    #: ``True`` for the normaliser: its ``remap`` (normalise a variable with the statistics of another,
    #: e.g. ``cp: tp``) also applies to the processors after it in the chain that normalise physical
    #: values (:attr:`uses_statistics_remap`), and to the output boundings. This keeps the behaviour of
    #: earlier releases, where the normaliser rewrote the shared statistics in place when it was built.
    shares_statistics_remap = False

    #: ``True`` for processors that normalise physical values (e.g. thresholds) with the view's
    #: statistics. :class:`Processors` sets their ``statistics_remap`` to the remap of the normalisers
    #: before them in the chain (:attr:`shares_statistics_remap`).
    uses_statistics_remap = False

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the preprocessor.

        Processors do not hold statistics: each ``transform``/``inverse_transform`` call receives the
        ``statistics`` and ``name_to_index`` of the view it processes, through
        :meth:`~anemoi.models.data.sources.Source.apply_func`.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict, optional
            Deprecated and ignored; kept so that subclasses still passing it keep working.

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

        if statistics is not None:
            warnings.warn(
                f"{self.__class__.__name__}: the 'statistics' constructor argument is deprecated and ignored. "
                "Processors use the statistics carried by each Source view instead.",
                FutureWarning,
                stacklevel=2,
            )

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

        # The normaliser remap reaches the processors listed after the normaliser (in config order),
        # as when the normaliser rewrote the shared statistics at construction; see
        # BasePreprocessor.shares_statistics_remap. The full remap is kept for the output boundings.
        remap = {}
        for _, processor in processors:
            if getattr(processor, "uses_statistics_remap", False):
                processor.statistics_remap = dict(remap)
            if getattr(processor, "shares_statistics_remap", False):
                remap.update(processor.remap)
        self.statistics_remap = remap

        if inverse:
            # Reverse the order of processors for inverse transformation
            # e.g. first impute then normalise forward but denormalise then de-impute for inverse
            processors = processors[::-1]

        self.processors = nn.ModuleDict(processors)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [{'inverse' if self.inverse else 'forward'}]({self.processors})"

    @property
    def has_state_only_processors(self) -> bool:
        """Whether any processor applies to states only (see :attr:`BasePreprocessor.applies_to_tendencies`)."""
        return any(not getattr(processor, "applies_to_tendencies", True) for processor in self.processors.values())

    def forward(
        self,
        x: Source,
        in_place: bool = True,
        skip_output_constraints: bool = False,
        tendency: bool = False,
        **kwargs,
    ) -> Source:
        """Process the input tensor.

        Parameters
        ----------
        x : Source
            Input tensor.
        in_place : bool
            Whether to process the tensor in place.
        skip_output_constraints : bool
            Skip the processors that constrain outputs (clipping, masking), and apply only the
            processors that change the representation (normalisation, remapping). Used to convert
            states between their normalised and physical forms without constraining them.
        tendency : bool
            ``x`` holds tendencies: skip the processors that apply to states only
            (:attr:`BasePreprocessor.applies_to_tendencies` is ``False``), i.e. output constraints and
            remappings, so the tendency is only normalised with the statistics it carries.
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
            if skip_output_constraints and getattr(processor, "constrains_output", False):
                continue
            if tendency and not getattr(processor, "applies_to_tendencies", True):
                continue
            x = processor(x, in_place=in_place, inverse=self.inverse, **kwargs)

        return x
