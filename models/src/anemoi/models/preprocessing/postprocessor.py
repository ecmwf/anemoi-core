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
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.activations import CustomRelu
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing import resolve_variable_indices
from anemoi.models.preprocessing.normalization import NORMALIZATION_STATISTICS
from anemoi.models.preprocessing.normalization import check_normalization_method
from anemoi.models.preprocessing.normalization import missing_statistics
from anemoi.models.preprocessing.normalization import normalize_value
from anemoi.models.preprocessing.normalization import remapped_statistics

LOGGER = logging.getLogger(__name__)


class Postprocessor(BasePreprocessor):
    """Class for Basic Postprocessors.

    For Postprocessors just the inverse_transform method is implemented.
    transform is not needed and corresponds to the identity function.

    Postprocessors constrain predicted states (:attr:`constrains_output`): they are skipped when a
    tendency is converted (:attr:`applies_to_tendencies`), and applied to the state rebuilt from it.
    """

    constrains_output = True
    applies_to_tendencies = False

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
    ) -> None:
        """Initialize the Postprocessor.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        """
        super().__init__(config, data_indices)

        self._prepare_postprocessing_indices_list()
        self._create_postprocessing_indices()

        self._validate_indices()

    def _validate_indices(self):
        assert (
            len(self.index_training_output) == len(self.index_inference_output) <= len(self.postprocessorfunctions)
        ), (
            f"Error creating postprocessing indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.postprocessorfunctions)}"
        )

    def _prepare_postprocessing_indices_list(self):
        """Prepare the postprocessor indices list."""
        self.num_training_output_vars = len(self.data_indices.data.output.name_to_index)
        self.num_inference_output_vars = len(self.data_indices.model.output.name_to_index)

        (
            self.postprocessed_names,
            self.index_training_output,
            self.index_inference_output,
            self.postprocessorfunctions,
        ) = ([], [], [], [])

    def _create_postprocessing_indices(self):
        """Create the indices for postprocessing."""

        # Create indices for postprocessing
        for name in self.data_indices.data.output.name_to_index:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"Postprocessor: skipping {name} as no postprocessing method is specified")
                continue
            assert name in self.data_indices.model.output.name_to_index, (
                f"Postprocessor: {name} not found in inference output indices. "
                f"Postprocessors cannot be applied to forcing variables."
            )

            self.postprocessed_names.append(name)
            self.index_training_output.append(self._get_index(self.data_indices.data.output.name_to_index, name))
            self.index_inference_output.append(self._get_index(self.data_indices.model.output.name_to_index, name))
            self.postprocessorfunctions.append(self._get_postprocessor_function(method, name))

    def transform(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Identity: postprocessors only act on the way out (``inverse_transform``)."""
        return x

    def _get_index(self, name_to_index_dict, name):
        return name_to_index_dict.get(name, None)

    def _get_postprocessor_function(self, method, name):
        if method == "relu":
            postprocessor_function = torch.nn.functional.relu
        elif method == "hardtanh":
            postprocessor_function = torch.nn.Hardtanh(min_val=-1, max_val=1)  # default hardtanh
        elif method == "hardtanh_0_1":
            postprocessor_function = torch.nn.Hardtanh(min_val=0, max_val=1)
        else:
            raise ValueError(f"Unknown postprocessing method: {method}")

        LOGGER.info(f"Postprocessor: applying {method} to {name}")
        return postprocessor_function

    def _output_indices(
        self,
        names: list[str],
        x: torch.Tensor,
        name_to_index: Optional[dict[str, int]],
        training_indices: list[int | None],
        inference_indices: list[int | None],
    ) -> list[int | None]:
        """Locate ``names`` in the model output ``x``: by name if ``name_to_index`` is given, else by width."""
        return resolve_variable_indices(
            self.__class__.__name__,
            names,
            x.shape[-1],
            name_to_index,
            {
                "training output": (self.num_training_output_vars, training_indices),
                "inference output": (self.num_inference_output_vars, inference_indices),
            },
        )

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        name_to_index: Optional[dict[str, int]] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Postprocess model output tensor."""
        if not in_place:
            x = x.clone()

        index = self._output_indices(
            self.postprocessed_names,
            x,
            name_to_index,
            self.index_training_output,
            self.index_inference_output,
        )

        # Replace values
        for postprocessor, idx_dst in zip(self.postprocessorfunctions, index):
            if idx_dst is not None:
                x[..., idx_dst] = postprocessor(x[..., idx_dst])
        return x


class NormalizedReluPostprocessor(Postprocessor):
    """Postprocess with a ReLU activation and customizable thresholds.

    Expects the config to have keys corresponding to customizable thresholds and lists of variables to postprocess and a normalizer to apply to thresholds.:
    ```
    normalizer: 'mean-std'
    1:
        - y
    0:
        - x
    3.14:
        - q
    ```
    Thresholds are in un-normalized (physical) space. If normalizer is specified, each threshold is
    normalized on every call with the ``statistics`` of the view being post-processed (passed in by
    :meth:`~anemoi.models.data.sources.Source.apply_func`), the same statistics the
    :class:`~anemoi.models.preprocessing.normalizer.InputNormalizer` uses. This is necessary if in
    config file the normalizer is specified before the postprocessor, e.g.:
    ```
    data:
        processors:
          normalizer:
            _target_: anemoi.models.preprocessing.normalizer.InputNormalizer
            config:
              default: "mean-std"
          normalized_relu_postprocessor:
            _target_: anemoi.models.preprocessing.postprocessor.NormalizedReluPostprocessor
            config:
              271.15:
              - x1
              0:
              - x2
              normalizer: 'mean-std'
    """

    #: Thresholds of variables the normaliser remaps (e.g. ``cp: tp``) are normalised with the remapped
    #: statistics when the normaliser comes first in the chain, see
    #: :attr:`~anemoi.models.preprocessing.BasePreprocessor.shares_statistics_remap`.
    uses_statistics_remap = True

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
    ) -> None:
        super().__init__(config, data_indices)

        check_normalization_method(self.normalizer, self.__class__.__name__)

    def _get_postprocessor_function(self, method: float, name: str) -> float:
        """Return the physical threshold for ``name``; it is normalized per call in :meth:`inverse_transform`."""
        LOGGER.info(
            "NormalizedReluPostprocessor: applying a ReLU with threshold %s (physical units, %s normalization) to %s.",
            method,
            self.normalizer,
            name,
        )
        return float(method)

    def _normalized_threshold(self, threshold: float, statistics: Optional[dict], index: int) -> float:
        """Map a physical ``threshold`` into the space of the variable at ``index`` of the view."""
        return normalize_value(self.normalizer, threshold, statistics or {}, index)

    def _check_statistics(self, statistics: Optional[dict], width: int) -> None:
        """Raise unless ``statistics`` holds what the normalizer needs, one value per variable of the view."""
        required = NORMALIZATION_STATISTICS[self.normalizer]
        if not required:
            return
        if statistics is None:
            msg = f"{self.__class__.__name__} with normalizer {self.normalizer!r} needs the view's statistics."
            raise ValueError(msg)
        missing = missing_statistics([self.normalizer], statistics)
        if missing:
            msg = f"{self.__class__.__name__} with normalizer {self.normalizer!r} needs the statistics {missing}."
            raise ValueError(msg)
        for key in required:
            if len(statistics[key]) != width:
                msg = (
                    f"{self.__class__.__name__}: statistic {key!r} has {len(statistics[key])} values, "
                    f"but the tensor has {width} variables."
                )
                raise ValueError(msg)

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        name_to_index: Optional[dict[str, int]] = None,
        statistics: Optional[dict] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Apply the thresholded ReLU, normalizing each threshold with the view's ``statistics``.

        ``statistics`` and ``name_to_index`` are those of the view (see
        :meth:`~anemoi.models.data.sources.Source.apply_func`): one value per variable along the
        last axis of ``x``. ``statistics`` may be omitted when the normalizer is ``"none"``.
        """
        if not in_place:
            x = x.clone()

        index = self._output_indices(
            self.postprocessed_names,
            x,
            name_to_index,
            self.index_training_output,
            self.index_inference_output,
        )
        self._check_statistics(statistics, x.shape[-1])
        remap = getattr(self, "statistics_remap", None) or {}
        if remap and statistics is not None and NORMALIZATION_STATISTICS[self.normalizer]:
            if name_to_index is None:
                msg = f"{self.__class__.__name__}: the normaliser remap {remap} needs the view's name_to_index."
                raise ValueError(msg)
            statistics = remapped_statistics(statistics, name_to_index, remap, self.__class__.__name__)

        # The physical thresholds are read from the config (``methods``), not from
        # ``postprocessorfunctions``: checkpoints pickled before thresholds were normalized
        # per call hold already-normalized CustomRelu objects there.
        for name, idx_dst in zip(self.postprocessed_names, index):
            if idx_dst is None:
                continue
            threshold = float(self.methods.get(name, self.default))
            threshold = float(self._normalized_threshold(threshold, statistics, idx_dst))
            x[..., idx_dst] = CustomRelu(threshold)(x[..., idx_dst])
        return x


class ConditionalPostprocessor(Postprocessor):
    """Base class for postprocessors that conditionally apply a transformation based on another variable.

    This class is intended to be subclassed for specific implementations.
    It expects the config to have keys corresponding to customizable values and lists of variables to postprocess.
    """

    def _prepare_postprocessing_indices_list(self):
        """Prepare the postprocessor indices list."""

        super()._prepare_postprocessing_indices_list()

        # retrieve index of masking variable
        self.masking_variable = self.remap
        self.masking_variable_training_output = self.data_indices.data.output.name_to_index.get(
            self.masking_variable, None
        )
        self.masking_variable_inference_output = self.data_indices.model.output.name_to_index.get(
            self.masking_variable, None
        )

    def fill_with_value(self, x: torch.Tensor, index: list[int], fill_mask: torch.tensor):
        for idx_dst, value in zip(index, self.postprocessorfunctions):
            if idx_dst is not None:
                x[..., idx_dst][fill_mask] = value
        return x

    @abstractmethod
    def get_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get a mask from data for conditional postprocessing.
        This method must be implemented by subclasses.

        Parameters:
            x (torch.Tensor): The output for reference variable.

        Returns:
            torch.Tensor: A mask tensor indicating the locations for postprocessing of shape x.shape.
        """
        pass

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        name_to_index: Optional[dict[str, int]] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Set values in the output tensor."""
        if not in_place:
            x = x.clone()

        index = self._output_indices(
            self.postprocessed_names,
            x,
            name_to_index,
            self.index_training_output,
            self.index_inference_output,
        )
        (masking_variable,) = self._output_indices(
            [self.masking_variable],
            x,
            name_to_index,
            [self.masking_variable_training_output],
            [self.masking_variable_inference_output],
        )
        if masking_variable is None:
            msg = f"{self.__class__.__name__}: the masking variable {self.masking_variable!r} is not in the output."
            raise ValueError(msg)

        # Replace with value if masking variable is zero
        postprocessor_mask = self.get_locations(x[..., masking_variable])

        # Replace values
        return self.fill_with_value(x, index, postprocessor_mask)


class ConditionalZeroPostprocessor(ConditionalPostprocessor):
    """Sets values to specified value where another variable is zero.

    Expects the config to have keys corresponding to customizable values and
    lists of variables to postprocess and a masking/reference variable to use for postprocessing.:

    ```
    default: "none"
    remap: "x"
    0:
        - y
    5.0:
        - x
    3.14:
        - q
    ```

    If "x" is zero, "y" will be postprocessed with 0, "x" with 5.0 and "q" with 3.14.
    """

    def _get_postprocessor_function(self, method: float, name: str):
        """For ConditionalZeroPostprocessor, the 'method' is the constant value to fill
        when the masking variable is zero. This function simply returns the value.
        """
        LOGGER.info(
            f"ConditionalZeroPostprocessor: replacing valus in {name} with value {method} if {self.masking_variable} is zero."
        )
        return method

    def get_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get zero mask from data"""
        # reference/masking variable is already selected. Mask covers all remaining dimensions.
        return x == 0


class ConditionalNaNPostprocessor(ConditionalPostprocessor):
    """Sets values to NaNs where another variable is NaN.

    Expects the config to have list of variables to postprocess and a
    masking/reference variable to use for postprocessing.:

    ```
    default: "none"
    remap: "x"
    nan:
        - y
    ```

    The module sets "y" NaN, at NaN locations of "x".
    """

    def _get_postprocessor_function(self, method: float, name: str):
        """For ConditionalNaNPostprocessor, the 'method' is a NaN to fill
        when the masking variable is NaN. This function simply returns a NaN.
        """
        LOGGER.info(
            f"ConditionalNaNPostprocessor: replacing values in {name} with value NaN if {self.masking_variable} is NaN."
        )
        return torch.nan

    def get_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get NaN mask from data"""
        # reference/masking variable is already selected. Mask covers all remaining dimensions.
        return torch.isnan(x)
