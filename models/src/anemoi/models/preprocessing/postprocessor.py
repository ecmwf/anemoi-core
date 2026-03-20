# (C) Copyright 2024 Anemoi contributors.
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

LOGGER = logging.getLogger(__name__)


class Postprocessor(BasePreprocessor):
    """Class for Basic Postprocessors.

    For Postprocessors just the inverse_transform method is implemented.
    transform is not needed and corresponds to the identity function.
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the Postprocessor.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)

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
            self.index_training_output,
            self.index_inference_output,
            self.postprocessorfunctions,
        ) = ([], [], [])

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

            self.index_training_output.append(self._get_index(self.data_indices.data.output.name_to_index, name))
            self.index_inference_output.append(self._get_index(self.data_indices.model.output.name_to_index, name))
            self.postprocessorfunctions.append(self._get_postprocessor_function(method, name))

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

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Postprocess model output tensor."""
        if not in_place:
            x = x.clone()

        if x.shape[-1] == self.num_training_output_vars:
            index = self.index_training_output
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.index_inference_output
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
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
    Thresholds are in un-normalized space. If normalizer is specified, the threshold values are normalized.
    This is necessary if in config file the normalizer is specified before the postprocessor, e.g.:
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

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:

        self.statistics = statistics

        super().__init__(config, data_indices, statistics)

        # Validate normalizer input
        if self.normalizer not in {"none", "mean-std", "min-max", "max", "std"}:
            raise ValueError(
                "Normalizer must be one of: 'none', 'mean-std', 'min-max', 'max', 'std' in NormalizedReluBounding."
            )

    def _get_postprocessor_function(self, method: float, name: str) -> CustomRelu:
        """Get the relu function class for the specified threshold and name."""
        stat_index = self.data_indices.data.input.name_to_index[name]
        normalized_value = method
        if self.normalizer == "mean-std":
            mean = self.statistics["mean"][stat_index]
            std = self.statistics["stdev"][stat_index]
            normalized_value = (method - mean) / std
        elif self.normalizer == "min-max":
            min_stat = self.statistics["minimum"][stat_index]
            max_stat = self.statistics["maximum"][stat_index]
            normalized_value = (method - min_stat) / (max_stat - min_stat)
        elif self.normalizer == "max":
            max_stat = self.statistics["maximum"][stat_index]
            normalized_value = method / max_stat
        elif self.normalizer == "std":
            std = self.statistics["stdev"][stat_index]
            normalized_value = method / std
        postprocessor_function = CustomRelu(normalized_value)

        LOGGER.info(
            f"NormalizedReluPostprocessor: applying NormalizedRelu with threshold {normalized_value} after {self.normalizer} normalization to {name}."
        )
        return postprocessor_function


class ConditionalPostprocessor(Postprocessor):
    """Base class for postprocessors that conditionally apply a transformation based on another variable.

    This class is intended to be subclassed for specific implementations.
    It expects the config to have keys corresponding to customizable values and lists of variables to postprocess.
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

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

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Set values in the output tensor."""
        if not in_place:
            x = x.clone()

        # Replace with value if masking variable is zero
        if x.shape[-1] == self.num_training_output_vars:
            index = self.index_training_output
            masking_variable = self.masking_variable_training_output
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.index_inference_output
            masking_variable = self.masking_variable_inference_output
        else:
            raise ValueError(
                f"Output tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

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
