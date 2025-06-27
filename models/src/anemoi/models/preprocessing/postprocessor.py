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

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.layers.activations import CustomRelu

LOGGER = logging.getLogger(__name__)


class Postprocessor(BasePreprocessor):
    """Base class for Imputers."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the imputer.

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

        self._create_imputation_indices()

        self._validate_indices()

    def _validate_indices(self):
        assert (
            len(self.index_training_output) == len(self.index_inference_output) <= len(self.postprocessorfunctions)
        ), (
            f"Error creating imputation indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.postprocessorfunctions)}"
        )

    def _create_imputation_indices(
        self,
    ):
        """Create the indices for imputation."""
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_output,
            self.index_inference_output,
            self.postprocessorfunctions,
        ) = ([], [], [])

        # Create indices for imputation
        for name in name_to_index_training_output:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"Postprocessor: skipping {name} as no postprocessing method is specified")
                continue

            assert name in name_to_index_inference_output, (
                f"Postprocessor: {name} not found in inference output indices. "
                f"Postprocessors cannot be applied to foorcing variables."
            )

            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            if method == "relu":
                self.postprocessorfunctions.append(torch.nn.functional.relu)
            elif method == "hardtanh":
                self.postprocessorfunctions.append(torch.nn.functional.hardtanh)

            LOGGER.info(f"Postprocessor: applying {method} to {name}")

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
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

    Expects the config to have keys corresponding to customizable thresholds as lists of variables to impute.:
    ```
    nornmalizer: 'mean-std'
    1:
        - y
    0:
        - x
    3.14:
        - q
    ```
    Thresholds are in un-normalized space. If normalizer is specified, the threshold values are not normalized.
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
        normalizer: str = "none",
    ) -> None:

        self.normalizer = normalizer
        self.statistics = statistics

        # Validate normalizer input
        if self.normalizer not in {"none", "mean-std", "min-max", "max", "std"}:
            raise ValueError(
                "Normalizer must be one of: 'none', 'mean-std', 'min-max', 'max', 'std' in NormalizedReluBounding."
            )

        super().__init__(config, data_indices, statistics)

    def _create_imputation_indices(
        self,
    ):
        """Create the indices for imputation."""
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_output,
            self.index_inference_output,
            self.postprocessorfunctions,
        ) = ([], [], [])

        # Create indices for imputation
        for name in name_to_index_training_output:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"CustomReluPostprocessor: skipping {name} as no postprocessing method is specified")
                continue

            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

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

            self.postprocessorfunctions.append(CustomRelu(normalized_value))

            LOGGER.info(
                f"NormalizedReluPostprocessor: applying NormalizedRelu with threshold {normalized_value} after {self.normalizer} normalization to {name}."
            )


class ConditionalZeroPostprocessor(Postprocessor):
    """Sets values to specified value where another variable is zero.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
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

    If "x" is zero, "y" will be imputed with 0, "x" with 5.0 and "q" with 3.14.
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        self._create_imputation_indices()

        self._validate_indices()

    def _create_imputation_indices(
        self,
    ):
        """Create the indices for imputation."""
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        self.masking_variable = self.remap
        self.masking_variable_training_output = name_to_index_training_output.get(self.masking_variable, None)
        self.masking_variable_inference_output = name_to_index_inference_output.get(self.masking_variable, None)

        (
            self.index_training_output,
            self.index_inference_output,
            self.postprocessorfunctions,
        ) = ([], [], [])

        # Create indices for imputation
        for name in name_to_index_training_output:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"ConditionalZeroPostprocessor: skipping {name} as no method is specified.")
                continue

            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            self.postprocessorfunctions.append(method)

            LOGGER.info(
                f"ConditionalZeroPostprocessor: replacing valus in {name} with value {self.postprocessorfunctions[-1]} if {self.masking_variable} is zero."
            )

    def _expand_subset_mask(self, x: torch.Tensor, mask: torch.tensor) -> torch.Tensor:
        """Expand the subset of the mask to the correct shape."""
        return mask.expand(*x.shape[:-2], -1)

    def get_zeros(self, x: torch.Tensor) -> torch.Tensor:
        """get zero mask from data"""
        # The mask is only saved for the last dimension (grid)
        idx = [slice(0, 1)] * (x.ndim - 2) + [slice(None), slice(None)]
        return (x[idx] == 0).squeeze()

    def fill_with_value(self, x, index, fill_mask):
        for idx_dst, value in zip(index, self.postprocessorfunctions):
            if idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, fill_mask)] = value
        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Set values in the output tensor."""
        if not in_place:
            x = x.clone()

        # Replace original nans with nan again
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

        zero_mask = self.get_zeros(x[..., masking_variable])

        # Replace values
        return self.fill_with_value(x, index, zero_mask)
