# (C) Copyright 2024 Anemoi contributors.
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
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class BaseImputer(BasePreprocessor, ABC):
    """Base class for Imputers.

    If not stated otherwise, the imputers assume that in the training dataset the NaN locations
    for each variables are fixed.
    The intended behaviour is the following:

    In training and validation (self.inference_mode = False):

    - NaN locations must be fixed. In the first forward pass the NaN locations are retrieved from the batch
    - transform (preprocessor): NaN locations do not need to be recalculated as their position is fixed
    - inverse_transform (postprocessor): at the fixed NaN locations the NaNs are put back into place

    In inference (self.inference_mode = True):

    - We cannot ensure that NaN locations are fixed. Missing only one NaN in the input data could cause the model prediction to fail which should be avoided. Therefore, the NaN locations are recalculated in every forward pass.
    - transform (preprocessor): We recalculate the NaN locations in every forward pass. The cached NaN locations must not be overwritten!
    - inverse_transform (postprocessor): Same behavious as in training/validation. Cached NaN locations from training are used to put NaNs into place.
    """

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

        self.nan_locations = None
        # weight imputed values with zero in loss calculation
        self.loss_mask_training = None

    def _validate_indices(self):
        assert len(self.index_training_input) == len(self.index_inference_input) <= len(self.replacement), (
            f"Error creating imputation indices {len(self.index_training_input)}, "
            f"{len(self.index_inference_input)}, {len(self.replacement)}"
        )
        assert len(self.index_training_output) == len(self.index_inference_output) <= len(self.replacement), (
            f"Error creating imputation indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.replacement)}"
        )

    def _create_imputation_indices(
        self,
        statistics=None,
    ):
        """Create the indices for imputation."""
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_input,
            self.index_inference_input,
            self.index_training_output,
            self.index_inference_output,
            self.replacement,
        ) = ([], [], [], [], [])

        # Create indices for imputation
        for name in name_to_index_training_input:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"Imputer: skipping {name} as no imputation method is specified")
                continue

            self.index_training_input.append(name_to_index_training_input[name])
            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_input.append(name_to_index_inference_input.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            if statistics is None:
                self.replacement.append(method)
            elif isinstance(statistics, dict):
                assert method in statistics, f"{method} is not a method in the statistics metadata"
                self.replacement.append(statistics[method][name_to_index_training_input[name]])
            else:
                raise TypeError(f"Statistics {type(statistics)} is optional and not a dictionary")

            LOGGER.debug(f"Imputer: replacing NaNs in {name} with value {self.replacement[-1]}")

    def _expand_subset_mask(self, x: torch.Tensor, idx_src: int, nan_locations: torch.Tensor) -> torch.Tensor:
        """Expand the subset of the mask to the correct shape."""
        return nan_locations[:, idx_src].expand(*x.shape[:-2], -1)

    def get_nans(self, x: torch.Tensor) -> torch.Tensor:
        """get NaN mask from data"""
        # The mask is only saved for the last two dimensions (grid, variable)
        idx = [slice(0, 1)] * (x.ndim - 2) + [slice(None), slice(None)]
        return torch.isnan(x[idx].squeeze())

    def get_cached_or_recomputed_nan_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get the cached or recomputed NaN locations.

        If in inference mode, recompute the NaN locations, else, use cached one.
        """
        # Reset the NaN locations for preprocesor in inference mode.
        if self.inference_mode:
            LOGGER.debug("Imputer: recalculating NaN locations for inference mode.")
            # get current NaN locations
            return self.get_nans(x)
        # if not in inference: work with reference to cached nan_locations
        return self.nan_locations

    def fill_with_value(self, x, index, nan_locations: torch.Tensor):
        for idx_src, (idx_dst, value) in zip(self.index_training_input, zip(index, self.replacement)):
            if idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src, nan_locations)] = value
        return x

    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()

        # Cache NaN locations for training and postprocessing and set loss mask in first forward pass
        if self.nan_locations is None:
            self.nan_locations = self.get_nans(x)

            # Initialize training loss mask to weigh imputed values with zeroes once
            self.loss_mask_training = torch.ones(
                (x.shape[-2], len(self.data_indices.model.output.name_to_index)), device=x.device
            )  # shape (grid, n_outputs)
            # for all variables that are imputed and part of the model output, set the loss weight to zero at NaN location
            for idx_src, idx_dst in zip(self.index_training_input, self.index_inference_output):
                if idx_dst is not None:
                    self.loss_mask_training[:, idx_dst] = (~self.nan_locations[:, idx_src]).int()

        # Ensure the user is running inference from a version of anemoi-inference that sets the inference_mode attribute
        assert hasattr(
            self, "inference_mode"
        ), "Inference mode is not available. This could mean you are using an outdated anemoi-inference."

        # work with cached or recomputed nan_nanlocations depending on inference mode
        nan_locations = self.get_cached_or_recomputed_nan_locations(x)

        # Choose correct index based on number of variables
        if x.shape[-1] == self.num_training_input_vars:
            index = self.index_training_input
        elif x.shape[-1] == self.num_inference_input_vars:
            index = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        # Replace values
        return self.fill_with_value(x, index, nan_locations)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor using the cached nan_locations."""
        if not in_place:
            x = x.clone()

        # Replace original nans with nan again
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
        for idx_src, idx_dst in zip(self.index_training_input, index):
            if idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src, self.nan_locations)] = torch.nan
        return x


class InputImputer(BaseImputer):
    """Imputes missing values using the statistics supplied.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    mean:
        - y
    maximum:
        - x
    minimum:
        - q
    ```
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        self._create_imputation_indices(statistics)

        self._validate_indices()


class ConstantImputer(BaseImputer):
    """Imputes missing values using the constant value.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    1:
        - y
    5.0:
        - x
    3.14:
        - q
    ```
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


class CopyImputer(BaseImputer):
    """Imputes missing values copying them from another variable.
    ```
    default: "none"
    variable_to_copy:
        - variable_missing_1
        - variable_missing_2
    ```
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
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_input,
            self.index_inference_input,
            self.index_training_output,
            self.index_inference_output,
            self.replacement,
        ) = ([], [], [], [], [])

        # Create indices for imputation
        for name in name_to_index_training_input:
            key_to_copy = self.methods.get(name, self.default)

            if key_to_copy == "none":
                LOGGER.debug(f"Imputer: skipping {name} as no imputation method is specified")
                continue

            self.index_training_input.append(name_to_index_training_input[name])
            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_input.append(name_to_index_inference_input.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            self.replacement.append(key_to_copy)

            LOGGER.debug(f"Imputer: replacing NaNs in {name} with value coming from variable :{self.replacement[-1]}")

    def fill_with_value(self, x, index, nan_locations: torch.Tensor):
        # Replace values
        for idx_src, (idx_dst, value) in zip(self.index_training_input, zip(index, self.replacement)):
            if idx_dst is not None:
                assert not torch.isnan(
                    x[..., self.data_indices.data.input.name_to_index[value]][
                        self._expand_subset_mask(x, idx_src, nan_locations)
                    ]
                ).any(), f"NaNs found in {value}."
                x[..., idx_dst][self._expand_subset_mask(x, idx_src, nan_locations)] = x[
                    ..., self.data_indices.data.input.name_to_index[value]
                ][self._expand_subset_mask(x, idx_src, nan_locations)]
        return x

    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()

        # Cache NaN locations for training and postprocessing and set loss mask in first forward pass
        if self.nan_locations is None:
            self.nan_locations = self.get_nans(x)

            # Initialize training loss mask to weigh imputed values with zeroes once
            self.loss_mask_training = torch.ones(
                (x.shape[-2], len(self.data_indices.model.output.name_to_index)), device=x.device
            )  # shape (grid, n_outputs)
            # for all variables that are imputed and part of the model output, set the loss weight to zero at NaN location
            for idx_src, idx_dst in zip(self.index_training_input, self.index_inference_output):
                if idx_dst is not None:
                    self.loss_mask_training[:, idx_dst] = (~self.nan_locations[:, idx_src]).int()

        # Ensure the user is running inference from a version of anemoi-inference that sets the inference_mode attribute
        assert hasattr(
            self, "inference_mode"
        ), "Inference mode is not available. This could mean you are using an outdated anemoi-inference."

        # work with cached or recomputed nan_nanlocations depending on inference mode
        nan_locations = self.get_cached_or_recomputed_nan_locations(x)

        # Choose correct index based on number of variables
        if x.shape[-1] == self.num_training_input_vars:
            index = self.index_training_input
        elif x.shape[-1] == self.num_inference_input_vars:
            index = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        return self.fill_with_value(x, index, nan_locations)


class DynamicMixin:
    """
    Mixin to add dynamic imputation behavior.
    To be used when NaN maps change at different timesteps.
    """

    def get_nans(self, x: torch.Tensor) -> torch.Tensor:
        """Override to calculate NaN locations dynamically."""
        return torch.isnan(x)

    def fill_with_value(self, x, index, nan_locations):
        # Replace values
        for idx, value in zip(index, self.replacement):
            if idx is not None:
                x[..., idx][nan_locations[..., idx]] = value
        return x

    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()

        # Initilialize mask every time
        nan_locations = self.get_nans(x)

        self.loss_mask_training = torch.ones(
            (x.shape[-2], len(self.data_indices.model.output.name_to_index)), device=x.device
        )

        # Choose correct index based on number of variables
        if x.shape[-1] == self.num_training_input_vars:
            index = self.index_training_input
        elif x.shape[-1] == self.num_inference_input_vars:
            index = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        return self.fill_with_value(x, index, nan_locations)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        return x


class DynamicInputImputer(DynamicMixin, InputImputer):
    "Imputes missing values using the statistics supplied and a dynamic NaN map."

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        InputImputer.__init__(self, config, data_indices, statistics)
        warnings.warn(
            "You are using a dynamic Imputer: NaN values will not be present in the model predictions. \
                      The model will be trained to predict imputed values. This might deteriorate performances."
        )


class DynamicConstantImputer(DynamicMixin, ConstantImputer):
    "Imputes missing values using the constant value and a dynamic NaN map."

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        ConstantImputer.__init__(self, config, data_indices, statistics)
        warnings.warn(
            "You are using a dynamic Imputer: NaN values will not be present in the model predictions. \
                      The model will be trained to predict imputed values. This might deteriorate performances."
        )


class DynamicCopyImputer(DynamicMixin, CopyImputer):
    """Dynamic Copy imputation behavior."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        CopyImputer.__init__(self, config, data_indices, statistics)
        warnings.warn(
            "You are using a dynamic Imputer: NaN values will not be present in the model predictions. \
                      The model will be trained to predict imputed values. This might deteriorate performances."
        )

    def fill_with_value(self, x, index, nan_locations):

        if x.shape[-1] == self.num_training_input_vars:
            indices = self.data_indices.data.input.name_to_index
        elif x.shape[-1] == self.num_inference_input_vars:
            indices = self.data_indices.model.input.name_to_index
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        # Replace values
        for idx, value in zip(index, self.replacement):
            if idx is not None:
                assert not torch.isnan(x[..., indices[value]][nan_locations[..., idx]]).any(), f"NaNs found in {value}."
                x[..., idx][nan_locations[..., idx]] = x[..., indices[value]][nan_locations[..., idx]]
        return x

    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        return DynamicMixin.transform(self, x, in_place)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        return DynamicMixin.inverse_transform(self, x, in_place)
