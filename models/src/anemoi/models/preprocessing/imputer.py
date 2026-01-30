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
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class BaseImputer(BasePreprocessor, ABC):
    """Base class for Imputers."""

    supports_skip_imputation = True

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

        self.register_buffer("nan_locations", torch.empty(0, dtype=torch.bool), persistent=False)
        # weight imputed values with zero in loss calculation
        self.register_buffer("loss_mask_training", torch.empty(0, dtype=torch.bool), persistent=False)

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

            if name_to_index_inference_input.get(name, None) is None:
                # if the variable is not in inference input (diagnostic variable), we cannot place NaNs in its inference output
                if method != self.default:
                    LOGGER.warning(
                        f"If placement of NaNs for diagnostic variables in inference output is desired, this needs to be handled by postprocessors: {name}"
                    )

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

            LOGGER.info(f"Imputer: replacing NaNs in {name} with value {self.replacement[-1]}")

    def get_nans(self, x: torch.Tensor) -> torch.Tensor:
        """Get NaN mask from data

        The mask is only saved for the first two dimensions (batch, timestep) and the last two dimensions (grid, variable)
        For the rest of the dimensions we select the first element since we assume the nan locations do not change along these dimensions.
        This means for the ensemble dimension: we assume that the NaN locations are the same for all ensemble members.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, time, ..., grid, variable)

        Returns
        -------
        torch.Tensor
            Tensor with NaN locations of shape (batch, time, ..., grid)
        """
        idx = [slice(None), slice(None)] + [0] * (x.ndim - 4) + [slice(None), slice(None)]
        return torch.isnan(x[idx])

    def _expand_subset_mask(self, x: torch.Tensor, idx_src: int, nan_locations: torch.Tensor) -> torch.Tensor:
        """Expand the subset of the nan location mask to the correct shape.

        The mask is only saved for the first dimension (batch) and the last two dimensions (grid, variable).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, time, ..., grid, variable)
        idx_src : int
            Index of the source variable in the nan locations mask
        nan_locations : torch.Tensor
            Tensor with NaN locations of shape (batch, grid, variable)

        Returns
        -------
        torch.Tensor
            Expanded tensor with NaN locations of shape (batch, time, ..., grid)
        """
        for i in x.shape[1:-2]:
            nan_locations = nan_locations.unsqueeze(1)

        return nan_locations[..., idx_src].expand(-1, *x.shape[1:-2], -1)

    def _clone_if_needed(self, x: torch.Tensor, in_place: bool) -> torch.Tensor:
        return x if in_place else x.clone()

    def _assert_nan_locations_batch(self, x: torch.Tensor) -> None:
        assert x.shape[0] == self.nan_locations.shape[0], (
            f"Batch dimension of input tensor ({x.shape[0]}) does not match the batch dimension of "
            f"nan locations ({self.nan_locations.shape[0]}). Are you using the postprocessors without running "
            "the preprocessor first?"
        )

    def _apply_nan_locations(
        self,
        x: torch.Tensor,
        src_index: list[int | None],
        dst_index: list[int | None] | dict[int, int],
    ) -> None:
        if isinstance(dst_index, dict):
            for idx_src in src_index:
                if idx_src is not None:
                    idx_dst = dst_index.get(idx_src)
                    if idx_dst is not None:
                        x[..., idx_dst][self._expand_subset_mask(x, idx_src, self.nan_locations)] = torch.nan
        else:
            for idx_src, idx_dst in zip(src_index, dst_index):
                if idx_src is not None and idx_dst is not None:
                    x[..., idx_dst][self._expand_subset_mask(x, idx_src, self.nan_locations)] = torch.nan

    def _store_training_nan_locations(self, nan_locations: torch.Tensor) -> None:
        """Persist NaN locations for training input (first timestep, full input space)."""
        subset = nan_locations[:, 0, ..., self.data_indices.data.input.full]
        if (
            len(self.nan_locations.shape) > 1
            and self.nan_locations.shape[0] == subset.shape[0]
            and self.nan_locations.shape[1] == subset.shape[1]
        ):
            self.nan_locations[:] = subset
        else:
            self.nan_locations = subset

    def _store_inference_nan_locations(self, nan_locations: torch.Tensor) -> None:
        """Persist NaN locations for inference input (first timestep)."""
        self.nan_locations = nan_locations[:, 0]

    def _reset_training_loss_mask(self, x: torch.Tensor) -> None:
        self.loss_mask_training = torch.ones(
            (x.shape[0], x.shape[-2], len(self.data_indices.model.output.name_to_index)), device=x.device
        )

    def _mask_training_loss(self, nan_locations: torch.Tensor) -> None:
        for idx_src, idx_dst in zip(self.index_training_input, self.index_inference_output):
            if idx_src is not None and idx_dst is not None:
                self.loss_mask_training[..., idx_dst] = (~nan_locations[:, 0, ..., idx_src]).int()

    def _transform_indices_for_input(self, x: torch.Tensor, nan_locations: torch.Tensor) -> list[int | None]:
        if x.shape[-1] == self.num_training_input_vars:
            self._store_training_nan_locations(nan_locations)
            self._reset_training_loss_mask(x)
            self._mask_training_loss(nan_locations)
            return self.index_training_input
        if x.shape[-1] == self.num_inference_input_vars:
            self._store_inference_nan_locations(nan_locations)
            return self.index_inference_input
        raise ValueError(
            f"Input tensor ({x.shape[-1]}) does not match the training "
            f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
        )

    def _inverse_indices_for_shape(self, x: torch.Tensor) -> tuple[list[int | None], list[int | None]] | None:
        if x.shape[-1] == self.num_training_output_vars:
            return self.index_inference_input, self.index_training_output
        if x.shape[-1] == self.num_inference_output_vars:
            return self.index_inference_input, self.index_inference_output
        if x.shape[-1] == self.num_training_input_vars:
            return self.index_training_input, self.index_training_input
        if x.shape[-1] == self.num_inference_input_vars:
            return self.index_inference_input, self.index_inference_input
        return None

    def _as_index_list(self, data_index: object) -> list[int]:
        return data_index.tolist() if torch.is_tensor(data_index) else list(data_index)

    def _is_subset(self, subset: list[int], full: list[int]) -> bool:
        return set(subset).issubset(set(full))

    def _subset_inverse_indices(
        self,
        data_index_list: list[int],
    ) -> tuple[list[int | None], dict[int, int]] | None:
        train_input_full = self.data_indices.data.input.full.tolist()
        if self._is_subset(data_index_list, train_input_full):
            return self._subset_inverse_indices_for_full(
                data_index_list,
                train_input_full,
                self.index_training_input,
            )

        infer_input_full = self.data_indices.model.input.full.tolist()
        if self._is_subset(data_index_list, infer_input_full):
            return self._subset_inverse_indices_for_full(
                data_index_list,
                infer_input_full,
                self.index_inference_input,
            )
        return None

    def _subset_inverse_indices_for_full(
        self,
        data_index_list: list[int],
        full_index: list[int],
        src_index: list[int | None],
    ) -> tuple[list[int | None], dict[int, int]]:
        full_pos = {full_idx: pos for pos, full_idx in enumerate(full_index)}
        dst_index_map = {full_pos[idx]: pos for pos, idx in enumerate(data_index_list)}
        src_index = [full_pos[idx] if idx in full_pos else None for idx in src_index]
        return src_index, dst_index_map

    def _apply_subset_inverse(self, x: torch.Tensor, data_index: object | None) -> bool:
        if data_index is None or self._inverse_indices_for_shape(x) is not None:
            return False

        data_index_list = self._as_index_list(data_index)
        diag_output = self.data_indices.data.output.diagnostic.tolist()
        if self._is_subset(data_index_list, diag_output):
            return True

        indices = self._subset_inverse_indices(data_index_list)
        if indices is None:
            return False

        src_index, dst_index_map = indices
        self._assert_nan_locations_batch(x)
        self._apply_nan_locations(x, src_index, dst_index_map)
        return True

    def fill_with_value(
        self, x: torch.Tensor, index_x: list[int], nan_locations: torch.Tensor, index_nl: list[int]
    ) -> torch.Tensor:
        """Fill NaN locations in the input tensor with the specified values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        index : list
            List of indices for the variables to be imputed
        nan_locations : torch.Tensor
            Tensor with NaN locations

        Returns
        -------
        torch.Tensor
            Tensor where NaN locations are filled with the specified values
        """
        # Expand the nan locations to match the shape of the input tensor
        for i in x.shape[2:-2]:
            nan_locations = nan_locations.unsqueeze(2)
        for idx_src, (idx_dst, value) in zip(index_nl, zip(index_x, self.replacement)):
            if idx_src is not None and idx_dst is not None:
                x[..., idx_dst][nan_locations[..., idx_src]] = value
        return x

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        in_place : bool
            Whether to process the tensor in place.
        skip_imputation : bool, optional
            When True, do not replace NaNs or update the stored NaN mask.
        """
        x = self._clone_if_needed(x, in_place)

        if skip_imputation:
            # Subset tensors (e.g., prognostic-only) are not safe to impute here because
            # imputation indices and masks are defined in full-input space. This path is
            # used during diffusion tendency computation where subset tensors are expected.
            return x

        # recalculate NaN locations every forward pass and save for backward pass
        nan_locations = self.get_nans(x)

        # choose correct index based on number of variables which are different for training and inference
        index = self._transform_indices_for_input(x, nan_locations)

        # Replace values
        return self.fill_with_value(x, index, nan_locations, index)

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        in_place : bool
            Whether to process the tensor in place.
        skip_imputation : bool, optional
            When True, do not re-insert NaNs or use the stored NaN mask.
        """
        x = self._clone_if_needed(x, in_place)

        if skip_imputation:
            return x

        if self._apply_subset_inverse(x, _kwargs.get("data_index")):
            return x

        indices = self._inverse_indices_for_shape(x)
        if indices is None:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

        src_index, dst_index = indices
        self._assert_nan_locations_batch(x)
        self._apply_nan_locations(x, src_index, dst_index)
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

        if isinstance(statistics, DictConfig):
            statistics = OmegaConf.to_container(statistics, resolve=True)
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

    def fill_with_value(
        self, x: torch.Tensor, index_x: list[int], nan_locations: torch.Tensor, index_nl: list[int]
    ) -> torch.Tensor:
        for i in x.shape[2:-2]:
            nan_locations = nan_locations.unsqueeze(2)
        # Replace values
        for idx_src, (idx_dst, value) in zip(index_nl, zip(index_x, self.replacement)):
            if idx_dst is not None:
                assert not torch.isnan(
                    x[..., self.data_indices.data.input.name_to_index[value]][nan_locations[..., idx_src]]
                ).any(), f"NaNs found in variable {value} to be copied."
                x[..., idx_dst][nan_locations[..., idx_src]] = x[
                    ..., self.data_indices.data.input.name_to_index[value]
                ][nan_locations[..., idx_src]]
        return x


class DynamicMixin:
    """Mixin to add dynamic imputation behavior.
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

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()

        if skip_imputation:
            return x

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

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **_kwargs) -> torch.Tensor:
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

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        return DynamicMixin.transform(self, x, in_place, skip_imputation=skip_imputation)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **_kwargs) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        return DynamicMixin.inverse_transform(self, x, in_place)
