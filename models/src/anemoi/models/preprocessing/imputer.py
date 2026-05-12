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

    @staticmethod
    def _resolve_pos(pos: Optional[int], ndim: int) -> Optional[int]:
        """Resolve a possibly-negative axis position to a positive one for ``ndim``."""
        if pos is None:
            return None
        return pos if pos >= 0 else ndim + pos

    def get_nans(self, x: torch.Tensor, layout=None) -> torch.Tensor:
        """Get NaN mask from data.

        Collapses the ensemble dim (NaN locations are assumed to be the same
        across ensemble members). When ``layout`` is supplied, the ensemble
        axis is read from it; otherwise the legacy ``(batch, time, ..., grid,
        variables)`` contract is used (requires ``ndim >= 4``).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        layout : TensorLayout, optional
            Per-tensor axis layout. If provided, only the ensemble axis is
            collapsed (set to index 0); all other axes are kept.

        Returns
        -------
        torch.Tensor
            NaN mask. Shape equals ``x.shape`` with the ensemble axis
            removed (if present). When ``layout`` is ``None`` the legacy
            shape ``(batch, time, ..., grid, variable)`` is returned with
            interior dims collapsed to 0.
        """
        if layout is None:
            idx = [slice(None), slice(None)] + [0] * (x.ndim - 4) + [slice(None), slice(None)]
            return torch.isnan(x[tuple(idx)])
        ens_pos = self._resolve_pos(layout.ensemble, x.ndim)
        if ens_pos is None:
            return torch.isnan(x)
        idx = [slice(None)] * x.ndim
        idx[ens_pos] = 0
        return torch.isnan(x[tuple(idx)])

    def _drop_time_axis(self, mask: torch.Tensor, layout) -> torch.Tensor:
        """Select index 0 along the time axis of ``mask`` (collapsing it).

        ``mask`` is the output of :meth:`get_nans` (i.e. ``x`` with the
        ensemble axis removed). When ``layout.time`` is ``None`` (sparse
        observations) the mask is returned unchanged.
        """
        if layout is None:
            # Legacy: time is at axis 1 in the original tensor; ensemble
            # was already collapsed, so it remains at axis 1 in mask.
            return mask[:, 0]
        if layout.time is None:
            return mask
        # Resolve time pos in the original tensor, then adjust for the
        # already-removed ensemble axis.
        ndim_x = mask.ndim + (1 if layout.ensemble is not None else 0)
        time_pos = self._resolve_pos(layout.time, ndim_x)
        ens_pos = self._resolve_pos(layout.ensemble, ndim_x)
        if ens_pos is not None and ens_pos < time_pos:
            time_pos -= 1
        idx = [slice(None)] * mask.ndim
        idx[time_pos] = 0
        return mask[tuple(idx)]

    def _expand_subset_mask(
        self,
        x: torch.Tensor,
        idx_src: int,
        nan_locations: torch.Tensor,
        layout=None,
    ) -> torch.Tensor:
        """Expand the saved nan-location mask so it broadcasts with ``x[..., idx_dst]``.

        The saved ``self.nan_locations`` (passed in as ``nan_locations``) has
        shape matching ``x`` minus the variables, time and ensemble axes.
        We index it along the variables axis (giving ``mask``) then insert
        size-1 dims for time and ensemble so it broadcasts with
        ``x[..., idx_dst]`` (i.e. ``x`` minus the variables axis).
        """
        if layout is None:
            for _ in x.shape[1:-2]:
                nan_locations = nan_locations.unsqueeze(1)
            return nan_locations[..., idx_src].expand(-1, *x.shape[1:-2], -1)

        mask = nan_locations[..., idx_src]
        var_pos = self._resolve_pos(layout.variables, x.ndim)
        time_pos = self._resolve_pos(layout.time, x.ndim)
        ens_pos = self._resolve_pos(layout.ensemble, x.ndim)
        # Insert size-1 dims for axes that were collapsed/dropped.
        # In the variables-removed target, axes after var_pos shift down by 1.
        for src_pos in sorted(p for p in (time_pos, ens_pos) if p is not None):
            target_pos = src_pos - 1 if src_pos > var_pos else src_pos
            mask = mask.unsqueeze(target_pos)
        return mask

    def fill_with_value(
        self,
        x: torch.Tensor,
        index_x: list[int],
        nan_locations: torch.Tensor,
        index_nl: list[int],
        layout=None,
    ) -> torch.Tensor:
        """Fill NaN locations in ``x`` with the configured replacement values."""
        if layout is None:
            for _ in x.shape[2:-2]:
                nan_locations = nan_locations.unsqueeze(2)
        else:
            # nan_locations is x with ensemble removed; re-insert a size-1
            # ensemble axis so it broadcasts back to x[..., idx_dst].
            if layout.ensemble is not None:
                var_pos = self._resolve_pos(layout.variables, x.ndim)
                ens_pos = self._resolve_pos(layout.ensemble, x.ndim)
                target_pos = ens_pos - 1 if ens_pos > var_pos else ens_pos
                nan_locations = nan_locations.unsqueeze(target_pos)
        for idx_src, (idx_dst, value) in zip(index_nl, zip(index_x, self.replacement)):
            if idx_src is not None and idx_dst is not None:
                x[..., idx_dst][nan_locations[..., idx_src]] = value
        return x

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        layout=None,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()
        if skip_imputation:
            return x

        # recalculate NaN locations every forward pass and save for backward pass
        nan_locations = self.get_nans(x, layout=layout)

        # Drop the time axis (if present) for storage; the mask used for
        # buffer-reuse and the loss-mask is per-(batch, grid, variable).
        first_time_mask = self._drop_time_axis(nan_locations, layout)

        # choose correct index based on number of variables which are different for training and inference
        if x.shape[-1] == self.num_training_input_vars:
            # training input

            # save nan locations for input variables from training input,
            # selecting the first timestep (already done above) and the
            # full input variable index. Reuse the registered buffer when
            # the shape hasn't changed across forward passes.
            new_nan_locations = first_time_mask[..., self.data_indices.data.input.full]
            if self.nan_locations.shape == new_nan_locations.shape:
                self.nan_locations[:] = new_nan_locations
            else:
                self.nan_locations = new_nan_locations

            # data indices for training input
            index = self.index_training_input

            # set training loss mask: same spatial/batch shape as
            # first_time_mask, with the variable axis sized to n_outputs.
            # When the layout has no batch axis (per-sample sparse path),
            # prepend a singleton batch dim so the produced mask shape
            # matches ``NaNMaskScaler.scale_dims = (BATCH, GRID, VARIABLE)``.
            spatial_shape = list(first_time_mask.shape[:-1])
            if layout is not None and layout.batch is None:
                spatial_shape = [1] + spatial_shape
                first_time_mask_for_loss = first_time_mask.unsqueeze(0)
            else:
                first_time_mask_for_loss = first_time_mask
            loss_shape = spatial_shape + [
                len(self.data_indices.model.output.name_to_index)
            ]
            self.loss_mask_training = torch.ones(loss_shape, device=x.device)

            # for all variables that are imputed and part of the model output, set the loss weight to zero at NaN location
            for idx_src, idx_dst in zip(self.index_training_input, self.index_inference_output):
                if idx_src is not None and idx_dst is not None:
                    self.loss_mask_training[..., idx_dst] = (~first_time_mask_for_loss[..., idx_src]).int()

        elif x.shape[-1] == self.num_inference_input_vars:
            # inference input

            # save nan masks of inference input for inverse transform
            self.nan_locations = first_time_mask

            # data indices for training input
            index = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        # Replace values
        return self.fill_with_value(x, index, nan_locations, index, layout=layout)

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        layout=None,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()
        if skip_imputation:
            return x

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

        # Sanity-check the batch dim, but only if the layout has one.
        if layout is None:
            assert x.shape[0] == self.nan_locations.shape[0], (
                f"Batch dimension of input tensor ({x.shape[0]}) does not match the "
                f"batch dimension of nan locations ({self.nan_locations.shape[0]}). "
                "Are you using the postprocessors without running the preprocessor first?"
            )
        elif layout.batch is not None:
            batch_pos = self._resolve_pos(layout.batch, x.ndim)
            assert x.shape[batch_pos] == self.nan_locations.shape[0], (
                f"Batch dimension of input tensor ({x.shape[batch_pos]}) does not match the "
                f"batch dimension of nan locations ({self.nan_locations.shape[0]}). "
                "Are you using the postprocessors without running the preprocessor first?"
            )

        # Replace values
        for idx_src, idx_dst in zip(self.index_inference_input, index):
            if idx_src is not None and idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src, self.nan_locations, layout=layout)] = torch.nan
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
        self,
        x: torch.Tensor,
        index_x: list[int],
        nan_locations: torch.Tensor,
        index_nl: list[int],
        layout=None,
    ) -> torch.Tensor:
        if layout is None:
            for _ in x.shape[2:-2]:
                nan_locations = nan_locations.unsqueeze(2)
        else:
            if layout.ensemble is not None:
                var_pos = self._resolve_pos(layout.variables, x.ndim)
                ens_pos = self._resolve_pos(layout.ensemble, x.ndim)
                target_pos = ens_pos - 1 if ens_pos > var_pos else ens_pos
                nan_locations = nan_locations.unsqueeze(target_pos)
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
