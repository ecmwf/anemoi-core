# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class TargetValueRangeScaler(BaseUpdatingScaler):
    """Apply multiplicative loss factors based on target-value ranges.

    This scaler is designed for cases such as radar reflectivity (`refc`) where
    large values are comparatively rare and should contribute more strongly to
    the loss. It updates per batch using the target portion of the current
    batch, broadcasting multiplicative factors across the ensemble dimension.

    The produced tensor is intended to be combined with existing Anemoi loss
    scalers (for example `general_variable`) in the standard ScaleTensor stack.
    That means the effective weight is multiplicative:

        final_weight = existing_weight * range_weight_factor

    Notes
    -----
    - The thresholds are interpreted in the *raw* variable units.
    - The current implementation assumes the selected variable is normalised
      with either `mean-std`, `std`, `min-max` or `none`.
    """

    scale_dims = (TensorDim.BATCH_SIZE, TensorDim.TIME, TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(
        self,
        variable: str,
        thresholds: list[float],
        data_indices: IndexCollection,
        statistics: dict,
        range_weight_factors: list[float] | None = None,
        apply_to: str | list[str] = "self",
        normalization: str = "mean-std",
        norm: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(norm=norm)
        legacy_weights = kwargs.pop("weights", None)
        del kwargs

        if range_weight_factors is None:
            range_weight_factors = legacy_weights

        if range_weight_factors is None:
            msg = "range_weight_factors must be provided."
            raise ValueError(msg)

        if len(range_weight_factors) != len(thresholds) + 1:
            msg = "range_weight_factors must have exactly len(thresholds) + 1 entries."
            raise ValueError(msg)

        self.variable = variable
        self.thresholds = [float(x) for x in thresholds]
        self.range_weight_factors = [float(x) for x in range_weight_factors]
        self.data_indices = data_indices
        self.statistics = statistics
        self.normalization = normalization
        self.apply_to = apply_to

        self.full_var_idx = int(self.data_indices.data.input.name_to_index[variable])
        self.output_var_indices = self._resolve_output_indices(apply_to)
        self._logged_first_batch = False

        LOGGER.info(
            "%s configured for variable=%s normalization=%s thresholds=%s range_weight_factors=%s apply_to=%s",
            self.__class__.__name__,
            self.variable,
            self.normalization,
            self.thresholds,
            self.range_weight_factors,
            self.apply_to,
        )

    def _resolve_output_indices(self, apply_to: str | list[str]) -> list[int]:
        """Resolve which model-output variables should receive the range-based factors."""
        name_to_index = self.data_indices.model.output.name_to_index

        if apply_to == "self":
            return [int(name_to_index[self.variable])]

        if apply_to == "all":
            return [int(idx) for idx in name_to_index.values()]

        if not isinstance(apply_to, list) or len(apply_to) == 0:
            msg = "apply_to must be 'self', 'all', or a non-empty list of output variable names."
            raise ValueError(msg)

        missing = [name for name in apply_to if name not in name_to_index]
        if missing:
            msg = f"Variables in apply_to were not found in model outputs: {missing}"
            raise ValueError(msg)

        return [int(name_to_index[name]) for name in apply_to]

    def _to_raw_units(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalised batch values back to raw units for thresholding."""
        if self.normalization == "none":
            return values

        mean = float(self.statistics["mean"][self.full_var_idx])
        stdev = float(self.statistics["stdev"][self.full_var_idx])
        minimum = float(self.statistics["minimum"][self.full_var_idx])
        maximum = float(self.statistics["maximum"][self.full_var_idx])

        if self.normalization == "mean-std":
            return values * stdev + mean
        if self.normalization == "std":
            return values * stdev
        if self.normalization == "min-max":
            return values * (maximum - minimum) + minimum

        msg = f"Unsupported normalization mode for {self.__class__.__name__}: {self.normalization}"
        raise ValueError(msg)

    def _bucketize_weights(self, raw_target: torch.Tensor) -> torch.Tensor:
        """Map raw target values to the configured piecewise-constant multiplicative factors."""
        out = torch.full_like(raw_target, self.range_weight_factors[0], dtype=torch.float32)
        for threshold, weight in zip(self.thresholds, self.range_weight_factors[1:], strict=True):
            out = torch.where(raw_target >= threshold, torch.tensor(weight, device=out.device, dtype=out.dtype), out)
        return out

    def on_batch_start(
        self,
        model: AnemoiModelInterface,
        dataset_name: str | None = None,
        batch: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if batch is None:
            return None

        dataset_batch = batch
        if isinstance(batch, dict):
            assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
            dataset_batch = batch[dataset_name]

        if not torch.is_tensor(dataset_batch):
            return None
        if dataset_batch.ndim != 5:
            LOGGER.warning(
                "%s expected a 5D batch tensor (bs,time,ens,grid,var), got shape=%s",
                self.__class__.__name__,
                tuple(dataset_batch.shape),
            )
            return None

        n_step_input = int(getattr(model, "n_step_input"))
        n_step_output = int(getattr(model.config.training, "multistep_output", 1))

        target_slice = dataset_batch[:, n_step_input : n_step_input + n_step_output, 0, :, self.full_var_idx]
        raw_target = self._to_raw_units(target_slice)
        variable_weights = self._bucketize_weights(raw_target)

        bsz, tlen, grid = variable_weights.shape
        nvars = len(self.data_indices.model.output.full)
        weights = torch.ones((bsz, tlen, grid, nvars), device=variable_weights.device, dtype=torch.float32)
        weights[..., self.output_var_indices] = variable_weights.unsqueeze(-1)

        if not self._logged_first_batch:
            unique_weights = torch.unique(variable_weights.detach().cpu()).tolist()
            bucket_counts = {
                str(float(weight)): int((variable_weights == weight).sum().detach().cpu())
                for weight in self.range_weight_factors
            }
            total_count = int(variable_weights.numel())
            LOGGER.info(
                "%s applied for variable=%s dataset=%s raw_target_range=[%.3f, %.3f] unique_range_weights=%s bucket_counts=%s total_count=%d",
                self.__class__.__name__,
                self.variable,
                dataset_name,
                float(raw_target.min().detach().cpu()),
                float(raw_target.max().detach().cpu()),
                unique_weights,
                bucket_counts,
                total_count,
            )
            self._logged_first_batch = True

        return weights
