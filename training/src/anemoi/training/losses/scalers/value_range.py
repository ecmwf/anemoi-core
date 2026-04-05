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
    """Apply variable-specific loss weights based on target-value ranges.

    This scaler is designed for cases such as radar reflectivity (`refc`) where
    large values are comparatively rare and should contribute more strongly to
    the loss. It updates per batch using the target portion of the current
    batch, broadcasting weights across the ensemble dimension.

    Notes
    -----
    - The thresholds are interpreted in the *raw* variable units.
    - The current implementation assumes the selected variable is normalised
      with either `mean-std`, `std` or `none`.
    """

    scale_dims = (TensorDim.BATCH_SIZE, TensorDim.TIME, TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(
        self,
        variable: str,
        thresholds: list[float],
        weights: list[float],
        data_indices: IndexCollection,
        statistics: dict,
        normalization: str = "mean-std",
        norm: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(norm=norm)
        del kwargs

        if len(weights) != len(thresholds) + 1:
            msg = "weights must have exactly len(thresholds) + 1 entries."
            raise ValueError(msg)

        self.variable = variable
        self.thresholds = [float(x) for x in thresholds]
        self.weights = [float(x) for x in weights]
        self.data_indices = data_indices
        self.statistics = statistics
        self.normalization = normalization

        self.full_var_idx = int(self.data_indices.data.input.name_to_index[variable])
        self.output_var_idx = int(self.data_indices.model.output.name_to_index[variable])

    def _to_raw_units(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalised batch values back to raw units for thresholding."""
        if self.normalization == "none":
            return values

        mean = float(self.statistics["mean"][self.full_var_idx])
        stdev = float(self.statistics["stdev"][self.full_var_idx])

        if self.normalization == "mean-std":
            return values * stdev + mean
        if self.normalization == "std":
            return values * stdev

        msg = f"Unsupported normalization mode for {self.__class__.__name__}: {self.normalization}"
        raise ValueError(msg)

    def _bucketize_weights(self, raw_target: torch.Tensor) -> torch.Tensor:
        """Map raw target values to the configured piecewise-constant weights."""
        out = torch.full_like(raw_target, self.weights[0], dtype=torch.float32)
        for threshold, weight in zip(self.thresholds, self.weights[1:], strict=True):
            out = torch.where(raw_target >= threshold, torch.tensor(weight, device=out.device, dtype=out.dtype), out)
        return out

    def on_batch_start(
        self,
        model: AnemoiModelInterface,
        dataset_name: str | None = None,
        batch: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del model
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
        n_step_output = int(getattr(model, "n_step_output"))

        target_slice = dataset_batch[:, n_step_input : n_step_input + n_step_output, 0, :, self.full_var_idx]
        raw_target = self._to_raw_units(target_slice)
        variable_weights = self._bucketize_weights(raw_target)

        bsz, tlen, grid = variable_weights.shape
        nvars = len(self.data_indices.model.output.full)
        weights = torch.ones((bsz, tlen, grid, nvars), device=variable_weights.device, dtype=torch.float32)
        weights[..., self.output_var_idx] = variable_weights
        return weights
