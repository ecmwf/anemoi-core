# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

import torch

from anemoi.training.diagnostics.callbacks.plot_adapter import TemporalDownscalerPlotAdapter
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import as_timedelta

if TYPE_CHECKING:
    from torch import nn

    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


def _normalised_zero_and_scale(pre_processors: nn.Module | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the normaliser's ``(_norm_add, _norm_mul)`` buffers (full data space), if any.

    With ``x_norm = x * _norm_mul + _norm_add``, the normalised value of a physical zero is ``_norm_add``.
    """
    if pre_processors is None:
        return None
    for processor in getattr(pre_processors, "processors", {}).values():
        if hasattr(processor, "_norm_add") and hasattr(processor, "_norm_mul"):
            return processor._norm_add, processor._norm_mul
    return None


class TemporalDownscaler(BaseSingleStepTask):
    """Temporal downscaling task implementation.

    Optionally applies a night mask to accumulated radiation outputs (e.g. ``ssr``, ``ssrd``, ``fdir``):
    a prediction at output time ``t`` is set to zero where the (instantaneous) mask variable, typically
    ``insolation``, is zero both at ``t`` and at ``t - output_timestep``, i.e. the sun was down during the
    whole accumulation window. This assumes the accumulation period equals ``output_timestep``. For an
    output at ``t = 0`` (left boundary) the previous time is not in the batch and only ``t`` is used.

    The mask is computed from the batch, so it is applied in training/validation only.

    Example config::

        task:
          _target_: anemoi.training.tasks.TemporalDownscaler
          input_timestep: 6h
          output_timestep: 1h
          night_mask:
            mask_variable: insolation
            variables: [ssr, ssrd, fdir]
    """

    name: str = "temporal-downscaler"

    def __init__(
        self,
        input_timestep: str,
        output_timestep: str,
        output_left_boundary: bool = False,
        output_right_boundary: bool = False,
        night_mask: dict | None = None,
        **_kwargs,
    ) -> None:
        self.input_timestep = input_timestep
        input_timedelta = as_timedelta(input_timestep)
        output_timedelta = as_timedelta(output_timestep)

        input_offsets = [datetime.timedelta(hours=0), input_timedelta]

        assert input_timedelta % output_timedelta == datetime.timedelta(
            0,
        ), "Input timestep must be an integer multiple of output timestep for temporal downscaling."
        num_output_steps = input_timedelta // output_timedelta
        output_offsets = [output_timedelta * (i + 1) for i in range(num_output_steps - 1)]

        if output_left_boundary:
            output_offsets = [datetime.timedelta(hours=0), *output_offsets]

        if output_right_boundary:
            output_offsets = [*output_offsets, input_timedelta]

        super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)
        self._plot_adapter = TemporalDownscalerPlotAdapter(self)

        self.night_mask = None
        if night_mask:
            variables = list(night_mask.get("variables") or [])
            assert variables, "night_mask.variables must list at least one variable to mask."
            self.night_mask = {
                "mask_variable": night_mask.get("mask_variable", "insolation"),
                "variables": variables,
            }
            # For each output, the batch index of t and of t - output_timestep (None if not in the batch).
            offsets = self.get_offsets()
            self._night_mask_time_pairs = [
                (offsets.index(o), offsets.index(o - output_timedelta) if (o - output_timedelta) in offsets else None)
                for o in self._output_offsets
            ]

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return self.input_timestep

    def log_training_state(self) -> None:
        if self.night_mask is not None:
            LOGGER.info(
                "TemporalDownscaler: night mask on %s where %s == 0 at t and t-1 output step.",
                self.night_mask["variables"],
                self.night_mask["mask_variable"],
            )

    def mask_predictions(
        self,
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        data_indices: dict[str, IndexCollection],
        pre_processors: nn.ModuleDict | dict | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Zero the masked variables at night points.

        ``y_pred`` is in model-output space ``(bs, n_out, ens, grid, n_model_out)`` and ``batch`` in full data
        space ``(bs, time, ens, grid, n_data)``; both are normalised, so physical zero is mapped through the
        normaliser.
        """
        if self.night_mask is None:
            return y_pred

        mask_var = self.night_mask["mask_variable"]
        out = dict(y_pred)
        n_masked = 0
        for dataset_name, pred in y_pred.items():
            indices = data_indices[dataset_name]
            name_to_index = indices.name_to_index
            model_out = indices.model.output.name_to_index
            targets = [v for v in self.night_mask["variables"] if v in model_out]
            if mask_var not in name_to_index or not targets:
                continue  # dataset without the mask variable / masked outputs; checked across datasets below
            missing = [v for v in self.night_mask["variables"] if v not in model_out]
            if missing:
                msg = f"night_mask variables {missing} are not model outputs of dataset '{dataset_name}'."
                raise ValueError(msg)

            norm = _normalised_zero_and_scale(pre_processors[dataset_name] if pre_processors is not None else None)
            mask_idx = name_to_index[mask_var]

            # Physical mask variable at every batch time: (bs, time, ens, grid)
            mask_field = batch[dataset_name][..., mask_idx]
            if norm is not None:
                norm_add, norm_mul = norm
                mask_field = (mask_field - norm_add[mask_idx]) / norm_mul[mask_idx]
            dark = mask_field <= 0

            t_idx = [t for t, _ in self._night_mask_time_pairs]
            prev_idx = [p if p is not None else t for t, p in self._night_mask_time_pairs]
            night = dark[:, t_idx] & dark[:, prev_idx]  # (bs, n_out, ens_b, grid)

            pred_idx = torch.tensor([model_out[v] for v in targets], device=pred.device)
            if norm is not None:
                zero = norm[0][[name_to_index[v] for v in targets]].to(pred.dtype)
            else:
                zero = torch.zeros(len(targets), dtype=pred.dtype, device=pred.device)

            masked = torch.where(night.unsqueeze(-1), zero, pred[..., pred_idx])
            pred = pred.clone()
            pred[..., pred_idx] = masked
            out[dataset_name] = pred
            n_masked += 1

        if n_masked == 0:
            msg = (
                f"night_mask is configured (mask_variable='{mask_var}', variables={self.night_mask['variables']}) "
                "but no dataset contains the mask variable and any of these as model outputs. "
                "Check that the mask variable is not removed from the data config."
            )
            raise ValueError(msg)
        return out
