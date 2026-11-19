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

from anemoi.utils.dates import as_timedelta

if TYPE_CHECKING:
    from torch import nn

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.tasks.base import BaseTask

LOGGER = logging.getLogger(__name__)


class NightMask:
    """Set accumulated fields to zero in the predictions where the sun was down for the whole accumulation window.

    A field valued at time ``t`` that is accumulated over ``(t - accumulation_period, t]`` (for example hourly
    ``ssr``, ``ssrd`` or ``fdir``) can only be non-zero if the sun was up at some point in that window. The
    predictions of the listed variables are therefore set to zero where the mask variable (by default
    ``insolation``, an instantaneous forcing) is ``<= 0`` both at ``t`` and at ``t - accumulation_period``.
    Testing the value at ``t`` alone would wrongly zero the hour that follows sunset.

    The mask variable is read from the batch, at the output times and one accumulation period earlier, so this
    works with any task whose batch holds these times. The mask is applied before the loss, so the loss and the
    validation metrics see the masked predictions. It is not applied in inference by anemoi-training:
    anemoi-inference provides the equivalent ``night_mask`` post-processor.

    Parameters
    ----------
    variables : list[str]
        Model output variables set to zero at night.
    accumulation_period : str | datetime.timedelta
        Period over which the variables are accumulated, e.g. ``"1h"``.
    mask_variable : str, optional
        Forcing in the dataset whose non-positive values mark night, by default ``"insolation"``.
    """

    def __init__(
        self,
        variables: list[str],
        accumulation_period: str | datetime.timedelta,
        mask_variable: str = "insolation",
    ) -> None:
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        if not self.variables:
            msg = "night_mask: `variables` must list at least one variable."
            raise ValueError(msg)

        self.accumulation_period = as_timedelta(accumulation_period)
        if self.accumulation_period <= datetime.timedelta(0):
            msg = f"night_mask: accumulation_period must be positive, got {self.accumulation_period}."
            raise ValueError(msg)

        self.mask_variable = mask_variable
        self._warned_missing_previous = False

    def __repr__(self) -> str:
        return (
            f"NightMask(variables={self.variables}, mask_variable={self.mask_variable!r}, "
            f"accumulation_period={self.accumulation_period})"
        )

    def validate(self, data_indices: IndexCollection, dataset_name: str = "") -> None:
        """Raise if the mask variable is not in the dataset or a masked variable is not a model output."""
        where = f" of dataset '{dataset_name}'" if dataset_name else ""
        if self.mask_variable not in data_indices.name_to_index:
            msg = (
                f"night_mask: mask variable '{self.mask_variable}' is not in the data{where}. "
                "It must be a variable of the dataset (and not be removed from the data config)."
            )
            raise ValueError(msg)

        not_outputs = [v for v in self.variables if v not in data_indices.model.output.name_to_index]
        if not_outputs:
            msg = f"night_mask: variables {not_outputs} are not model outputs{where}."
            raise ValueError(msg)

    def time_indices(self, task: BaseTask, **step_kwargs: object) -> tuple[list[int], list[int]]:
        """Batch positions of the output times and of the times one accumulation period earlier.

        Positions refer to the time axis of the batch, as for the targets (``task.get_batch_output_indices``).
        An output whose earlier time is not in the batch (e.g. an output at the left boundary of a temporal
        downscaler) uses its own time for both, i.e. only the value at ``t``, and a warning is logged once.
        """
        batch_offsets = task.get_offsets()
        outputs = task.get_output_offsets(**step_kwargs)

        current, previous = [], []
        for offset in outputs:
            current.append(batch_offsets.index(offset))
            earlier = offset - self.accumulation_period
            if earlier in batch_offsets:
                previous.append(batch_offsets.index(earlier))
            else:
                previous.append(current[-1])
                if not self._warned_missing_previous:
                    LOGGER.warning(
                        "night_mask: time %s - %s is not in the batch: using %s only for outputs at such times.",
                        offset,
                        self.accumulation_period,
                        self.mask_variable,
                    )
                    self._warned_missing_previous = True
        return current, previous

    @staticmethod
    def _normaliser_add(pre_processors: nn.Module | None) -> torch.Tensor | None:
        """Offset of the normaliser: the normalised value of a physical zero is ``_norm_add``."""
        if pre_processors is None:
            return None
        for processor in getattr(pre_processors, "processors", {}).values():
            if hasattr(processor, "_norm_add") and hasattr(processor, "_norm_mul"):
                return processor._norm_add  # noqa: SLF001
        return None

    def apply(
        self,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        *,
        data_indices: IndexCollection,
        pre_processors: nn.Module | None,
        time_indices: tuple[list[int], list[int]],
    ) -> torch.Tensor:
        """Mask the predictions of one dataset.

        Parameters
        ----------
        y_pred : torch.Tensor
            Normalised predictions ``(batch, n_output_times, ensemble, grid, n_model_output)``.
        batch : torch.Tensor
            Normalised batch ``(batch, time, ensemble, grid, n_data)`` holding the mask variable.
        data_indices : IndexCollection
            Data indices of the dataset.
        pre_processors : nn.Module | None
            Pre-processors of the dataset, to find the normalised value of a physical zero.
        time_indices : tuple[list[int], list[int]]
            Output and earlier batch positions from :meth:`time_indices`.

        Returns
        -------
        torch.Tensor
            Predictions with the variables set to (normalised) zero where it was night; a new tensor.
        """
        current, previous = time_indices
        name_to_index = data_indices.name_to_index
        model_output = data_indices.model.output.name_to_index

        norm_add = self._normaliser_add(pre_processors)
        if norm_add is None:
            norm_add = torch.zeros(len(name_to_index), device=batch.device)
        norm_add = norm_add.to(device=y_pred.device, dtype=y_pred.dtype)

        # The normaliser is increasing (x * mul + add with mul > 0): "<= 0" is "<= normalised zero".
        dark = batch[..., name_to_index[self.mask_variable]] <= norm_add[name_to_index[self.mask_variable]]
        night = dark[:, current] & dark[:, previous]  # (batch, n_output_times, ensemble or 1, grid)

        if night.shape[-1] != y_pred.shape[-2]:
            msg = (
                f"night_mask: the predictions have {y_pred.shape[-2]} grid points but the batch has "
                f"{night.shape[-1]}. The mask needs predictions and batch on the same grid."
            )
            raise ValueError(msg)

        output_index = torch.tensor([model_output[v] for v in self.variables], device=y_pred.device)
        zero = norm_add[[name_to_index[v] for v in self.variables]]

        masked = torch.where(night.unsqueeze(-1), zero, y_pred[..., output_index])
        y_pred = y_pred.clone()
        y_pred[..., output_index] = masked
        return y_pred
