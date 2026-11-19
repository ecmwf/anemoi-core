# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from anemoi.utils.dates import as_timedelta

if TYPE_CHECKING:
    from torch import nn

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.tasks.base import BaseTask

LOGGER = logging.getLogger(__name__)

OPERATORS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
COMBINE = ("all", "any")


class ForcingMask:
    """Set predicted variables to fixed values where a condition on a forcing holds.

    The condition ``forcing <operator> threshold`` is evaluated at each output time plus each offset and combined
    over the offsets with ``all`` or ``any``. The forcing is read from the batch and the mask is applied to the
    predictions before the loss. It is not applied at inference.

    Parameters
    ----------
    mask_variable : str
        Forcing the condition is evaluated on.
    fill_values : Mapping[float, list[str]]
        Model output variables to set to each value (physical units) where the condition holds.
    operator : str, optional
        ``lt``, ``le``, ``gt`` or ``ge``, by default ``le``.
    threshold : float, optional
        Threshold in physical units, by default 0.
    offsets : list[str], optional
        Times relative to each output at which the forcing is evaluated, as duration strings (e.g. ``["0h", "-1h"]``).
    combine : str, optional
        ``all`` (default) or ``any`` of the offsets must satisfy the condition.
    """

    OPTIONS = ("mask_variable", "operator", "threshold", "offsets", "combine")

    def __init__(
        self,
        mask_variable: str,
        fill_values: Mapping[float, Sequence[str]],
        operator: str = "le",
        threshold: float = 0.0,
        offsets: Sequence[str] = ("0h",),
        combine: str = "all",
    ) -> None:
        self.fill_values: dict[float, list[str]] = {}
        for value, names in fill_values.items():
            names = [names] if isinstance(names, str) else list(names or [])
            if names:
                self.fill_values[float(value)] = names
        self.variables = [v for names in self.fill_values.values() for v in names]
        if not self.variables:
            msg = "forcing_mask: at least one fill value must list at least one variable."
            raise ValueError(msg)
        duplicates = sorted({v for v in self.variables if self.variables.count(v) > 1})
        if duplicates:
            msg = f"forcing_mask: variables {duplicates} are listed under more than one fill value."
            raise ValueError(msg)
        self._fills = [value for value, names in self.fill_values.items() for _ in names]

        if operator not in OPERATORS:
            msg = f"forcing_mask: unknown operator '{operator}', expected one of {sorted(OPERATORS)}."
            raise ValueError(msg)
        if combine not in COMBINE:
            msg = f"forcing_mask: unknown combine '{combine}', expected one of {list(COMBINE)}."
            raise ValueError(msg)

        offsets = [offsets] if isinstance(offsets, str) else list(offsets)
        if not offsets or not all(isinstance(o, str) for o in offsets):
            msg = f"forcing_mask: `offsets` must be a non-empty list of duration strings, got {offsets!r}."
            raise ValueError(msg)

        self.mask_variable = mask_variable
        self.operator = operator
        self.threshold = float(threshold)
        self.offsets = [as_timedelta(o) for o in offsets]
        if len(set(self.offsets)) != len(self.offsets):
            msg = f"forcing_mask: `offsets` must be distinct, got {offsets}."
            raise ValueError(msg)
        self.combine = combine
        self._warned_missing_times = False

    @classmethod
    def from_config(cls, config: Mapping) -> ForcingMask:
        """Build from a config laid out like the imputer: numeric keys are fill values, entries their variables."""
        config = dict(config)
        default = config.pop("default", "none")
        if default not in ("none", None):
            msg = f"forcing_mask: `default` must be 'none', got {default!r}."
            raise ValueError(msg)
        if "mask_variable" not in config:
            msg = "forcing_mask: `mask_variable` is required."
            raise ValueError(msg)

        options = {key: config.pop(key) for key in cls.OPTIONS if key in config}
        fill_values: dict[float, Sequence[str]] = {}
        for key, names in config.items():
            try:
                if isinstance(key, bool):
                    raise TypeError
                value = float(key)
            except (TypeError, ValueError):
                msg = f"forcing_mask: unknown key {key!r}: expected an option or a numeric fill value."
                raise ValueError(msg) from None
            if value in fill_values:
                msg = f"forcing_mask: the fill value {value} is given more than once."
                raise ValueError(msg)
            fill_values[value] = names
        return cls(fill_values=fill_values, **options)

    def __repr__(self) -> str:
        offsets = [str(o) for o in self.offsets]
        return (
            f"ForcingMask(condition='{self.mask_variable} {self.operator} {self.threshold}' "
            f"{self.combine} of offsets={offsets}, fill_values={self.fill_values})"
        )

    def validate(self, data_indices: IndexCollection, dataset_name: str = "") -> None:
        """Raise if the mask variable is not a forcing of the data or a masked variable is not a model output."""
        where = f" of dataset '{dataset_name}'" if dataset_name else ""
        if self.mask_variable not in data_indices.name_to_index:
            msg = f"forcing_mask: mask variable '{self.mask_variable}' is not in the data{where}."
            raise ValueError(msg)
        if self.mask_variable not in data_indices.forcing:
            msg = f"forcing_mask: mask variable '{self.mask_variable}' is not a forcing{where}."
            raise ValueError(msg)

        not_outputs = [v for v in self.variables if v not in data_indices.model.output.name_to_index]
        if not_outputs:
            msg = f"forcing_mask: variables {not_outputs} are not model outputs{where}."
            raise ValueError(msg)

    def time_indices(self, task: BaseTask, **step_kwargs: object) -> torch.Tensor:
        """Batch positions of the forcing, shape ``(n_outputs, n_offsets)``.

        An offset whose time is not in the batch is ignored for that output (a warning is logged once) and an error
        is raised if none is available.
        """
        batch_offsets = task.get_offsets()
        indices = []
        for output in task.get_output_offsets(**step_kwargs):
            available = [batch_offsets.index(output + o) for o in self.offsets if output + o in batch_offsets]
            if not available:
                msg = f"forcing_mask: no offset {[str(o) for o in self.offsets]} is in the batch for {output}."
                raise ValueError(msg)
            if len(available) < len(self.offsets) and not self._warned_missing_times:
                LOGGER.warning("forcing_mask: some offsets are not in the batch for the output at %s: ignored.", output)
                self._warned_missing_times = True
            indices.append(available + [available[0]] * (len(self.offsets) - len(available)))  # pad: all/any unchanged
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _normaliser(pre_processors: nn.Module | None, n_variables: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(add, mul)`` such that ``x_normalised = x * mul + add``; identity without a normaliser."""
        if pre_processors is not None:
            for processor in getattr(pre_processors, "processors", {}).values():
                if hasattr(processor, "_norm_add") and hasattr(processor, "_norm_mul"):
                    return processor._norm_add, processor._norm_mul
        return torch.zeros(n_variables), torch.ones(n_variables)

    def apply(
        self,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        *,
        data_indices: IndexCollection,
        pre_processors: nn.Module | None,
        time_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return a copy of the normalised predictions with the masked variables overwritten.

        ``y_pred`` is ``(batch, n_outputs, ensemble, grid, n_model_output)`` and ``batch`` is
        ``(batch, time, ensemble, grid, n_data)``. The threshold and fill values are converted to normalised units.
        """
        name_to_index = data_indices.name_to_index
        model_output = data_indices.model.output.name_to_index

        add, mul = self._normaliser(pre_processors, len(name_to_index))
        add, mul = add.to(device=y_pred.device, dtype=y_pred.dtype), mul.to(device=y_pred.device, dtype=y_pred.dtype)

        forcing_index = name_to_index[self.mask_variable]
        threshold = self.threshold * mul[forcing_index] + add[forcing_index]  # the normaliser is increasing
        holds = OPERATORS[self.operator](batch[..., forcing_index], threshold)  # (batch, time, ensemble or 1, grid)

        per_offset = holds[:, time_indices.to(holds.device)]  # (batch, n_outputs, n_offsets, ensemble or 1, grid)
        mask = per_offset.all(dim=2) if self.combine == "all" else per_offset.any(dim=2)

        if mask.shape[-1] != y_pred.shape[-2]:
            msg = f"forcing_mask: {y_pred.shape[-2]} prediction grid points but {mask.shape[-1]} in the batch."
            raise ValueError(msg)

        output_index = torch.tensor([model_output[v] for v in self.variables], device=y_pred.device)
        data_index = [name_to_index[v] for v in self.variables]
        fill = torch.tensor(self._fills, device=y_pred.device, dtype=y_pred.dtype) * mul[data_index] + add[data_index]

        masked = torch.where(mask.unsqueeze(-1), fill, y_pred[..., output_index])
        y_pred = y_pred.clone()
        y_pred[..., output_index] = masked
        return y_pred
