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

    The condition is ``forcing <operator> threshold`` (e.g. ``insolation <= 0``). It is evaluated on the forcing at
    the time of each output plus each of the ``offsets`` (relative to the output time) and combined over the offsets
    with ``all`` or ``any``. Where the combined condition holds, the predictions of the variables of each entry of
    ``fill_values`` are set to its value. The forcing is read from the batch, so any task whose batch holds the
    required times works, and the mask is applied before the loss, so the loss and the validation metrics see the
    masked predictions.

    The forcing must be a forcing variable of the dataset: it is known at inference time too, unlike a prognostic or
    diagnostic variable, whose true value must not decide the prediction. The mask is not applied at inference by
    anemoi-training.

    The configuration has the same layout as the imputer and the conditional post-processors: the fill value is
    the key and the variables the list, next to the other options (see :meth:`from_config`).

    Example: zero hourly accumulated radiation where the sun was down for the whole accumulation window, that is
    where ``insolation <= 0`` both at ``t`` and at ``t - 1h``::

        ForcingMask(mask_variable="insolation", fill_values={0.0: ["ssr"]}, offsets=["0h", "-1h"])

    Parameters
    ----------
    mask_variable : str
        Forcing the condition is evaluated on.
    fill_values : Mapping[float, list[str]]
        For each fill value, in physical units, the model output variables set to it where the condition holds.
    operator : str, optional
        Comparison with the threshold: ``lt``, ``le``, ``gt`` or ``ge``, by default ``le``.
    threshold : float, optional
        Threshold, in physical (not normalised) units, by default 0.
    offsets : list[str], optional
        Times, relative to each output time, at which the forcing is evaluated, e.g. ``["0h", "-1h"]``. Strings only
        (a bare number is not a valid duration). By default ``["0h"]``: the output time only.
    combine : str, optional
        ``all`` (default): the condition must hold at every offset; ``any``: at one offset at least.
    """

    OPTIONS = ("mask_variable", "operator", "threshold", "offsets", "combine")

    def __init__(
        self,
        mask_variable: str,
        fill_values: Mapping[float, Sequence[str]],
        operator: str = "le",  # noqa: A002
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
            msg = f"forcing_mask: `offsets` must be a non-empty list of durations as strings, got {offsets!r}."
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
        """Build from a dictionary laid out like the imputer configuration.

        Besides ``mask_variable`` (required), ``operator``, ``threshold``, ``offsets`` and ``combine`` (optional,
        see the class), the keys are the fill values and the entries the variables set to them. ``default`` may
        be given as ``"none"`` for consistency with the imputer: there is no default fill value::

            mask_variable: insolation
            operator: le
            threshold: 0
            offsets: [0h, -1h]
            default: "none"
            0.0: [ssr, ssrd, fdir]
            273.15: [t2m]
        """
        config = dict(config)
        default = config.pop("default", "none")
        if default not in ("none", None):
            msg = f"forcing_mask: `default` must be 'none' (there is no default fill value), got {default!r}."
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
                msg = (
                    f"forcing_mask: unknown key {key!r}: expected one of {[*cls.OPTIONS, 'default']} or a numeric "
                    "fill value listing the variables to set to it."
                )
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
        """Raise if the mask variable is not a forcing or a masked variable is not a model output."""
        where = f" of dataset '{dataset_name}'" if dataset_name else ""
        if self.mask_variable not in data_indices.name_to_index:
            msg = (
                f"forcing_mask: mask variable '{self.mask_variable}' is not in the data{where}. "
                "It must be a variable of the dataset (and not be removed from the data config)."
            )
            raise ValueError(msg)
        if self.mask_variable not in data_indices.forcing:
            msg = (
                f"forcing_mask: mask variable '{self.mask_variable}' is not a forcing{where}. "
                "Only forcings can be used: they are known at inference time."
            )
            raise ValueError(msg)

        not_outputs = [v for v in self.variables if v not in data_indices.model.output.name_to_index]
        if not_outputs:
            msg = f"forcing_mask: variables {not_outputs} are not model outputs{where}."
            raise ValueError(msg)

    def time_indices(self, task: BaseTask, **step_kwargs: object) -> torch.Tensor:
        """Batch positions of the forcing for each output and offset.

        Positions refer to the time axis of the batch, as for the targets (``task.get_batch_output_indices``).
        An offset whose time is not in the batch for an output (e.g. ``-1h`` for an output at the left boundary of a
        temporal downscaler) is ignored for that output, and a warning is logged once. If none of the offsets is
        in the batch for an output, an error is raised.

        Returns
        -------
        torch.Tensor
            Integer tensor of shape ``(n_outputs, n_offsets)``. Missing offsets repeat an available one, which
            leaves ``all`` and ``any`` unchanged.
        """
        batch_offsets = task.get_offsets()
        indices = []
        for output in task.get_output_offsets(**step_kwargs):
            available = [batch_offsets.index(output + o) for o in self.offsets if output + o in batch_offsets]
            if not available:
                msg = (
                    f"forcing_mask: for the output at {output}, none of the times {[str(o) for o in self.offsets]} "
                    "relative to it is in the batch."
                )
                raise ValueError(msg)
            if len(available) < len(self.offsets) and not self._warned_missing_times:
                LOGGER.warning(
                    "forcing_mask: some of the offsets %s are not in the batch for the output at %s: they are ignored "
                    "for such outputs.",
                    [str(o) for o in self.offsets],
                    output,
                )
                self._warned_missing_times = True
            indices.append(available + [available[0]] * (len(self.offsets) - len(available)))
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _normaliser(pre_processors: nn.Module | None, n_variables: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(add, mul)`` of the normaliser: ``x_normalised = x * mul + add``. Identity if there is none."""
        if pre_processors is not None:
            for processor in getattr(pre_processors, "processors", {}).values():
                if hasattr(processor, "_norm_add") and hasattr(processor, "_norm_mul"):
                    return processor._norm_add, processor._norm_mul  # noqa: SLF001
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
        """Mask the predictions of one dataset.

        The threshold and the fill value are given in physical units and converted with the normaliser, which
        increases with the physical value, so the comparison can be done on the normalised forcing.

        Parameters
        ----------
        y_pred : torch.Tensor
            Normalised predictions ``(batch, n_output_times, ensemble, grid, n_model_output)``.
        batch : torch.Tensor
            Normalised batch ``(batch, time, ensemble, grid, n_data)`` holding the forcing.
        data_indices : IndexCollection
            Data indices of the dataset.
        pre_processors : nn.Module | None
            Pre-processors of the dataset, to convert physical values to normalised ones.
        time_indices : torch.Tensor
            ``(n_outputs, n_offsets)`` batch positions from :meth:`time_indices`.

        Returns
        -------
        torch.Tensor
            Predictions with the variables overwritten where the condition holds; a new tensor.
        """
        name_to_index = data_indices.name_to_index
        model_output = data_indices.model.output.name_to_index

        add, mul = self._normaliser(pre_processors, len(name_to_index))
        add, mul = add.to(device=y_pred.device, dtype=y_pred.dtype), mul.to(device=y_pred.device, dtype=y_pred.dtype)

        forcing_index = name_to_index[self.mask_variable]
        threshold = self.threshold * mul[forcing_index] + add[forcing_index]
        holds = OPERATORS[self.operator](batch[..., forcing_index], threshold)  # (batch, time, ensemble or 1, grid)

        time_indices = time_indices.to(holds.device)
        per_offset = holds[:, time_indices]  # (batch, n_outputs, n_offsets, ensemble or 1, grid)
        mask = per_offset.all(dim=2) if self.combine == "all" else per_offset.any(dim=2)

        if mask.shape[-1] != y_pred.shape[-2]:
            msg = (
                f"forcing_mask: the predictions have {y_pred.shape[-2]} grid points but the batch has "
                f"{mask.shape[-1]}. The mask needs predictions and batch on the same grid."
            )
            raise ValueError(msg)

        output_index = torch.tensor([model_output[v] for v in self.variables], device=y_pred.device)
        data_index = [name_to_index[v] for v in self.variables]
        fill = torch.tensor(self._fills, device=y_pred.device, dtype=y_pred.dtype) * mul[data_index] + add[data_index]

        masked = torch.where(mask.unsqueeze(-1), fill, y_pred[..., output_index])
        y_pred = y_pred.clone()
        y_pred[..., output_index] = masked
        return y_pred
