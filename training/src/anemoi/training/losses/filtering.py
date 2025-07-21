# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function


# TODO(Harrison): Consider renaming and reworking to a RemappingLossWrapper or similar, as it remaps variables
class FilteringLossWrapper(BaseLoss):
    """Loss wrapper to filter variables to compute the loss on."""

    def __init__(
        self,
        loss: dict[str, Any] | Callable | BaseLoss,
        predicted_variables: list[str] | None = None,
        target_variables: list[str] | None = None,
        **kwargs,
    ):
        """Loss wrapper to filter variables to compute the loss on.

        Parameters
        ----------
        loss : Type[torch.nn.Module] | dict[str, Any]
            wrapped loss
        predicted_variables : list[str] | None
            predicted variables to keep, if None, all variables are kept
        target_variables : list[str] | None
            target variables to keep, if None, all variables are kept
        """
        if predicted_variables and target_variables:
            assert len(predicted_variables) == len(
                target_variables,
            ), "predicted and target variables must have the same length for loss computation"

        super().__init__()

        self._loss_scaler_specification = {}
        if isinstance(loss, str):
            self._loss_scaler_specification = ["*"]
            self.loss = get_loss_function(DictConfig({"_target_": loss}), scalers={}, **dict(kwargs))
        elif isinstance(loss, DictConfig | dict):
            self._loss_scaler_specification = loss.pop("scalers", ["*"])
            self.loss = get_loss_function(loss, scalers={}, **dict(kwargs))
        elif isinstance(loss, type):
            self._loss_scaler_specification = ["*"]
            self.loss = loss(**kwargs)
        elif isinstance(loss, BaseLoss):
            self._loss_scaler_specification = loss.scaler
            self.loss = loss
        else:
            msg = f"Invalid loss type provided: {type(loss)}. Expected a str or dict or BaseLoss."
            raise TypeError(msg)

        self.predicted_variables = predicted_variables
        self.target_variables = target_variables

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Hook to set the data indices for the loss."""
        self.data_indices = data_indices
        name_to_index = data_indices.data.output.name_to_index
        model_output = data_indices.model.output
        output_indices = model_output.full

        if self.predicted_variables is not None:
            predicted_indices = [model_output.name_to_index[name] for name in self.predicted_variables]
        else:
            predicted_indices = output_indices
        if self.target_variables is not None:
            target_indices = [name_to_index[name] for name in self.target_variables]
        else:
            target_indices = output_indices

        assert len(predicted_indices) == len(
            target_indices,
        ), "predicted and target variables must have the same length for loss computation"

        self.predicted_indices = predicted_indices
        self.target_indices = target_indices

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        squash = kwargs.get("squash", True)
        if squash:
            return self.loss(pred[..., self.predicted_indices], target[..., self.target_indices], **kwargs)
        len_model_output = pred.shape[-1]
        loss = torch.zeros(len_model_output, dtype=pred.dtype, device=pred.device, requires_grad=False)
        loss_per_variable = self.loss(
            pred[..., self.predicted_indices],
            target[..., self.target_indices],
            **kwargs,
        )
        loss[self.predicted_indices] = loss_per_variable
        return loss
