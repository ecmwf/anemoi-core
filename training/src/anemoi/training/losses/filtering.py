# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
from collections.abc import Callable
from typing import Any

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim


# TODO(Harrison): Consider renaming and reworking to a RemappingLossWrapper or similar, as it remaps variables
class FilteringLossWrapper(BaseLoss):
    """Loss wrapper to filter variables to compute the loss on."""

    def __init__(
        self,
        loss: dict[str, Any] | Callable | BaseLoss,
        predicted_variables: list[str] | None = None,
        target_variables: list[str] | None = None,
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
        assert isinstance(
            loss,
            BaseLoss,
        ), f"Invalid loss type provided: {type(loss)}. Expected a str or dict or BaseLoss."
        self.loss = loss
        self.predicted_variables = predicted_variables
        self.target_variables = target_variables

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        dimension = dimension if isinstance(dimension, int) else dimension[0]
        if dimension == TensorDim.VARIABLE and self.predicted_variables is not None:
            # filter scaler to only predicted variables
            # target variables will be scaled like the predicted variables since that is what
            # it compares to in the loss
            # even if a different scaling exists for the target variable
            scaler = scaler[self.predicted_indices]
        self.scaler.add_scaler(dimension=dimension, scaler=scaler, name=name)

    def set_data_indices(self, data_indices: IndexCollection) -> BaseLoss:
        """Hook to set the data indices for the loss."""
        self.data_indices = data_indices
        name_to_index = data_indices.data.output.name_to_index
        data_output = data_indices.data.output
        model_output = data_indices.model.output
        model_output_indices = model_output.full

        if self.predicted_variables is not None:
            predicted_indices = [model_output.name_to_index[name] for name in self.predicted_variables]
        else:
            predicted_indices = model_output_indices
            self.predicted_variables = model_output.includes
        if self.target_variables is not None:
            target_indices = [name_to_index[name] for name in self.target_variables]
        else:
            self.target_variables = [var for var in data_output.includes if var in self.predicted_variables]
            target_indices = torch.tensor(
                [i for (n, i) in name_to_index.items() if n in self.target_variables],
                dtype=data_output.full.dtype,
            )

        assert len(predicted_indices) == len(
            target_indices,
        ), "predicted and target variables must have the same length for loss computation"

        self.predicted_indices = predicted_indices
        self.target_indices = target_indices
        return self

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
