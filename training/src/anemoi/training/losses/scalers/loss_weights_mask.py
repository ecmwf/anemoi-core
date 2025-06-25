# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from anemoi.models.interface import AnemoiModelInterface

LOGGER = logging.getLogger(__name__)


class NaNMaskScaler(BaseUpdatingScaler):

    scale_dims: tuple[TensorDim] = (TensorDim.BATCH_SIZE, TensorDim.GRID, TensorDim.VARIABLE)

    def initial_scaling_values(self) -> torch.Tensor | None:
        """Get initial scaling values.

        Returns
        -------
        torch.Tensor
            Initial scaling values, default is a tenser of ones.
        """
        # TODO(sara,harrison): should this in general (for all scalers) be a tensor and not a numpy array?
        # TODO(sara,harrison): could this be on the model device?
        # by default on CPU and for the updating scalers this means the updates happen on CPU
        # lots of moving scalers around, so this is not ideal
        return torch.ones(tuple([1] * len(self.scale_dims)))

    def on_train_batch_start(self, model: AnemoiModelInterface) -> None:
        """Update loss scaling.

        Get  mask multiplying NaN locations with zero.
        At this stage, returns a loss slicing mask with all values set to 1.
        Always when calling the imputer, the NaN positions are updated.
        Before every application of training loss function, the mask is replaced.
        """
        loss_weights_mask = self.get_scaling_values()
        device = loss_weights_mask.device
        # iterate over all pre-processors and check if they have a loss_mask_training attribute
        for pre_processor in model.pre_processors.processors.values():
            if hasattr(pre_processor, "loss_mask_training"):
                loss_weights_mask = loss_weights_mask * pre_processor.loss_mask_training.to(device)

        return loss_weights_mask

    def on_valid_batch_start(self, model: AnemoiModelInterface) -> None:
        """Update loss scaling for validation batch."""
        return self.on_train_batch_start(model)
