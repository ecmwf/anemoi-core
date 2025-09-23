# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class NaNMaskScaler(BaseUpdatingScaler):

    scale_dims: tuple[TensorDim] = (TensorDim.BATCH_SIZE, TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(self, norm: str | None = None, use_processors_tendencies: bool = False, **kwargs) -> None:
        """Initialise NanMaskScaler.

        Parameters
        ----------
        norm : str, optional
            Type of normalisation to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        self.use_processors_tendencies = use_processors_tendencies
        del kwargs

    def on_batch_start(self, model: AnemoiModelInterface, dataset_name: str | None = None) -> torch.Tensor | None:
        """Update loss scaling.

        Get mask multiplying NaN locations with zero.
        At this stage, returns a loss slicing mask with all values set to 1.
        Always when calling the imputer, the NaN positions are updated.
        Before every application of training loss function, the mask is replaced.

        Parameters
        ----------
        model : AnemoiModelInterface
            The model.
        dataset_name : str, optional
            The dataset name for multi-dataset scenarios.
        """
        loss_weights_mask = None
        processors = []

        # Handle pre_processors
        if hasattr(model, "pre_processors"):
            assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
            # Multi-dataset case: get pre_processors for specific dataset
            if dataset_name in model.pre_processors:
                processors.append(model.pre_processors[dataset_name])

        # Handle pre_processors_tendencies
        if self.use_processors_tendencies and hasattr(model, "pre_processors_tendencies"):
            # Multi-dataset case: get pre_processors_tendencies for specific dataset
            assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
            if dataset_name in model.pre_processors_tendencies:
                processors.append(model.pre_processors_tendencies[dataset_name])

        # iterate over all pre-processors and check if they have a loss_mask_training attribute
        for pre_processors in processors:
            for pre_processor in pre_processors.processors.values():
                if hasattr(pre_processor, "loss_mask_training"):
                    if loss_weights_mask is None:
                        loss_weights_mask = pre_processor.loss_mask_training
                    else:
                        # multiply the masks together
                        loss_weights_mask = loss_weights_mask * pre_processor.loss_mask_training

        return loss_weights_mask
