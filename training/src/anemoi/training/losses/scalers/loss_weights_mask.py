# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np

from anemoi.models.preprocessing import BasePreprocessor
from anemoi.training.losses.scalers.base_scaler import BaseDelayedScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class NaNMaskScaler(BaseDelayedScaler):

    scale_dims: tuple[TensorDim] = (TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(self, norm: str | None = None, **kwargs) -> None:
        """Initialise NanMaskScaler.

        Parameters
        ----------
        norm : str, optional
            Type of normalisation to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        del kwargs

    def get_scaling_values(self) -> np.ndarray:
        return np.ones(tuple([1] * len(self.scale_dims)))

    def get_delayed_scaling_values(self, processors: list[BasePreprocessor]) -> np.ndarray:
        """Get loss scaling.

        Get mask multiplying NaN locations with zero.
        At this stage, returns a loss slicing mask with all values set to 1.
        When calling the imputer for the first time, the NaN positions are available.
        Before first application of loss function, the mask is replaced.

        Parameters
        ----------
        processors : list[BasePreprocessor]
            List of pre-processors to check for loss_mask_training attribute.
        """
        loss_weights_mask = np.ones((1, 1))
        # iterate over all provided pre-processors and check if they have a loss_mask_training attribute
        for pre_processor in processors:
            if hasattr(pre_processor, "loss_mask_training"):
                loss_weights_mask = loss_weights_mask * pre_processor.loss_mask_training.cpu().numpy()
        return loss_weights_mask
