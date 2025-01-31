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

import numpy as np
from typing import TYPE_CHECKING

from anemoi.training.losses.scaling import BaseScaler

if TYPE_CHECKING:
    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class NaNMaskScaler(BaseScaler):

    def __init__(self, data_indices: IndexCollection, norm: str = None, **kwargs) -> None:
        """Initialise BaseScaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(data_indices, (-2, -1), norm=norm)
        del kwargs

    def get_scaling(self) -> np.ndarray:
        """Get loss scaling.

        Get  mask multiplying NaN locations with zero.
        At this stage, returns a loss slicing mask with all values set to 1.
        When calling the imputer for the first time, the NaN positions are available.
        Before first application of loss function, the mask is replaced.
        """
        return np.ones((1, 1))
