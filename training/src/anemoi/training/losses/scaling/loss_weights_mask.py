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

import torch

from anemoi.training.losses.scaling import BaseScaler

LOGGER = logging.getLogger(__name__)


class NaNMaskScaler(BaseScaler):

    def get_scaling(self) -> torch.Tensor:
        """Get loss scaling.

        Get  mask multiplying NaN locations with zero.
        At this stage, returns a loss slicing mask with all values set to 1.
        When calling the imputer for the first time, the NaN positions are available.
        Before first application of loss function, the mask is replaced.
        """
        return torch.ones((1, 1))
