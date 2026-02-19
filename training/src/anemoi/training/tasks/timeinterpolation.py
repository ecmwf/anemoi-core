# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from operator import itemgetter

import numpy as np

from anemoi.training.tasks.base import BaseSingleStepTask

LOGGER = logging.getLogger(__name__)


class TimeInterpolationTask(BaseSingleStepTask):
    """Time interpolation task implementation."""

    name: str = "timeinterpolation"

    def __init__(
        self,
        explicit_input_times: list[int],
        explicit_output_times: list[int],
        **_kwargs,
    ) -> None:
        self.boundary_times = explicit_input_times  # [0, 6]
        self.interp_times = explicit_output_times  # [1, 2, 3, 4, 5]

        self.imap = np.array(sorted(set(self.boundary_times + self.interp_times)))

    def get_batch_input_time_indices(self, *args, **kwargs) -> list[int]:
        return list(itemgetter(*self.boundary_times)(self.imap))

    def get_batch_output_time_indices(self, *args, **kwargs) -> list[int]:
        return list(itemgetter(*self.interp_times)(self.imap))

    def get_relative_time_indices(self, *args, **kwargs) -> list[int]:
        return self.imap.tolist()
