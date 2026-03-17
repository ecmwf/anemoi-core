# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import as_timedelta

LOGGER = logging.getLogger(__name__)


class TimeInterpolationTask(BaseSingleStepTask):
    """Time interpolation task implementation.

    Input and output offsets are specified as duration strings
    (e.g. ``["0H", "6H"]`` and ``["1H", "2H", "3H", "4H", "5H"]``).
    """

    name: str = "timeinterpolation"

    def __init__(
        self,
        inputs_offsets: list[str],
        outputs_offsets: list[str],
        **_kwargs,
    ) -> None:
        inputs_offsets = [as_timedelta(offset) for offset in inputs_offsets]
        outputs_offsets = [as_timedelta(offset) for offset in outputs_offsets]
        super().__init__(inputs_offsets=inputs_offsets, outputs_offsets=outputs_offsets)
