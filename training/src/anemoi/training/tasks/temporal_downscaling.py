# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

from anemoi.training.diagnostics.callbacks.plot_adapter import TemporalDownscalerPlotAdapter
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import as_timedelta

LOGGER = logging.getLogger(__name__)


class TemporalDownscaler(BaseSingleStepTask):
    """Temporal downscaling task implementation."""

    name: str = "temporal-downscaler"

    def __init__(
        self,
        input_timestep: str,
        output_timestep: str,
        data_frequency: str,        
        output_left_boundary: bool = False,
        output_right_boundary: bool = False,
        **_kwargs,
    ) -> None:
        input_timedelta = as_timedelta(input_timestep)
        output_timedelta = as_timedelta(output_timestep)
        data_frequency = as_timedelta(data_frequency)

        assert output_timedelta.total_seconds() % data_frequency.total_seconds() == 0, \
        "Output timestep must be an integer multiple of data frequency."

        data_output_factor = output_timedelta // data_frequency

        input_offsets = [datetime.timedelta(hours=0), input_timedelta*data_output_factor]

        assert input_timedelta % output_timedelta == datetime.timedelta(
            0,
        ), "Input timestep must be an integer multiple of output timestep for temporal downscaling."
        num_output_steps = input_timedelta // output_timedelta
        output_offsets = [output_timedelta * (i + 1) * data_output_factor for i in range(num_output_steps - 1)]
        if output_left_boundary:
            output_offsets = [datetime.timedelta(hours=0), *output_offsets]

        if output_right_boundary:
            output_offsets = [*output_offsets, input_timedelta*data_output_factor]

        super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)
        self._plot_adapter = TemporalDownscalerPlotAdapter(self)
