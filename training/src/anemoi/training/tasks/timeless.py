# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

from anemoi.training.tasks.base import BaseSingleStepTask


class BaseTimelessTask(BaseSingleStepTask):
    """Base class for timeless tasks.

    Both input and output are a single snapshot at t=0.
    """

    def __init__(self, **_kwargs) -> None:
        super().__init__(input_offset=[datetime.timedelta(0)], output_offset=[datetime.timedelta(0)])


class DownscalingTask(BaseTimelessTask):
    """Downscaling task implementation."""

    name: str = "downscaling"


class AutoencodingTask(BaseTimelessTask):
    """Autoencoding task implementation."""

    name: str = "autoencoding"
