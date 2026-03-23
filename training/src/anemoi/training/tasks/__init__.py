# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .custom import CustomTask
from .forecasting import ForecastingTask
from .temporal_downscaling import TemporalDownscalingTask
from .timeless import AutoencodingTask
from .timeless import SpatialDownscalingTask

__all__ = [
    "AutoencodingTask",
    "CustomTask",
    "ForecastingTask",
    "SpatialDownscalingTask",
    "TemporalDownscalingTask",
]
