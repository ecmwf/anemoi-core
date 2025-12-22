# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spatial-domain losses.

Note
----
Spectral losses that historically lived in this module (e.g. ``LogFFT2Distance``)
have been moved to :mod:`anemoi.training.losses.spectral`. They are re-exported
here for backwards compatibility.
"""

from __future__ import annotations

from anemoi.training.losses.spectral import FourierCorrelationLoss
from anemoi.training.losses.spectral import LogFFT2Distance

__all__ = [
    "FourierCorrelationLoss",
    "LogFFT2Distance",
]