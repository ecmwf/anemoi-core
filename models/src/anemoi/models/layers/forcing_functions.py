# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Masks for :class:`~anemoi.models.layers.bounding.ForcingMaskBounding`.

Each is a callable taking the output-time forcings (a ``Source``) and returning a boolean tensor of shape
``(batch, time, ensemble, grid)``.
"""

from __future__ import annotations

import datetime
import math
import operator
from typing import TYPE_CHECKING

import torch

from anemoi.models.preprocessing.normalization import normalization_coefficients
from anemoi.utils.dates import as_timedelta

if TYPE_CHECKING:
    from anemoi.models.data import Source

OPERATORS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
DAYS_PER_YEAR = 365.25


def physical_forcing(forcings: Source, name: str, normalizer: str = "none") -> torch.Tensor:
    """Return forcing ``name`` in physical units."""
    index = forcings.name_to_index[name]
    mul, add = normalization_coefficients(normalizer, forcings.statistics, index, name)
    return (forcings.values[..., index] - add) / mul


class ForcingCondition:
    """``forcing <operator> threshold`` for a forcing of the dataset, e.g. ``lsm < 0.5``, in physical units."""

    def __init__(
        self,
        name: str,
        operator: str = "le",
        threshold: float = 0.0,
        normalizer: str = "none",  # noqa: A002
    ) -> None:
        if operator not in OPERATORS:
            msg = f"Unknown operator {operator!r}, expected one of {list(OPERATORS)}."
            raise ValueError(msg)
        self.name, self.operator, self.threshold, self.normalizer = name, operator, threshold, normalizer

    def __call__(self, forcings: Source) -> torch.Tensor:
        return OPERATORS[self.operator](physical_forcing(forcings, self.name, self.normalizer), self.threshold)


def insolation(cos_julian_day, sin_julian_day, cos_local_time, sin_local_time, latitude, shift_hours: float = 0.0):
    """Insolation ``shift_hours`` after the time of the forcings, ``latitude`` in radians.

    The formula of the ``insolation`` forcing of earthkit-data, written with the ``cos/sin_julian_day`` and
    ``cos/sin_local_time`` forcings: the hour angle is the local-time angle minus pi plus the equation-of-time
    correction, so the longitude is not needed.
    """
    day = 2 * math.pi * shift_hours / (24 * DAYS_PER_YEAR)  # a shift also moves the day angle
    cj = cos_julian_day * math.cos(day) - sin_julian_day * math.sin(day)
    sj = sin_julian_day * math.cos(day) + cos_julian_day * math.sin(day)
    c2, s2 = cj * cj - sj * sj, 2 * sj * cj
    c3, s3 = 4 * cj**3 - 3 * cj, 3 * sj - 4 * sj**3

    declination = torch.deg2rad(
        0.396372 - 22.91327 * cj + 4.025430 * sj - 0.387205 * c2 + 0.051967 * s2 - 0.154527 * c3 + 0.084798 * s3
    )
    angle = torch.deg2rad(0.004297 + 0.107029 * cj - 1.837877 * sj - 0.837378 * c2 - 2.340475 * s2)
    angle = angle + math.radians(15.0 * shift_hours)
    cos_hour_angle = -(cos_local_time * torch.cos(angle) - sin_local_time * torch.sin(angle))

    sin_part = torch.sin(latitude) * torch.sin(declination)
    return (sin_part + torch.cos(latitude) * torch.cos(declination) * cos_hour_angle).clamp(min=0)


class NoSunlightInWindow:
    """True where the sun was down for the whole window ``[t - window, t]`` before each output time ``t``.

    For radiation accumulated over ``window``. The insolation is sampled every hour, as testing only the two ends
    would miss a short day (sunrise and sunset) inside a long window. It is computed from the
    ``cos/sin_julian_day`` and ``cos/sin_local_time`` forcings, which must be forcings of the dataset.

    Parameters
    ----------
    window : str
        Accumulation period, a whole number of hours (the model's output timestep), e.g. ``6h``.
    normalizer : str, optional
        Normalisation method of the four forcings above, by default ``none``.
    """

    FORCINGS = ("cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time")

    def __init__(self, window: str, normalizer: str = "none") -> None:
        hours = as_timedelta(window) / datetime.timedelta(hours=1)
        if hours <= 0 or hours != int(hours):
            msg = f"window must be a positive whole number of hours, got {window!r}."
            raise ValueError(msg)
        self.shifts = [-float(h) for h in range(int(hours) + 1)]
        self.normalizer = normalizer

    def __call__(self, forcings: Source) -> torch.Tensor:
        cj, sj, cl, sl = (physical_forcing(forcings, name, self.normalizer) for name in self.FORCINGS)
        latitude = forcings.coordinates[:, 0].to(device=cj.device, dtype=cj.dtype)  # radians
        return torch.stack([insolation(cj, sj, cl, sl, latitude, shift) for shift in self.shifts]).amax(dim=0) <= 0
