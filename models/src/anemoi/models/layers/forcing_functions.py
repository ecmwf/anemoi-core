# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Quantities computed from the output-time forcings, for use as the mask of a ForcingMaskBounding."""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING

import torch
from anemoi.utils.dates import as_timedelta

from anemoi.models.preprocessing.normalization import normalization_coefficients

if TYPE_CHECKING:
    from anemoi.models.data import Source

DAYS_PER_YEAR = 365.25


def physical_forcing(forcings: Source, name: str, normalizer: str = "none") -> torch.Tensor:
    """Return forcing ``name`` in physical units, ``(batch, time, ensemble, grid)``."""
    if name not in forcings.name_to_index:
        msg = f"Forcing {name!r} is not among the target forcings {sorted(forcings.name_to_index)}."
        raise ValueError(msg)
    index = forcings.name_to_index[name]
    mul, add = normalization_coefficients(normalizer, forcings.statistics, index, name)
    return (forcings.values[..., index] - add) / mul


def insolation(
    cos_julian_day: torch.Tensor,
    sin_julian_day: torch.Tensor,
    cos_local_time: torch.Tensor,
    sin_local_time: torch.Tensor,
    latitude: torch.Tensor,
    shift_hours: float = 0.0,
) -> torch.Tensor:
    """Cosine of the solar zenith angle, clipped at zero, ``shift_hours`` after the time of the forcings.

    Same formula as the ``insolation`` forcing of earthkit-data (and so of anemoi datasets), written in terms of
    the ``cos/sin_julian_day`` and ``cos/sin_local_time`` forcings: the solar hour angle is the local-time angle
    minus pi plus the equation-of-time correction, so the longitude is not needed. ``latitude`` is in radians.
    """
    # a shift moves the day angle (2 pi jd / 365.25) and the local-time angle (2 pi lt / 24)
    day = 2 * math.pi * shift_hours / (24 * DAYS_PER_YEAR)
    cj = cos_julian_day * math.cos(day) - sin_julian_day * math.sin(day)
    sj = sin_julian_day * math.cos(day) + cos_julian_day * math.sin(day)
    c2, s2 = cj * cj - sj * sj, 2 * sj * cj
    c3, s3 = 4 * cj**3 - 3 * cj, 3 * sj - 4 * sj**3

    declination = torch.deg2rad(
        0.396372 - 22.91327 * cj + 4.025430 * sj - 0.387205 * c2 + 0.051967 * s2 - 0.154527 * c3 + 0.084798 * s3
    )
    correction = torch.deg2rad(0.004297 + 0.107029 * cj - 1.837877 * sj - 0.837378 * c2 - 2.340475 * s2)
    angle = correction + math.radians(15.0 * shift_hours)
    cos_hour_angle = -(cos_local_time * torch.cos(angle) - sin_local_time * torch.sin(angle))

    sin_part = torch.sin(latitude) * torch.sin(declination)
    zenith = sin_part + torch.cos(latitude) * torch.cos(declination) * cos_hour_angle
    return zenith.clamp(min=0)


class MaxInsolationOverWindow:
    """Maximum insolation over the window ``[t - window, t]`` before each output time ``t``.

    Zero exactly where the sun was down for the whole window, e.g. the accumulation period of ``ssr``. The
    insolation is sampled every ``step``: for a window longer than a few hours, testing only the two ends would
    miss a short day (sunrise and sunset) inside the window.

    Parameters
    ----------
    window : str | datetime.timedelta
        Length of the window, e.g. the model's output timestep (``1h``, ``6h``).
    step : str | datetime.timedelta, optional
        Sampling interval within the window, by default ``1h``.
    normalizer : str, optional
        Normalisation of the ``cos/sin_julian_day`` and ``cos/sin_local_time`` forcings, by default ``none``.
    """

    FORCINGS = ("cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time")

    def __init__(
        self,
        window: str | datetime.timedelta,
        step: str | datetime.timedelta = "1h",
        normalizer: str = "none",
    ) -> None:
        window, step = as_timedelta(window), as_timedelta(step)
        if window <= datetime.timedelta(0) or step <= datetime.timedelta(0):
            msg = f"MaxInsolationOverWindow: window and step must be positive, got {window} and {step}."
            raise ValueError(msg)
        n_steps = math.ceil(window / step)
        hours = window.total_seconds() / 3600
        self.shifts = [-min(i * step.total_seconds() / 3600, hours) for i in range(n_steps + 1)]
        self.normalizer = normalizer

    def __repr__(self) -> str:
        return f"MaxInsolationOverWindow(shifts_hours={self.shifts})"

    def __call__(self, forcings: Source) -> torch.Tensor:
        """Return the maximum insolation over the window, ``(batch, time, ensemble, grid)``."""
        cj, sj, cl, sl = (physical_forcing(forcings, name, self.normalizer) for name in self.FORCINGS)
        latitude = forcings.coordinates[:, 0].to(device=cj.device, dtype=cj.dtype)  # radians, (grid,)
        values = [insolation(cj, sj, cl, sl, latitude, shift) for shift in self.shifts]
        return torch.stack(values).amax(dim=0)
