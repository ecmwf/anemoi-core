# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Conversions for the time information carried by samples, sources and templates.

Every entry of a time axis (a gridded step or a tabular window) has a time offset, measured from the
sample's reference time (relative offset 0), and every sample has the absolute ``reference_date`` of that
offset. So ``valid_time[t] = reference_date + time_offsets[t]``.
"""

import datetime
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any

import numpy as np

from anemoi.utils.dates import frequency_to_string

TimeOffsets = tuple[datetime.timedelta, ...]


def as_timedelta(value: Any) -> datetime.timedelta:
    """Return ``value`` (a :class:`datetime.timedelta` or :class:`numpy.timedelta64`) as a timedelta."""
    if isinstance(value, datetime.timedelta):
        return value
    if isinstance(value, np.timedelta64):
        return datetime.timedelta(seconds=int(value.astype("timedelta64[s]").astype(np.int64)))
    msg = f"Expected a datetime.timedelta or numpy.timedelta64 time offset, got {type(value).__name__}: {value!r}."
    raise TypeError(msg)


def as_datetime(value: Any) -> datetime.datetime:
    """Return ``value`` (a :class:`datetime.datetime` or :class:`numpy.datetime64`) as a datetime."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[s]").astype(datetime.datetime)
    msg = f"Expected a datetime.datetime or numpy.datetime64 date, got {type(value).__name__}: {value!r}."
    raise TypeError(msg)


def as_time_offsets(values: Iterable[Any] | None) -> TimeOffsets | None:
    """Return ``values`` as a tuple of timedeltas, or ``None``."""
    if values is None:
        return None
    return tuple(as_timedelta(v) for v in values)


def as_reference_dates(values: Sequence[Any] | None) -> list[datetime.datetime | None] | None:
    """Return one reference date per sample as datetimes (``None`` entries kept), or ``None``."""
    if values is None:
        return None
    return [None if v is None else as_datetime(v) for v in values]


def select_time_offsets(offsets: TimeOffsets | None, indices: Sequence[int]) -> TimeOffsets | None:
    """Return the offsets at ``indices`` (already resolved to ints), or ``None``."""
    if offsets is None:
        return None
    return tuple(offsets[i] for i in indices)


def shift_time_offsets(offsets: TimeOffsets | None, delta: datetime.timedelta) -> TimeOffsets | None:
    """Return the offsets measured from a reference moved on by ``delta``."""
    if offsets is None:
        return None
    return tuple(o - delta for o in offsets)


def shift_reference_dates(
    dates: list[datetime.datetime | None] | None,
    delta: datetime.timedelta,
) -> list[datetime.datetime | None] | None:
    """Return the reference dates moved on by ``delta``."""
    if dates is None:
        return None
    return [None if d is None else d + delta for d in dates]


def check_time_offsets(owner: str, offsets: TimeOffsets | None, time_size: int) -> None:
    """Raise unless ``offsets`` is ``None`` or has one entry per time step."""
    if offsets is not None and len(offsets) != time_size:
        msg = f"{owner} has {time_size} time steps but {len(offsets)} time offsets."
        raise ValueError(msg)


def format_time_offsets(offsets: TimeOffsets) -> str:
    """Return the offsets as a short human-readable string, e.g. ``[-6h, 0h, 6h]``."""

    def _fmt(offset: datetime.timedelta) -> str:
        if offset < datetime.timedelta(0):
            return "-" + frequency_to_string(-offset)
        return frequency_to_string(offset)

    return "[" + ", ".join(_fmt(o) for o in offsets) + "]"
