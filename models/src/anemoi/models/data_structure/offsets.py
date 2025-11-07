# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import logging
from math import gcd
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from anemoi.models.data_structure import SampleProvider

from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)

# Utility functions to handle offsets


def offset_to_string(x) -> str:
    # copied here to make sure that the automatically generated keys are stable
    # so we don't use frequency_to_string from anemoi.utils

    assert isinstance(x, datetime.timedelta), type(x)

    total_seconds = int(x.total_seconds())

    if not total_seconds:
        return "0h"

    if total_seconds < 0:
        return f"-{offset_to_string(-x)}"

    if total_seconds % (24 * 3600) == 0 and total_seconds >= 10 * (24 * 3600):
        return f"{total_seconds // (24 * 3600)}d"

    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"

    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"

    return f"{total_seconds}s"


def offset_to_timedelta(x) -> datetime.timedelta:
    if isinstance(x, str) and x.startswith("m"):
        x = "-" + x[1:]
    return frequency_to_timedelta(x)


def offset_to_np_timedelta(v) -> np.timedelta64:
    if isinstance(v, np.timedelta64):
        return v.astype("timedelta64[s]")
    v = offset_to_timedelta(v)
    v = np.timedelta64(v, "s")
    return v


def sum_offsets(a, b) -> str:
    a = offset_to_timedelta(a)
    b = offset_to_timedelta(b)
    x = a + b
    return offset_to_string(x)


def normalise_date(v):
    if isinstance(v, np.datetime64):
        return v.astype("datetime64[s]")
    if isinstance(v, datetime.datetime):
        return np.datetime64(v, "s")
    # if isinstance(v, str):
    #    return np.datetime64(v)
    raise ValueError(f"Cannot normalise date {v} of type {type(v)}")


class _DatesBlock:
    # represents a block of dates with start, end, frequency and missing dates
    # missing can be specified as a list of dates or as a list of indices
    #
    # internally, missing is always stored as a list of dates
    # missing_indices can be used to get the list of indices

    def __init__(
        self,
        start: datetime.datetime | np.datetime64,
        end: datetime.datetime | np.datetime64,
        frequency: str | np.timedelta64,
        missing_indices: NDArray[np.int64] | list[int] | set[int],
    ):
        self.start: np.datetime64 = normalise_date(start)
        self.end: np.datetime64 = normalise_date(end)
        self.frequency: np.timedelta64 = offset_to_np_timedelta(frequency)

        if self.start > self.end:
            raise ValueError(f"Invalid DatesBlock with start {self.start} > end {self.end}")
        if self.frequency <= np.timedelta64(0, "s"):
            raise ValueError(f"Invalid DatesBlock with non-positive frequency {self.frequency}")

        self.missing: NDArray[np.datetime64] = self._missing_as_dates(missing_indices)
        if np.any(self.missing < self.start) or np.any(self.missing > self.end):
            raise ValueError(f"Missing dates {self.missing} are out of range [{self.start}, {self.end}]")

        assert self.start.dtype == "datetime64[s]", self.start.dtype
        assert self.end.dtype == "datetime64[s]", self.end.dtype
        assert self.frequency.dtype == "timedelta64[s]", self.frequency.dtype
        assert self.missing.dtype == "datetime64[s]", self.missing.dtype

    @property
    def missing_indices(self):
        # Get the indices of the missing dates
        return self._missing_as_indices(self.missing)

    @property
    def length_including_missing(self):
        return int((self.end - self.start) / self.frequency) + 1

    @property
    def length_removing_missing(self):
        return self.length_including_missing - len(self.missing)

    def __len__(self):
        return self.length_removing_missing

    def _missing_as_dates(self, missing: np.ndarray | list[int] | set[int]) -> np.ndarray:  # ["datetime64[s]"]
        if isinstance(missing, set):
            missing = np.array(sorted(list(missing)), dtype=int)
        if isinstance(missing, list) and all(isinstance(i, int) for i in missing):
            missing = np.array(missing, dtype=int)
        assert isinstance(missing, np.ndarray), type(missing)

        if missing.dtype == "datetime64[s]":
            return missing
        assert isinstance(missing, np.ndarray), type(missing)
        assert missing.dtype == int, missing.dtype
        assert isinstance(self.frequency, np.timedelta64), type(self.frequency)

        return self.start + missing * self.frequency

    def _missing_as_indices(self, missing: np.ndarray) -> np.ndarray[int]:
        if isinstance(missing, np.ndarray) and missing.dtype == int:
            return missing

        missing = np.array(missing, dtype="datetime64[s]")
        missing = (missing - np.datetime64(self.start)) / self.frequency
        # check that all missing are integers
        for v in missing:
            if int(v) != v:
                raise ValueError(
                    f"Missing date {v} is not aligned with start {self.start} and frequency {self.frequency}"
                )
        return missing.astype(int)

    def __add__(self, offset: str | np.timedelta64) -> "_DatesBlock":
        offset = offset_to_np_timedelta(offset)
        return _DatesBlock(
            self.start + offset,
            self.end + offset,
            self.frequency,
            self.missing + offset,
        )

    def __sub__(self, offset: str | np.timedelta64) -> "_DatesBlock":
        return self + (-offset_to_np_timedelta(offset))

    def __contain__(self, date):
        assert False, "not used"
        date = normalise_date(date)
        i = (date - self.start) / self.frequency
        return int(i) == i

    def __and__(self, other: "_DatesBlock") -> "_DatesBlock":
        # Returns the intersection of two DatesBlocks, i.e common dates
        a = self
        b = other
        if a is None:
            return b
        if b is None:
            return a

        # find smallest multiple frequency
        a_frequency = a.frequency.astype("timedelta64[s]")
        b_frequency = b.frequency.astype("timedelta64[s]")
        frequency = np.lcm(a_frequency.astype(int), b_frequency.astype(int))
        frequency = frequency.astype("timedelta64[s]")
        assert frequency.astype(int) % a_frequency.astype(int) == 0
        assert frequency.astype(int) % b_frequency.astype(int) == 0

        a_start = a.start.astype("datetime64[s]")
        b_start = b.start.astype("datetime64[s]")
        a_freq = a.frequency.astype("timedelta64[s]")
        b_freq = b.frequency.astype("timedelta64[s]")

        # Solve: a_start + k1*a_freq = b_start + k2*b_freq
        a0, b0 = a_start.astype(int), b_start.astype(int)
        fa, fb = a_freq.astype(int), b_freq.astype(int)
        d = b0 - a0
        g = gcd(fa, fb)
        if d % g != 0:
            raise ValueError(f"No common date between {a} and {b}")

        def egcd(x, y):
            if y == 0:
                return 1, 0, x
            u, v, g = egcd(y, x % y)
            return v, u - (x // y) * v, g

        x0, y0, _ = egcd(fa, fb)
        x0 *= d // g
        start = a0 + x0 * fa
        # Ensure start is the first common date ≥ both starts
        lcm = abs(fa * fb) // g
        while start < max(a0, b0):
            start += lcm
        assert start == int(start), start
        start = int(start)
        start = np.datetime64(start, "s")
        assert (a_start - start) % a_frequency == 0
        assert (b_start - start) % b_frequency == 0

        a_end = a.end.astype("datetime64[s]")
        b_end = b.end.astype("datetime64[s]")
        end = min(a_end, b_end)
        # Round down to the last point that aligns with the new frequency
        delta = (end - start).astype("timedelta64[s]").astype(int)
        freq_s = frequency.astype("timedelta64[s]").astype(int)
        end = start + (delta // freq_s) * frequency
        assert (end - start) % frequency == 0

        a_missing = a.missing
        b_missing = b.missing
        a_missing = set(_ for _ in a_missing if (_ - start) % frequency == 0)
        b_missing = set(_ for _ in b_missing if (_ - start) % frequency == 0)
        missing = set(a_missing).union(set(b_missing))
        missing = np.array(sorted(list(missing)), dtype="datetime64[s]")
        missing = missing[missing >= start]
        missing = missing[missing <= end]
        assert isinstance(missing, np.ndarray), type(missing)

        return _DatesBlock(start, end, frequency, missing)

    def __rand__(self, other: "_DatesBlock") -> "_DatesBlock":
        return self.__and__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start}, {self.end}, {self.frequency}, missing={len(self.missing)})"

    def _as_array(self, missing_as_nans=False) -> NDArray[np.datetime64]:
        # Get all dates as an array, excluding missing dates
        total_length = int((self.end - self.start) / self.frequency) + 1
        all_dates = self.start + np.arange(total_length) * self.frequency
        mask = ~np.isin(all_dates, self.missing)
        if missing_as_nans:
            return np.where(mask, all_dates, np.datetime64("NaT"))
        else:
            return all_dates[mask]


class DatesGathererVisitor:
    # Use to finalise the sample_provider by visiting all datasets
    # and computing the overall date range and frequency and missing
    def __init__(self):
        self.dates_block = None

    def read_date_offsets(self, obj: "SampleProvider"):
        self.dates_block = obj.filter_available_dates(self.dates_block)


def find_required_steps_for_rollout(steps: list | int, input: list, target: list, frequency: str = None) -> list[str]:
    # Find all required offsets to cover input and target offsets plus steps
    # This function is not related to the rest of this file, except it uses offsets.

    # turn everything into timedelta
    if frequency is not None:
        freq = frequency_to_timedelta(frequency)

    if isinstance(steps, int):
        steps = list(range(steps))
    if all(isinstance(x, int) for x in steps):
        if frequency is None:
            raise ValueError("If steps is a list of integers, frequency must be provided")
        steps = [freq * s for s in steps]

    input = [offset_to_timedelta(x) for x in input]
    target = [offset_to_timedelta(x) for x in target]

    # find required offsets
    required_offsets = set()
    for step in steps:
        for i in input:
            required_offsets.add(i + step)
        for t in target:
            required_offsets.add(t + step)
    required_offsets = sorted(required_offsets)
    return [offset_to_string(x) for x in required_offsets]
