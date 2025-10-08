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
    # missing_indices() can be used to get the list of indices

    def __init__(
        self,
        start: datetime.datetime | np.datetime64,
        end: datetime.datetime | np.datetime64,
        frequency: str | np.timedelta64,
        missing_indices: NDArray[np.int64] | list[int],
    ):
        self.start: np.datetime64 = normalise_date(start)
        self.end: np.datetime64 = normalise_date(end)
        self.frequency: np.timedelta64 = offset_to_np_timedelta(frequency)

        missing = self._missing_as_dates(missing_indices)
        self.missing: NDArray[np.datetime64] = missing[self.start <= missing <= self.end]
        assert self.missing.dtype == "datetime64[s]", self.missing.dtype

    def missing_indices(self):
        # Get the indices of the missing dates
        return self._missing_as_indices(self.missing)

    def __len__(self):
        return int((self.end - self.start) / self.frequency) + 1

    def __repr__(self):
        v = {
            "start": self.start,
            "end": self.end,
            "frequency": self.frequency,
            "missing": len(self.missing),
        }
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in v.items())})"

    def _missing_as_dates(self, missing_indices: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(missing_indices, np.ndarray) and missing_indices.dtype == int:
            return self.start + missing_indices * self.frequency

        missing_dates = []
        for i in sorted(missing_indices):
            missing_dates.append(self.start + i * self.frequency)
        return np.array(missing_dates, dtype="datetime64[s]")

    def _missing_as_indices(self, missing_dates: np.ndarray) -> np.ndarray:
        if isinstance(missing_dates, np.ndarray) and missing_dates.dtype == "datetime64[s]":
            return missing_dates
        missing_indices = []
        for d in sorted(missing_dates):
            delta = d - self.start
            if delta % self.frequency != 0:
                raise ValueError(
                    f"Missing date {d} is not aligned with start {self.start} and frequency {self.frequency}"
                )
            i = delta // self.frequency
            missing_indices.append(int(i))
        return np.array(missing_indices, dtype=int)


class OffsetManagerVisitor:
    # Use to finalise the sample_provider by visiting all datasets
    # and computing the overall date range and frequency
    # and then updating each dataset with the required index offsets
    def __init__(self):
        self.dates_block = None

    def read_date_offsets(self, obj: "SampleProvider"):
        # Update the overall self.dates_block from the object's offset and its dataset's dates_block
        #

        in_dataset = obj._dates_block_in_dataset()
        if in_dataset is None:
            return

        offset = offset_to_np_timedelta(obj._offset)

        in_container = _DatesBlock(
            max(in_dataset.start + offset, in_dataset.start),
            min(in_dataset.end + offset, in_dataset.end),
            frequency=in_dataset.frequency,
            missing_indices=in_dataset.missing + offset,
        )

        if self.dates_block is None:
            LOGGER.debug(f"Found sample dates_block: {in_container}")
            self.dates_block = in_container
            return

        start = max(self.dates_block.start, in_container.start)
        end = min(self.dates_block.end, in_container.end)
        frequency = self.dates_block.frequency
        missing = np.array(list(set(self.dates_block.missing).union(set(in_container.missing))), dtype="datetime64[s]")
        missing = self.dates_block.missing

        self.dates_block = _DatesBlock(start, end, frequency, missing)

        LOGGER.debug(
            f"Updated sample dates_block: {self.dates_block} from {in_container} ({obj.variables} {obj._offset})"
        )

    def write_index_offsets(self, obj: "SampleProvider"):
        # Update the object's index offset from the overall self.dates_block and its own offset
        #
        #  Using upper case for sample dates and lower case for dataset dates:
        #
        #  (1) D = S + i . F         (D sample date, S sample start date, i sample index, F sample frequency)
        #  (2) d = s + j . f         (d dataset date, s dataset start date, j dataset index, f dataset frequency)
        #  (3) d = D + offset        required data is at date d (different from sample date D)
        #
        #  => j . f = d - s                         (2)
        #  => j . f = (D + offset) - s              (substituting d from (3))
        #  => j . f = (S + i . F + offset) - s      (substituting D from (1))
        #  => j . f = (S - s + offset) + i . F
        #  => j = (S - s + offset) / f + i . (F / f)
        #  => j = add_to_i + i . multiply_i
        #
        # add_to_i = (S - s + offset) / f
        # multiply_i = F / f

        in_dataset = obj._dates_block_in_dataset()
        if in_dataset is None:
            return

        offset = offset_to_np_timedelta(obj._offset)

        add_to_i = (self.dates_block.start - in_dataset.start + offset) / in_dataset.frequency
        multiply_i = self.dates_block.frequency / in_dataset.frequency

        assert int(add_to_i) == add_to_i, add_to_i
        assert int(multiply_i) == multiply_i, multiply_i
        add_to_i = int(add_to_i)
        multiply_i = int(multiply_i)

        LOGGER.debug(f"Setting dynamic request for container ({obj.variables}, {obj._offset}):")
        LOGGER.debug(f"   offset  : {offset}")
        LOGGER.debug(f"   sample dates: {self.dates_block}")
        LOGGER.debug(f"   in dataset  : {in_dataset}")
        LOGGER.debug(f"   start difference (S - s + offset): {self.dates_block.start} - {in_dataset.start} + {offset}")
        LOGGER.debug(f"                                    : {self.dates_block.start - in_dataset.start + offset}")
        LOGGER.debug(f"   offset / f: {offset} / {in_dataset.frequency} = {offset / in_dataset.frequency}")
        LOGGER.debug(f"   => add_to_i: {add_to_i}, multiply_i: {multiply_i}")

        # update the dataset dates_block to the overall one
        # and provide the index offset and multiplier factor
        obj.finalise(add_to_i=add_to_i, multiply_i=multiply_i, date_block=self.dates_block)


def find_required_steps_for_rollout(steps: list | int, input: list, target: list, frequency: str = None) -> list[str]:
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
