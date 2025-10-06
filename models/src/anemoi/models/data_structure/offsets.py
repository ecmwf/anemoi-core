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

import numpy as np

from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)

# Utility functions to handle offsets


def offset_to_string(x):
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


def offset_to_timedelta(x):
    if isinstance(x, str) and x.startswith("m"):
        x = "-" + x[1:]
    return frequency_to_timedelta(x)


def normalise_offset(x):
    return offset_to_string(offset_to_timedelta(x))


def sum_offsets(a, b):
    a = offset_to_timedelta(a)
    b = offset_to_timedelta(b)
    x = a + b
    return offset_to_string(x)


class DatesBlock:
    def __init__(self, start, end, frequency, missing_indices):
        self.start = self.normalise_date(start)
        self.end = self.normalise_date(end)
        self.frequency = self.normalise_offset(frequency)
        # must be last because it uses start and end and frequency
        self.missing = self._missing_as_dates(missing_indices)

    def missing_indices(self):
        return self._missing_as_indices(self.missing)

    def __repr__(self):
        v = {
            "start": self.start,
            "end": self.end,
            "frequency": self.frequency,
            "missing": len(self.missing),
        }
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in v.items())})"

    def normalise_date(self, v):
        if isinstance(v, np.datetime64):
            # Ensure the datetime64 is in seconds precision
            return v.astype("datetime64[s]")
        if isinstance(v, datetime.datetime):
            return np.datetime64(v, "s")
        # if isinstance(v, str):
        #    return np.datetime64(v)
        raise ValueError(f"Cannot normalise date {v} of type {type(v)}")

    def normalise_offset(self, v):
        if isinstance(v, np.timedelta64):
            return v.astype("timedelta64[s]")
        v = offset_to_timedelta(v)
        v = np.timedelta64(v, "s")
        return v

    def _missing_as_dates(self, missing_indices: np.ndarray) -> np.ndarray:
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
    def __init__(self):
        self.dates_block = None

    def read_date_offsets(self, obj):
        in_dataset = obj._dates_block
        if in_dataset is None:
            return
        assert isinstance(in_dataset, DatesBlock), f"Expected DatesBlock, got {type(in_dataset)}: {in_dataset}"

        offset = in_dataset.normalise_offset(obj._offset)

        def recompute_missing(missing, start, end):
            _missing = [m for m in missing if start <= m <= end]
            return np.array(_missing, dtype="datetime64[s]")

        _start = max(in_dataset.start + offset, in_dataset.start)
        _end = min(in_dataset.end + offset, in_dataset.end)
        _missing = recompute_missing(in_dataset.missing + offset, _start, _end)
        _frequency = in_dataset.frequency
        in_container = DatesBlock(_start, _end, _frequency, _missing)

        if self.dates_block is None:
            LOGGER.debug(f"Found sample dates_block: {in_container}")
            self.dates_block = in_container
            return

        start = max(self.dates_block.start, in_container.start)
        end = min(self.dates_block.end, in_container.end)
        frequency = self.dates_block.frequency
        missing = np.array(list(set(self.dates_block.missing).union(set(in_container.missing))), dtype="datetime64[s]")
        missing = recompute_missing(self.dates_block.missing, start, end)

        self.dates_block = DatesBlock(start, end, frequency, missing)

        LOGGER.debug(
            f"Updated sample dates_block: {self.dates_block} from {in_container} ({obj.variables} {obj._offset})"
        )

    def write_index_offsets(self, obj):
        #  Using upper case for sample dates and lower case for dataset dates:
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

        in_dataset = obj._dates_block
        if in_dataset is None:
            return
        assert isinstance(in_dataset, DatesBlock), f"Expected DatesBlock, got {type(in_dataset)}: {in_dataset}"

        offset = self.dates_block.normalise_offset(obj._offset)

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
        # LOGGER.debug(f"   in container: {in_container}")
        LOGGER.debug(f"   start difference (S - s + offset): {self.dates_block.start} - {in_dataset.start} + {offset}")
        LOGGER.debug(f"                                    : {self.dates_block.start - in_dataset.start + offset}")
        LOGGER.debug(f"   offset / f: {offset} / {in_dataset.frequency} = {offset / in_dataset.frequency}")
        LOGGER.debug(f"   => add_to_i: {add_to_i}, multiply_i: {multiply_i}")
        obj.set_request(add_to_i=add_to_i, multiply_i=multiply_i)
