# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import re

import numpy as np

from anemoi.models.data_structure.offsets import _DatesBlock
from anemoi.models.data_structure.offsets import merge_dates_blocks_with_offset
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

# import logging
#
# logging.basicConfig(level="DEBUG")


def dates_block(name, missing_indices=None):
    start, end, freq = re.match(r"(\S+)-to-(\S+)-frequency-(\S+)", name).groups()
    start_dt = datetime.datetime.fromisoformat(start)
    end_dt = datetime.datetime.fromisoformat(end)
    return _DatesBlock(start=start_dt, end=end_dt, frequency=freq, missing_indices=missing_indices)


def _(d, frequency="1d"):
    freq = frequency_to_timedelta(frequency)

    first = datetime.datetime(2001, 1, 1)
    missing_dates = set(d.missing.astype(datetime.datetime))
    present_dates = set(d._as_array(missing_as_nans=False).astype(datetime.datetime))

    res = []
    for date in [first + i * freq for i in range(1000)]:
        if date > d.end:
            break
        if date in missing_dates:
            res.append("__")
        elif date in present_dates:
            res.append(date.strftime("%d"))
        else:
            res.append("  ")

    return "(" + frequency_to_string(d.frequency.astype(datetime.timedelta)) + ") " + " ".join(res)


def test_dates_block():
    a = dates_block("2001-01-01-00:00:00-to-2001-01-06-00:00:00-frequency-24h", missing_indices=[2, 5])
    print("a", _(a))
    assert len(a) == 4, len(a)
    assert np.all(a.missing_indices == [2, 5]), a.missing_indices

    b = dates_block("2001-01-03-00:00:00-to-2001-01-09-00:00:00-frequency-24h", missing_indices=[3, 4])
    print("b", _(b))
    assert len(b) == 5, len(b)
    assert np.all(b.missing_indices == [3, 4]), b.missing_indices

    c = a & b
    print("c", _(c))
    assert len(c) == 2, len(c)
    assert np.all(c.missing_indices == [0, 3]), c.missing_indices


def test_add_offset():
    a = dates_block("2001-01-01-00:00:00-to-2001-01-10-00:00:00-frequency-24h", missing_indices=[2, 5])
    b = a + "24h"
    print("a", _(a))
    print("b", _(b))
    assert len(a) == 8, len(a)
    assert len(b) == 7, (len(a), len(b))
    assert np.all(a.missing_indices == [2, 5]), a.missing_indices
    assert np.all(b.missing_indices == [2, 5]), (a.missing_indices, b.missing_indices)


def test_add_offset_2():
    a = dates_block("2001-01-01-00:00:00-to-2001-01-10-00:00:00-frequency-48h", missing_indices=[2])
    b = a + "24h"
    print("a", _(a))
    print("b", _(b))
    assert len(a) == 4, len(a)
    assert len(b) == 4, (len(a), len(b))
    assert np.all(a.missing_indices == [2]), a.missing_indices
    assert np.all(b.missing_indices == [2]), (a.missing_indices, b.missing_indices)


def test_add_offset_3():
    a = dates_block("2001-01-01-00:00:00-to-2001-01-10-00:00:00-frequency-72h", missing_indices=[2])
    b = a + "48h"
    print("a", _(a))
    print("b", _(b))
    assert len(a) == 3, len(a)
    assert len(b) == 2, (len(a), len(b))
    assert np.all(a.missing_indices == [2]), a.missing_indices
    assert np.all(b.missing_indices == [2]), (a.missing_indices, b.missing_indices)


def container(name, missing_indices, offset):
    class Container:
        def __init__(self, name):
            self._dates_block_in_dataset = dates_block(name, missing_indices=missing_indices)
            self._offset = offset
            self.variables = []

        def __repr__(self):
            return _(self._dates_block_in_dataset) + f"  Offset={self._offset}"

    return Container(name)


def _test_update_dates_block(a, b):
    print()
    current = None
    current = merge_dates_blocks_with_offset(current, a)
    print("a      :", a)
    print("after a:", _(current))
    current = merge_dates_blocks_with_offset(current, b)
    print("b      :", b)
    print("after b:", _(current))
    return current


def test_update_dates_block_1():
    c = _test_update_dates_block(
        container("2001-01-01-00:00:00-to-2001-01-10-00:00:00-frequency-24h", missing_indices=[2], offset="0h"),
        container("2001-01-03-00:00:00-to-2001-01-12-00:00:00-frequency-24h", missing_indices=[3, 4], offset="48h"),
    )
    assert len(c) == 4, len(c)
    assert np.all(c.missing_indices == [3, 4]), c.missing_indices


def test_update_dates_block_2():
    c = _test_update_dates_block(
        container("2001-01-01-00:00:00-to-2001-01-10-00:00:00-frequency-24h", missing_indices=[2], offset="0h"),
        container("2001-01-03-00:00:00-to-2001-01-12-00:00:00-frequency-24h", missing_indices=[3], offset="0h"),
    )
    assert len(c) == 6, len(c)
    assert np.all(c.missing_indices == [0, 3]), c.missing_indices


def test_update_dates_block_3():
    c = _test_update_dates_block(
        container("2001-01-01-00:00:00-to-2001-01-10-00:00:00-frequency-48h", missing_indices=[2], offset="0h"),
        container("2001-01-03-00:00:00-to-2001-01-12-00:00:00-frequency-24h", missing_indices=[3], offset="0h"),
    )
    assert len(c) == 6, len(c)
    assert np.all(c.missing_indices == [2, 3]), c.missing_indices


def test_update_dates_block_4():
    c = _test_update_dates_block(
        container("2001-01-02-00:00:00-to-2001-01-11-00:00:00-frequency-48h", missing_indices=[2], offset="0h"),
        container("2001-01-03-00:00:00-to-2001-01-12-00:00:00-frequency-72h", missing_indices=[1], offset="0h"),
    )
    assert len(c) == 6, len(c)
    assert np.all(c.missing_indices == [2, 3]), c.missing_indices


if __name__ == "__main__":
    test_dates_block()
    test_add_offset()
    test_add_offset_2()
    test_add_offset_3()

    test_update_dates_block_1()
    test_update_dates_block_2()
    test_update_dates_block_3()
    test_update_dates_block_4()
