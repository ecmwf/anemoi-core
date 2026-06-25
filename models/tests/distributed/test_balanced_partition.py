# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range


@pytest.mark.parametrize(
    "total_size,n_partitions,expected",
    [
        (10, 3, [4, 3, 3]),  # remainder
        (12, 4, [3, 3, 3, 3]),  # even split
        (3, 5, [1, 1, 1, 0, 0]),  # total < partitions
        (0, 4, [0, 0, 0, 0]),  # zero total
    ],
)
def test_get_balanced_partition_sizes(
    total_size: int,
    n_partitions: int,
    expected: list[int],
) -> None:
    sizes = get_balanced_partition_sizes(total_size, n_partitions)

    assert sizes == expected
    assert len(sizes) == n_partitions
    assert sum(sizes) == total_size
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(
    "partition_sizes,offset",
    [
        ([4, 3, 3], 0),
        ([1, 1, 1, 1], 10),
        ([2, 0, 3], 5),
    ],
)
def test_get_partition_range(
    partition_sizes: list[int],
    offset: int,
) -> None:
    starts = []
    ends = []

    for pid in range(len(partition_sizes)):
        start, end = get_partition_range(partition_sizes, pid, offset)
        starts.append(start)
        ends.append(end)

        assert end - start == partition_sizes[pid]

    # contiguity
    for i in range(1, len(starts)):
        assert starts[i] == ends[i - 1]

    # offset applied correctly
    assert starts[0] == offset


@pytest.mark.parametrize(
    "total_size,n_partitions,partition_id,offset,expected_start,expected_end",
    [
        (10, 4, 0, 0, 0, 3),
        (10, 4, 1, 0, 3, 6),
        (10, 4, 2, 0, 6, 8),
        (10, 4, 3, 0, 8, 10),
        (12, 3, 1, 5, 9, 13),
        (10, 4, 0, 10, 10, 13),
    ],
)
def test_get_balanced_partition_range(
    total_size: int,
    n_partitions: int,
    partition_id: int,
    offset: int,
    expected_start: int,
    expected_end: int,
) -> None:
    start, end = get_balanced_partition_range(total_size, n_partitions, partition_id, offset)

    assert start == expected_start
    assert end == expected_end
    assert end - start == get_balanced_partition_sizes(total_size, n_partitions)[partition_id]


def test_get_partition_range_invalid_partition_id() -> None:
    with pytest.raises(ValueError, match="Invalid partition ID"):
        get_partition_range([3, 3, 4], partition_id=3)
