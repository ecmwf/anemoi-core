# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

LOGGER = logging.getLogger(__name__)


def get_balanced_partition_sizes(
    total_size: int,
    n_partitions: int,
) -> list[int]:
    """Partition the total size into n balanced parts, i.e. differing in size by at most 1.

    Parameters
    ----------
    total_size : int
        Total size to partition
    n_partitions : int
        Number of partitions

    Returns
    -------
    list[int]
        List of partition sizes

    """
    base_size = total_size // n_partitions
    remainder = total_size % n_partitions

    # distribute the remainder across the first #remainder parts
    return [base_size + 1] * remainder + [base_size] * (n_partitions - remainder)


def get_partition_range(
    partition_sizes: list[int],
    partition_id: int,
    offset: int = 0,
) -> tuple[int, int]:
    """Get the range for a specific partition.

    Parameters
    ----------
    partition_sizes : list[int]
        List of partition sizes
    partition_id : int
        Partition ID
    offset : int, optional
        Offset to add to the start and end indices, by default 0

    Returns
    -------
    tuple[int, int]
        Start and end indices for the specified partition

    """
    if partition_id < 0 or partition_id >= len(partition_sizes):
        msg = f"Invalid partition ID {partition_id}, expected in [0, {len(partition_sizes)})"
        raise ValueError(msg)

    start = sum(partition_sizes[:partition_id]) + offset
    end = start + partition_sizes[partition_id]

    return start, end


def get_balanced_partition_range(
    total_size: int,
    n_partitions: int,
    partition_id: int,
    offset: int = 0,
) -> tuple[int, int]:
    """Get the range for a balanced partition.

    Parameters
    ----------
    total_size : int
        Total size to partition
    n_partitions : int
        Number of partitions
    partition_id : int
        Partition ID
    offset : int, optional
        Offset to add to the start and end indices, by default 0

    Returns
    -------
    tuple[int, int]
        Start and end indices for the specified partition

    """
    partition_sizes = get_balanced_partition_sizes(total_size, n_partitions)
    return get_partition_range(partition_sizes, partition_id, offset)
