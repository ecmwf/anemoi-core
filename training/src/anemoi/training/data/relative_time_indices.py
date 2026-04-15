# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Sequence
from itertools import pairwise
import numpy as np

from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string

TimeIndices = slice | int | list[int] | np.ndarray


def normalize_time_indices(time_indices: TimeIndices) -> TimeIndices:
    """Collapse contiguous integer sequences into slices when possible.

    Using a list/array of integers triggers advanced indexing in NumPy and
    PyTorch, while slices use the cheaper basic-indexing path. We preserve
    sparse selections as explicit indices.
    """
    if isinstance(time_indices, (slice, int)):
        return time_indices

    if isinstance(time_indices, np.ndarray):
        if time_indices.ndim != 1:
            return time_indices
        indices = time_indices.tolist()
    elif isinstance(time_indices, Sequence):
        indices = list(time_indices)
    else:
        return time_indices

    if not indices:
        return indices

    if len(indices) == 1:
        start = indices[0]
        return slice(start, start + 1, 1)

    step = indices[1] - indices[0]
    if step <= 0:
        return indices

    if any(curr - prev != step for prev, curr in pairwise(indices)):
        return indices

    return slice(indices[0], indices[-1] + step, step)


def offset_time_indices(reference_index: int, relative_indices: TimeIndices) -> TimeIndices:
    """Shift relative time indices by a sample reference index."""
    relative_indices = normalize_time_indices(relative_indices)

    if isinstance(relative_indices, slice):
        start = None if relative_indices.start is None else reference_index + relative_indices.start
        stop = None if relative_indices.stop is None else reference_index + relative_indices.stop
        return slice(start, stop, relative_indices.step)

    if isinstance(relative_indices, int):
        return reference_index + relative_indices

    if isinstance(relative_indices, np.ndarray):
        return reference_index + relative_indices

    return [reference_index + offset for offset in relative_indices]


def compute_relative_date_indices(
    task: BaseTask,
    data_readers: dict,
    **kwargs,
) -> dict[str, list[int]]:
    """Compute relative date indices for each dataset based on task offsets."""
    offsets = task.get_offsets(**kwargs)

    relative_date_indices = {}
    for name, dr in data_readers.items():
        if any(o % dr.frequency for o in offsets):
            msg = (
                f"The frequency of `{name}` ({frequency_to_string(dr.frequency)}) is not compatible "
                f"with the task defined offsets ({[frequency_to_string(o) for o in offsets]}). "
                f"Check that the task offsets are compatible with the dataset frequency."
            )
            raise ValueError(msg)
        relative_date_indices[name] = [o // dr.frequency for o in offsets]

    return relative_date_indices
