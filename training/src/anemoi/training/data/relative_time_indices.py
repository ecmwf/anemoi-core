# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string


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
