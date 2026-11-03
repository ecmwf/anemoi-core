# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

from anemoi.utils.dates import frequency_to_string

if TYPE_CHECKING:
    import datetime
    from collections.abc import Mapping

    import numpy as np

    from anemoi.training.data.readers import DataReader
    from anemoi.training.tasks.base import BaseTask


def compute_model_relative_date_indices(
    task: BaseTask,
    timestep: datetime.timedelta,
    *,
    mode: str = "training",
) -> list[int]:
    """Compute task offsets as indices on a shared model time grid."""
    offsets = task.get_offsets(mode=mode)
    if any(offset % timestep for offset in offsets):
        msg = (
            f"Task `{task.__class__.__name__}` defines offsets "
            f"{[frequency_to_string(offset) for offset in offsets]} that are not exact multiples of "
            f"{frequency_to_string(timestep)}."
        )
        raise ValueError(msg)

    return sorted({int(offset // timestep) for offset in offsets})


def compute_relative_date_indices(
    task: BaseTask,
    data_readers: dict[str, DataReader],
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


def resolve_task_relative_indices_by_dataset(
    task: BaseTask,
    dataset_model_relative_indices_by_dataset: Mapping[str, np.ndarray],
    timestep: datetime.timedelta,
    *,
    mode: str,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[list[int]]]]:
    """Split each dataset's exact relative indices into task inputs and targets."""
    input_indices = {int(offset // timestep) for offset in task.get_input_offsets(mode=mode)}
    targets_by_step = [
        [int(offset // timestep) for offset in task.get_output_offsets(**step_kwargs)]
        for step_kwargs in task.steps(mode)
    ]
    target_indices = {value for step in targets_by_step for value in step}

    inputs: dict[str, list[int]] = {}
    targets: dict[str, list[int]] = {}
    step_targets: dict[str, list[list[int]]] = {}
    for name, relative_indices in dataset_model_relative_indices_by_dataset.items():
        available = [int(value) for value in relative_indices.tolist()]
        available_set = set(available)
        inputs[name] = [value for value in available if value in input_indices]
        targets[name] = [value for value in available if value in target_indices]
        step_targets[name] = [[value for value in step if value in available_set] for step in targets_by_step]

    return inputs, targets, step_targets
