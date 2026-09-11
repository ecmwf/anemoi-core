# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_string

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from anemoi.training.schemas.base_schema import BaseSchema
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


def _config_get(container: object | None, key: str) -> object | None:
    """Safely read a key from OmegaConf, dict-like, or attribute-based config objects."""
    if container is None:
        return None

    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, None)

    return getattr(container, key, None)


def resolve_config_frequency(
    config: BaseSchema,
    task: BaseTask | None = None,
) -> str | datetime.timedelta:
    """Resolve the shared model-grid frequency used by mixed-frequency time-index parsing."""
    if task is not None:
        timestep = getattr(task, "timestep", None)
        if timestep is not None:
            return timestep

    task_cfg = _config_get(config, "task")
    candidate = _config_get(task_cfg, "timestep")
    if candidate is not None:
        return str(candidate)

    data_cfg = _config_get(config, "data")
    candidate = _config_get(data_cfg, "frequency")
    if candidate is not None:
        return str(candidate)

    msg = "Could not determine shared model frequency from `task.timestep` or `data.frequency`."
    raise ValueError(msg)


def resolve_task_input_relative_indices(
    task: BaseTask,
    timestep: datetime.timedelta,
    *,
    mode: str,
) -> list[int]:
    """Return task input offsets as indices on the shared model grid."""
    return [int(offset // timestep) for offset in task.get_input_offsets(mode=mode)]


def resolve_task_target_relative_indices(
    task: BaseTask,
    timestep: datetime.timedelta,
    *,
    mode: str,
) -> list[int]:
    """Return task output offsets as indices on the shared model grid."""
    target_relative_indices: set[int] = set()
    for step_kwargs in task.steps(mode):
        target_relative_indices.update(
            int(offset // timestep) for offset in task.get_output_offsets(**step_kwargs)
        )
    return sorted(target_relative_indices)


def resolve_task_target_relative_indices_by_step(
    task: BaseTask,
    timestep: datetime.timedelta,
    *,
    mode: str,
) -> list[list[int]]:
    """Return task output indices separately for each rollout step."""
    return [
        [int(offset // timestep) for offset in task.get_output_offsets(**step_kwargs)]
        for step_kwargs in task.steps(mode)
    ]


def resolve_task_relative_indices_by_dataset(
    task: BaseTask,
    dataset_model_relative_indices_by_dataset: Mapping[str, np.ndarray],
    timestep: datetime.timedelta,
    *,
    mode: str,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[list[int]]]]:
    """Split each dataset's exact relative indices into task inputs and targets."""
    task_input_relative_indices = set(resolve_task_input_relative_indices(task, timestep, mode=mode))
    task_target_relative_indices = set(resolve_task_target_relative_indices(task, timestep, mode=mode))
    task_target_relative_indices_by_step = resolve_task_target_relative_indices_by_step(task, timestep, mode=mode)

    input_relative_indices_by_dataset: dict[str, list[int]] = {}
    target_relative_indices_by_dataset: dict[str, list[int]] = {}
    target_relative_indices_by_dataset_by_step: dict[str, list[list[int]]] = {}
    for dataset_name, relative_indices in dataset_model_relative_indices_by_dataset.items():
        dataset_relative_indices = [int(value) for value in relative_indices.tolist()]
        dataset_relative_indices_set = set(dataset_relative_indices)
        input_relative_indices_by_dataset[dataset_name] = [
            value for value in dataset_relative_indices if value in task_input_relative_indices
        ]
        target_relative_indices_by_dataset[dataset_name] = [
            value for value in dataset_relative_indices if value in task_target_relative_indices
        ]
        target_relative_indices_by_dataset_by_step[dataset_name] = [
            [value for value in step_indices if value in dataset_relative_indices_set]
            for step_indices in task_target_relative_indices_by_step
        ]

    return (
        input_relative_indices_by_dataset,
        target_relative_indices_by_dataset,
        target_relative_indices_by_dataset_by_step,
    )


def default_relative_date_indices(
    config: BaseSchema,
    task: BaseTask,
    mode: str = "training",
) -> list[int]:
    """Build the model-relative window from the task's current offset API."""
    frequency = resolve_config_frequency(config, task=task)
    timestep = datetime.timedelta(seconds=frequency_to_seconds(frequency))
    return compute_model_relative_date_indices(task, timestep, mode=mode)


def resolve_relative_date_indices(
    config: BaseSchema,
    task: BaseTask,
    mode: str = "training",
) -> list[int]:
    """Resolve the shared model-relative window for mixed-frequency alignment."""
    return default_relative_date_indices(
        config,
        task=task,
        mode=mode,
    )
