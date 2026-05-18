# (C) Copyright 2024 Anemoi contributors.
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
    import logging
    from collections.abc import Mapping

    import numpy as np

    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.tasks.base import BaseTask


def compute_model_relative_date_indices(
    task: BaseTask,
    *,
    mode: str = "training",
) -> list[int] | None:
    """Compute model-relative indices from a task's offsets when it exposes a single reference timestep."""
    timestep = getattr(task, "timestep", None)
    if timestep is None:
        return None

    offsets = task.get_offsets(mode=mode)
    if any(offset % timestep for offset in offsets):
        msg = (
            f"Task `{task.__class__.__name__}` defines offsets "
            f"{[frequency_to_string(offset) for offset in offsets]} that are not exact multiples of "
            f"{frequency_to_string(timestep)}."
        )
        raise ValueError(msg)

    relative_indices = [int(offset // timestep) for offset in offsets]
    if len(relative_indices) == 0:
        return []

    anchor_index = max(0, -min(relative_indices))
    return sorted({index + anchor_index for index in relative_indices})


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


def resolve_task_input_relative_indices(task: BaseTask) -> list[int]:
    """Return the task input positions on the shared model grid."""
    return [int(value) for value in task.get_batch_input_indices()]


def resolve_task_target_relative_indices(
    task: BaseTask,
    *,
    mode: str,
) -> list[int]:
    """Return the union of task output positions across the rollout window for one mode."""
    target_relative_indices: set[int] = set()
    for step_kwargs in task.steps(mode):
        target_relative_indices.update(int(value) for value in task.get_batch_output_indices(**step_kwargs))
    return sorted(target_relative_indices)


def resolve_task_relative_indices_by_dataset(
    task: BaseTask,
    dataset_model_relative_indices_by_dataset: Mapping[str, np.ndarray],
    *,
    mode: str,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Split each dataset's exact relative indices into task inputs and targets."""
    task_input_relative_indices = set(resolve_task_input_relative_indices(task))
    task_target_relative_indices = set(resolve_task_target_relative_indices(task, mode=mode))

    input_relative_indices_by_dataset: dict[str, list[int]] = {}
    target_relative_indices_by_dataset: dict[str, list[int]] = {}
    for dataset_name, relative_indices in dataset_model_relative_indices_by_dataset.items():
        dataset_relative_indices = [int(value) for value in relative_indices.tolist()]
        input_relative_indices_by_dataset[dataset_name] = [
            value for value in dataset_relative_indices if value in task_input_relative_indices
        ]
        target_relative_indices_by_dataset[dataset_name] = [
            value for value in dataset_relative_indices if value in task_target_relative_indices
        ]

    return input_relative_indices_by_dataset, target_relative_indices_by_dataset


def default_relative_date_indices(
    config: BaseSchema,
    task: BaseTask | None = None,
    mode: str = "training",
    val_rollout: int = 1,
    logger: logging.Logger | None = None,
) -> list[int]:
    """Build the default model-relative window from explicit-times or rollout config."""
    if task is not None:
        task_relative_indices = compute_model_relative_date_indices(task, mode=mode)
        if task_relative_indices is not None:
            return task_relative_indices

    explicit_times = getattr(getattr(config, "task", None), "explicit_times", None)
    if explicit_times is None:
        explicit_times = getattr(getattr(config, "training", None), "explicit_times", None)
    if explicit_times is not None:
        input_times = [int(value) for value in explicit_times.input]
        if len(input_times) == 0:
            msg = "`task.explicit_times.input` cannot be empty."
            raise ValueError(msg)

        target_offsets = [int(value) for value in explicit_times.target]
        anchor = max(input_times)
        target_times = [anchor + offset for offset in target_offsets]
        return sorted(set(input_times + target_times))

    task_cfg = getattr(config, "task", None)
    multistep_input = getattr(task_cfg, "multistep_input", None)
    multistep_output = getattr(task_cfg, "multistep_output", None)
    rollout_cfg = getattr(task_cfg, "rollout", None)

    if multistep_input is None or multistep_output is None:
        training_cfg = getattr(config, "training", None)
        multistep_input = getattr(training_cfg, "multistep_input", None)
        multistep_output = getattr(training_cfg, "multistep_output", None)
        rollout_cfg = getattr(training_cfg, "rollout", rollout_cfg)

    if multistep_input is None or multistep_output is None:
        msg = "Could not determine `multistep_input`/`multistep_output` from `config.task` or `config.training`."
        raise ValueError(msg)

    rollout_max = getattr(rollout_cfg, "maximum", getattr(rollout_cfg, "max", None))
    rollout_start = getattr(rollout_cfg, "start", 1)
    rollout_epoch_increment = getattr(rollout_cfg, "epoch_increment", 0)

    rollout_value = rollout_start
    if rollout_cfg is not None and rollout_epoch_increment > 0 and rollout_max is not None:
        rollout_value = rollout_max
    elif logger is not None:
        logger.warning("Falling back rollout to: %s", rollout_value)

    rollout = max(rollout_value, val_rollout)
    time_range = multistep_input + rollout * multistep_output
    return list(range(time_range))


def resolve_relative_date_indices(
    config: BaseSchema,
    task: BaseTask | None = None,
    mode: str = "training",
    val_rollout: int = 1,
    logger: logging.Logger | None = None,
) -> list[int]:
    """Resolve the shared model-relative window for mixed-frequency alignment."""
    return default_relative_date_indices(
        config,
        task=task,
        mode=mode,
        val_rollout=val_rollout,
        logger=logger,
    )
