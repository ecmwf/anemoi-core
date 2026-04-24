# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_string

if TYPE_CHECKING:
    import logging
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


def normalize_explicit_time_indices_config(
    explicit_time_indices_by_dataset: dict[str, dict[str, list[int]]] | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Normalize per-dataset sparse windows to the legacy positive index convention."""
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for dataset_name, dataset_cfg in (explicit_time_indices_by_dataset or {}).items():
        raw_input = dataset_cfg.get("input", None)
        raw_target = dataset_cfg.get("target", None)
        if raw_input is None or raw_target is None:
            msg = f"Explicit time indices for dataset '{dataset_name}' must define both `input` and `target`."
            raise ValueError(
                msg
            )

        input_indices = np.array(sorted({int(value) for value in raw_input}), dtype=np.int64)
        target_indices = np.array(sorted({int(value) for value in raw_target}), dtype=np.int64)
        if len(input_indices) == 0:
            msg = f"Explicit time indices for dataset '{dataset_name}' require a non-empty `input`."
            raise ValueError(
                msg
            )

        combined_indices = (
            np.concatenate([input_indices, target_indices]).astype(np.int64, copy=False)
            if len(target_indices) > 0
            else input_indices
        )
        anchor_index = max(0, -int(combined_indices.min()))
        if anchor_index > 0:
            input_indices = input_indices + anchor_index
            target_indices = target_indices + anchor_index

        normalized[str(dataset_name)] = {
            "input": input_indices,
            "target": target_indices,
        }
    return normalized


def parse_dataset_time_indices_config(config: BaseSchema) -> dict[str, dict[str, list[int]]] | None:
    """Parse optional per-dataset sparse time windows from config."""
    cfg = getattr(getattr(config, "task", None), "dataset_time_indices", None)
    if cfg is None:
        cfg = getattr(getattr(config, "training", None), "dataset_time_indices", None)
    if cfg is None:
        return None

    cfg = cfg.get("datasets", cfg)
    timestep_seconds = frequency_to_seconds(config.data.timestep)
    parsed: dict[str, dict[str, list[int]]] = {}
    for dataset_name, dataset_cfg in cfg.items():
        raw_input = dataset_cfg.get("input", None)
        raw_target = dataset_cfg.get("target", None)
        if raw_input is None or raw_target is None:
            msg = f"`training.dataset_time_indices[{dataset_name}]` must define both `input` and `target`."
            raise ValueError(msg)

        parsed_dataset_cfg: dict[str, list[int]] = {}
        for field_name, raw_values in {"input": raw_input, "target": raw_target}.items():
            parsed_values: list[int] = []
            for raw_value in raw_values:
                if isinstance(raw_value, Integral):
                    parsed_values.append(int(raw_value))
                    continue

                try:
                    if isinstance(raw_value, str) and raw_value.strip().lstrip("+-").isdigit():
                        parsed_values.append(int(raw_value.strip()))
                        continue

                    offset_seconds = frequency_to_seconds(raw_value)
                except (AssertionError, TypeError, ValueError) as exc:
                    msg = (
                        f"`training.dataset_time_indices[{dataset_name}].{field_name}` value {raw_value!r} "
                        "must be either an integer step or a duration like '-5m' or '1h'."
                    )
                    raise ValueError(msg) from exc

                if offset_seconds % timestep_seconds != 0:
                    msg = (
                        f"`training.dataset_time_indices[{dataset_name}].{field_name}` value {raw_value!r} "
                        f"is not an exact multiple of timestep {config.data.timestep!r}."
                    )
                    raise ValueError(msg)
                parsed_values.append(offset_seconds // timestep_seconds)

            parsed_dataset_cfg[field_name] = parsed_values

        if len(parsed_dataset_cfg["input"]) == 0:
            msg = f"`training.dataset_time_indices[{dataset_name}]` requires a non-empty `input` list."
            raise ValueError(
                msg
            )

        parsed[str(dataset_name)] = parsed_dataset_cfg

    normalized = normalize_explicit_time_indices_config(parsed)
    return {
        dataset_name: {
            "input": dataset_cfg["input"].tolist(),
            "target": dataset_cfg["target"].tolist(),
        }
        for dataset_name, dataset_cfg in normalized.items()
    } or None


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
    """Resolve the full model-relative window, including dataset-specific sparse requests."""
    relative_indices = set(
        default_relative_date_indices(
            config,
            task=task,
            mode=mode,
            val_rollout=val_rollout,
            logger=logger,
        ),
    )
    dataset_time_indices = parse_dataset_time_indices_config(config)
    if dataset_time_indices is None:
        return sorted(relative_indices)

    for dataset_cfg in dataset_time_indices.values():
        relative_indices.update(int(value) for value in dataset_cfg["input"])
        relative_indices.update(int(value) for value in dataset_cfg["target"])

    return sorted(relative_indices)
