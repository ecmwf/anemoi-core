"""Task-aligned residual target construction."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Callable

import torch

from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import as_timedelta


def _processors_for_outputs(
    processors: object,
    output_offsets: list[datetime.timedelta],
    *,
    broadcast_single: bool,
) -> list[Callable]:
    """Resolve processors by physical output offset, never over a MIMO tensor."""
    n_outputs = len(output_offsets)
    if isinstance(processors, StepwiseProcessors):
        if len(processors) != n_outputs:
            raise ValueError(f"Expected {n_outputs} residual processors, got {len(processors)}.")
        resolved = list(processors)
    elif isinstance(processors, Mapping):
        resolved = []
        for offset in output_offsets:
            processor = None
            for key, candidate in processors.items():
                if key == offset or str(key) == str(offset):
                    processor = candidate
                    break
                if isinstance(key, str):
                    try:
                        if as_timedelta(key) == offset:
                            processor = candidate
                            break
                    except (TypeError, ValueError):
                        pass
            resolved.append(processor)
    elif isinstance(processors, Sequence) and not callable(processors):
        if len(processors) != n_outputs:
            raise ValueError(f"Expected {n_outputs} residual processors, got {len(processors)}.")
        resolved = list(processors)
    elif callable(processors):
        if n_outputs > 1 and not broadcast_single:
            raise ValueError(
                "A single residual processor cannot be applied to multiple output steps implicitly. "
                "Set broadcast_single=True to explicitly broadcast time-invariant residual statistics."
            )
        resolved = [processors] * n_outputs
    else:
        raise ValueError("Residual processors are required for every output step.")

    if any(processor is None or not callable(processor) for processor in resolved):
        raise ValueError("Residual statistics are missing for one or more output offsets.")
    return resolved


def compute_residual_targets(
    targets: torch.Tensor,
    source: torch.Tensor,
    task: BaseTask,
    residual_processors: object,
    *,
    state_processor: Callable | None = None,
    direct_prediction_indices: Sequence[int] | None = None,
    broadcast_single: bool = False,
) -> torch.Tensor:
    """Compute independently normalized, task-aligned residual targets.

    source may contain the full task input window or already-aligned output
    steps. If direct-prediction channels are supplied, those channels use the
    state processor and all remaining channels use the per-output processor.
    """
    if targets.ndim < 2 or source.ndim < 2:
        raise ValueError("Residual targets and source must have a time dimension.")

    n_outputs = task.num_output_timesteps
    if targets.shape[1] != n_outputs:
        raise ValueError(f"Expected {n_outputs} target steps, got {targets.shape[1]}.")

    reference_indices = task.get_input_reference_positions(strict=True)
    if source.shape[1] == task.num_input_timesteps:
        source = source.index_select(1, torch.as_tensor(reference_indices, device=source.device))
    elif source.shape[1] != n_outputs:
        raise ValueError(
            f"Expected source to contain {task.num_input_timesteps} input steps or {n_outputs} output steps, "
            f"got {source.shape[1]}."
        )

    output_offsets = task.get_output_offsets()
    processors = _processors_for_outputs(
        residual_processors,
        output_offsets,
        broadcast_single=broadcast_single,
    )
    direct = set(direct_prediction_indices or ())
    if direct and state_processor is None:
        raise ValueError("state_processor is required when direct-prediction channels are configured.")

    result = []
    for step, processor in enumerate(processors):
        residual = targets[:, step] - source[:, step]
        if direct:
            step_target = residual.clone()
            residual_indices = [index for index in range(targets.shape[-1]) if index not in direct]
            if residual_indices:
                step_target[..., residual_indices] = processor(
                    residual[..., residual_indices],
                    in_place=False,
                )
            direct_indices = sorted(direct)
            step_target[..., direct_indices] = state_processor(
                targets[:, step, ..., direct_indices],
                in_place=False,
            )
            result.append(step_target.unsqueeze(1))
        else:
            result.append(processor(residual, in_place=False).unsqueeze(1))
    return torch.cat(result, dim=1)
