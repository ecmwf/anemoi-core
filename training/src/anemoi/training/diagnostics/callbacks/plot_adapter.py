# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plot adapter: single entry point for diagnostics callbacks.

Groups the five plot-related hooks so task classes expose one attribute
(plot_adapter) instead of five small methods.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anemoi.training.tasks.base import BaseTask


class BasePlotAdapter(ABC):
    """Abstract plotting contract. Subclasses define output_times, get_init_step, iter_plot_samples."""

    def __init__(self, task: BaseTask) -> None:
        self._task = task

    @property
    def is_ensemble(self) -> bool:
        return False

    def get_loss_plot_batch_start(self, **_kwargs) -> int:
        return 0

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        return output_tensor

    def select_members(self, tensor: Any, members: int | list[int] | None = None) -> Any:  # noqa: ARG002
        """Select ensemble members from tensor. No-op for non-ensemble adapters."""
        return tensor

    def prepare_loss_batch(self, batch: dict) -> dict:
        """Prepare batch for loss plotting. No-op for non-ensemble adapters."""
        return batch

    @abstractmethod
    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        """Yield (x, y_true, y_pred, tag_suffix) or (sample, recon, tag) per plot sample."""
        ...


class ForecasterPlotAdapter(BasePlotAdapter):
    """Rollout forecaster: multiple loss plots, n_step_output targets per step, multi-step iter."""

    def get_init_step(self) -> int:
        return -1

    def get_loss_plot_batch_start(self, rollout_step: int) -> int:
        return self._task.num_input_timesteps + rollout_step * self._task.num_output_timesteps

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        input_time_indices = self._task.get_batch_input_indices()

        input_data = data[input_time_indices, ...]

        x = input_data[self.get_init_step(), ...].squeeze()

        for rollout_step in range(self._task.validation_rollout):
            output_time_indices = self._task.get_batch_output_indices(rollout_step=rollout_step)

            output_data = data[output_time_indices, ...]

            for out_step in range(self._task.num_output_timesteps):
                y_true = output_data[out_step, ...].squeeze()
                y_pred = output_tensor[rollout_step, out_step, ...]
                y_pred = y_pred.squeeze() if hasattr(y_pred, "squeeze") else y_pred
                yield x, y_true, y_pred, f"rstep{rollout_step:02d}_out{out_step:02d}"


class TemporalDownscalerPlotAdapter(BasePlotAdapter):
    """Temporal downscaling: also squeeze (1, n_step_output, ...) -> (n_step_output, ...)."""

    def get_init_step(self) -> int:
        return 0

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        input_time_indices = self._task.get_batch_input_indices()
        output_time_indices = self._task.get_batch_output_indices()

        input_data = data[input_time_indices, ...]
        output_data = data[output_time_indices, ...]

        x = input_data[self.get_init_step(), ...].squeeze()
        for output_step in range(self._task.num_output_timesteps):
            y_true = output_data[output_step].squeeze()
            pred = (
                output_tensor[output_step, 0] if getattr(output_tensor, "ndim", 0) >= 4 else output_tensor[output_step]
            )
            y_pred = pred.squeeze() if hasattr(pred, "squeeze") else pred
            yield x, y_true, y_pred, f"istep{output_step + 1:02d}"

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        if getattr(output_tensor, "ndim", 0) == 5 and getattr(output_tensor, "shape", (0,))[0] == 1:
            return output_tensor.squeeze(0)
        return output_tensor


class AutoencoderPlotAdapter(BasePlotAdapter):
    """Autoencoder: single (sample, recon, tag) yield."""

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        sample = data[0, ...].squeeze()
        recon = output_tensor[0, ...].squeeze()
        yield sample, sample, recon, "recon"


class EnsemblePlotAdapter(BasePlotAdapter):
    """Wraps any task-specific adapter, adding ensemble member handling.

    This adapter decorates an inner (task-specific) adapter to handle the
    extra ensemble dimension present in ensemble training outputs.
    Batch shape convention: (B, T, E, G, V) where E is ensemble members.
    """

    def __init__(self, inner: BasePlotAdapter) -> None:
        self._inner = inner
        self._task = inner._task

    @property
    def is_ensemble(self) -> bool:
        return True

    def get_loss_plot_batch_start(self, **kwargs) -> int:
        return self._inner.get_loss_plot_batch_start(**kwargs)

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        return self._inner.prepare_plot_output_tensor(output_tensor)

    def select_members(self, tensor: Any, members: int | list[int] | None = None) -> Any:
        """Slice ensemble members from dim 2 of the output tensor.

        Parameters
        ----------
        tensor : Any
            Tensor with shape (..., members, grid, vars)
        members : int | list[int] | None
            Members to select. None returns all members, int/list selects specific members.
        """
        if members is None:
            return tensor
        if not isinstance(members, list):
            members = [members]
        return tensor[:, :, members, ...]

    def prepare_loss_batch(self, batch: dict) -> dict:
        """Squeeze ensemble dim to member 0 for loss plotting."""
        return {dataset: data[:, :, 0, :, :] for dataset, data in batch.items()}

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        yield from self._inner.iter_plot_samples(data, output_tensor)
