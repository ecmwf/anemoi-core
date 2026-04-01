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
    def output_times(self) -> int:
        """Number of rollout/outer steps for plotting."""
        return self._task.num_output_timesteps

    @abstractmethod
    def get_init_step(self, rollout_step: int) -> int:
        """Input step index for a given rollout step."""
        ...

    @property
    def step_names(self) -> list[str]:
        return self._task.step_names

    def get_loss_plot_batch_start(self, rollout_step: int) -> int:
        del rollout_step
        return 0

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        return output_tensor

    @abstractmethod
    def iter_plot_samples(
        self,
        data: Any,
        output_tensor: Any,
        output_times: int,
        max_out_steps: int | None = None,
    ) -> Iterator[tuple[Any, Any, Any, str] | tuple[Any, Any, str]]:
        """Yield (x, y_true, y_pred, tag_suffix) or (sample, recon, tag) per plot sample."""
        ...


class ForecasterPlotAdapter(BasePlotAdapter):
    """Rollout forecaster: multiple loss plots, n_step_output targets per step, multi-step iter."""

    def get_init_step(self, rollout_step: int) -> int:
        del rollout_step
        return 0

    def get_loss_plot_batch_start(self, rollout_step: int) -> int:
        return self._task.num_input_timesteps + rollout_step * self._task.num_output_timesteps

    def iter_plot_samples(
        self,
        data: Any,
        output_tensor: Any,
        output_times: int,
        max_out_steps: int | None = None,
    ) -> Iterator[tuple[Any, Any, Any, str]]:
        task = self._task
        max_out_steps = min(task.num_output_timesteps, max_out_steps or task.num_output_timesteps)
        for rollout_step in range(output_times):
            init_step = self.get_init_step(rollout_step)
            x = data[init_step, ...].squeeze()
            for out_step in range(max_out_steps):
                truth_idx = rollout_step * task.num_output_timesteps + out_step + 1
                y_true = data[truth_idx, ...].squeeze()
                y_pred = output_tensor[rollout_step, out_step, ...]
                y_pred = y_pred.squeeze() if hasattr(y_pred, "squeeze") else y_pred
                yield x, y_true, y_pred, f"rstep{rollout_step:02d}_out{out_step:02d}"


class TemporalDownscalingPlotAdapter(BasePlotAdapter):
    """Temporal downscaling: also squeeze (1, n_step_output, ...) -> (n_step_output, ...)."""

    def get_init_step(self, **_kwargs) -> int:
        return 0

    def iter_plot_samples(
        self,
        data: Any,
        output_tensor: Any,
        output_times: int,
        max_out_steps: int | None = None,
    ) -> Iterator[tuple[Any, Any, Any, str]]:
        del max_out_steps
        for output_step in range(output_times):
            init_step = self.get_init_step(output_step)
            x = data[init_step, ...].squeeze()
            y_true = data[output_step, ...].squeeze()
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

    def get_init_step(self, **_kwargs) -> int:
        return 0

    def iter_plot_samples(
        self,
        data: Any,
        output_tensor: Any,
        output_times: int,
        max_out_steps: int | None = None,
    ) -> Iterator[tuple[Any, Any, str]]:
        del output_times, max_out_steps
        sample = data[0, ...].squeeze()
        recon = output_tensor[0, ...].squeeze()
        yield sample, recon, "recon"
