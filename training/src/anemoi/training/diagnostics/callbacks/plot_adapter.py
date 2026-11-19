# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plot adapter: single entry point for diagnostics callbacks.

Groups the plot-related hooks so task classes expose one attribute
(plot_adapter) instead of five small methods.

The EnsemblePlotAdapterWrapper allows to wrap any task-specific adapter,
adding ensemble member handling without modifying the inner adapter's logic.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from anemoi.models.data.sources.base import Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anemoi.training.tasks.base import BaseTask


@dataclass(frozen=True)
class PlotStep:
    """The times shown in one sample figure.

    Attributes
    ----------
    input_index : int
        Position of the input time on the batch time axis.
    target_index : int
        Position of the target time on the batch time axis.
    prediction_step : int
        Which entry of the step outputs holds the prediction (the rollout step for forecasters).
    output_step : int
        Which output time within that entry is the prediction.
    tag : str
        Suffix naming the figure, e.g. ``rstep00_out00`` or ``istep01``.
    """

    input_index: int
    target_index: int
    prediction_step: int
    output_step: int
    tag: str


class BasePlotAdapter(ABC):
    """Abstract plotting contract. Subclasses define plot_steps and iter_plot_samples."""

    def __init__(self, task: BaseTask) -> None:
        self._task = task

    @property
    def is_ensemble(self) -> bool:
        return False

    @staticmethod
    def select_time(data: Any, time_indices: Any) -> Any:
        """Select timesteps along the logical time axis.

        Accepts either a Source or a plain array/tensor
        whose leading axis is the time axis.
        """
        if isinstance(data, Source):
            return data.select_time(time_indices).values
        return data[time_indices, ...]

    @property
    def default_plot_members(self) -> int | list[int] | None:
        """Default ``members`` selection for plot callbacks that don't request a specific subset.

        ``0`` (first member / deterministic view) for non-ensemble adapters;
        overridden by :class:`EnsemblePlotAdapterWrapper` to select all members.
        """
        return 0

    def get_loss_plot_batch_start(self, **_kwargs) -> int:
        return 0

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        return output_tensor

    def select_members(
        self,
        tensor: Any,
        members: int | list[int] | None = None,  # noqa: ARG002
        ensemble_axis: int | None = None,  # noqa: ARG002
    ) -> Any:
        """Select ensemble members from tensor. No-op for non-ensemble adapters."""
        return tensor

    def prepare_loss_batch(self, batch: dict) -> dict:
        """Prepare batch for loss plotting. No-op for non-ensemble adapters."""
        return batch

    @abstractmethod
    def plot_steps(self) -> list[PlotStep]:
        """Return the times shown in each sample figure, one entry per figure.

        Gridded and tabular (observation) plots both follow this list, so a task
        shows the same input and target times for every dataset.
        """
        ...

    @abstractmethod
    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        """Yield (x, y_true, y_pred, tag_suffix) or (sample, recon, tag) per plot sample."""
        ...


class ForecasterPlotAdapter(BasePlotAdapter):
    """Plot Adapter to adapt plots to the rollout set-up of the Forecaster Task.

    Handles multiple loss plots, n_step_output targets per step, multi-step iter.
    """

    def get_loss_plot_batch_start(self, rollout_step: int) -> int:
        return self._task.num_input_timesteps + rollout_step * self._task.num_output_timesteps

    def plot_steps(self) -> list[PlotStep]:
        """Show the latest input next to every output of every validation rollout step."""
        input_index = self._task.get_batch_input_indices()[-1]
        steps = []
        for validation_step_kwargs in self._task.steps("validation"):
            rollout_step = validation_step_kwargs["rollout_step"]
            output_indices = self._task.get_batch_output_indices(rollout_step=rollout_step)
            step_label = self._task.get_step_label(**validation_step_kwargs)  # dacycleNN / rstepNN for DA
            steps.extend(
                PlotStep(
                    input_index=input_index,
                    target_index=output_indices[out_step],
                    prediction_step=rollout_step,
                    output_step=out_step,
                    tag=f"{step_label}_out{out_step:02d}",
                )
                for out_step in range(self._task.num_output_timesteps)
            )
        return steps

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        steps = self.plot_steps()
        x = self.select_time(data, [steps[0].input_index])[0].squeeze()
        for step in steps:
            y_true = self.select_time(data, [step.target_index])[0].squeeze()
            y_pred = output_tensor[step.prediction_step, step.output_step, ...]
            y_pred = y_pred.squeeze() if hasattr(y_pred, "squeeze") else y_pred
            yield x, y_true, y_pred, step.tag


class TemporalDownscalerPlotAdapter(BasePlotAdapter):
    """Plot Adapter for TemporalDownscaler Task.

    Handles squeezing (1, n_step_output, ...) -> (n_step_output, ...).
    """

    def plot_steps(self) -> list[PlotStep]:
        """Show the first input next to each output time between the inputs."""
        input_index = self._task.get_batch_input_indices()[0]
        output_indices = self._task.get_batch_output_indices()
        return [
            PlotStep(
                input_index=input_index,
                target_index=output_indices[output_step],
                prediction_step=0,
                output_step=output_step,
                tag=f"istep{output_step + 1:02d}",
            )
            for output_step in range(self._task.num_output_timesteps)
        ]

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        steps = self.plot_steps()
        x = data[steps[0].input_index, ...].squeeze()
        for step in steps:
            y_true = data[step.target_index, ...].squeeze()
            pred = (
                output_tensor[step.output_step, 0]
                if getattr(output_tensor, "ndim", 0) >= 4
                else output_tensor[step.output_step]
            )
            y_pred = pred.squeeze() if hasattr(pred, "squeeze") else pred
            yield x, y_true, y_pred, step.tag

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        if getattr(output_tensor, "ndim", 0) == 5 and getattr(output_tensor, "shape", (0,))[0] == 1:
            return output_tensor.squeeze(0)
        return output_tensor


class AutoencoderPlotAdapter(BasePlotAdapter):
    """Plot Adapter for Autoencoder Task: single (sample, recon, tag) yield."""

    def plot_steps(self) -> list[PlotStep]:
        """Show the one sample next to its reconstruction."""
        return [PlotStep(input_index=0, target_index=0, prediction_step=0, output_step=0, tag="recon")]

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        (step,) = self.plot_steps()
        sample = data[step.input_index, ...].squeeze()
        recon = output_tensor[step.output_step, ...].squeeze()
        yield sample, sample, recon, step.tag


class EnsemblePlotAdapterWrapper(BasePlotAdapter):
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

    @property
    def default_plot_members(self) -> int | list[int] | None:
        """All members by default for ensemble runs (``None`` = no selection/all)."""
        return None

    def get_loss_plot_batch_start(self, **kwargs) -> int:
        return self._inner.get_loss_plot_batch_start(**kwargs)

    def prepare_plot_output_tensor(self, output_tensor: Any) -> Any:
        return self._inner.prepare_plot_output_tensor(output_tensor)

    def select_members(
        self,
        tensor: Any,
        members: int | list[int] | None = None,
        ensemble_axis: int | None = None,
    ) -> Any:
        """Slice ensemble members from the input tensor's ensemble axis.

        Parameters
        ----------
        tensor : Any
            Tensor carrying an ensemble axis at ``ensemble_axis``.
        members : int | list[int] | None
            Members to select. None returns all members, int/list selects specific members.
        ensemble_axis : int | None
            Physical position of the ensemble axis, as retrieved from the prediction
            view's layout. Defaults to 2, i.e., its position in the canonical gridded
            (batch, time, ensemble, grid, variables) layout.

        Returns
        -------
        Any
            Tensor with selected ensemble members. The ensemble axis is kept (with
            size 1 for a single member) so callers can rely on its position.
        """
        if members is None:
            return tensor
        if not isinstance(members, list):
            members = [members]
        axis = 2 if ensemble_axis is None else ensemble_axis % tensor.ndim
        return tensor[(slice(None),) * axis + (members,)]

    def prepare_loss_batch(self, batch: dict) -> dict:
        """Return the batch for loss plotting."""
        return batch

    def plot_steps(self) -> list[PlotStep]:
        return self._inner.plot_steps()

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        yield from self._inner.iter_plot_samples(data, output_tensor)
