# (C) Copyright 2024 Anemoi contributors.
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

PlotPayload holds the pre-processed (gathered, denormalized) data for a
single validation batch.  BasePlotAdapter.prepare_payload() produces it
once and caches it so that multiple callbacks sharing the same adapter
avoid redundant gather/denorm/mask work.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytorch_lightning as pl
    import torch

    from anemoi.training.tasks.base import BaseTask

LOGGER = logging.getLogger(__name__)


@dataclass
class PlotPayload:
    """Processed batch data ready for plotting callbacks.

    Produced once per batch by ``BasePlotAdapter.prepare_payload`` and shared
    across all callbacks that use that batch.

    Attributes
    ----------
    batch_idx : int
        The validation batch index this payload was produced for.
    batch : dict[str, torch.Tensor]
        Gathered (full-grid) batch tensors keyed by dataset name.
    outputs : tuple[torch.Tensor, list[dict[str, torch.Tensor]]]
        Gathered model outputs: (loss_scales, [pred_dict_per_step, ...]).
    post_processors : dict[str, Any]
        Deep-copied, CPU-resident post-processors keyed by dataset name.
    latlons : dict[str, np.ndarray]
        Latitude/longitude arrays in degrees, keyed by dataset name.
    """

    batch_idx: int
    batch: dict[str, torch.Tensor]
    outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]]
    post_processors: dict[str, Any]
    latlons: dict[str, np.ndarray] = field(default_factory=dict)


class BasePlotAdapter(ABC):
    """Abstract plotting contract. Subclasses define output_times, get_init_step, iter_plot_samples."""

    def __init__(self, task: BaseTask) -> None:
        self._task = task
        self._cached_payload: PlotPayload | None = None

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

    def prepare_payload(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, torch.Tensor],
        output: tuple[torch.Tensor, list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> PlotPayload:
        """Gather, denormalize, and cache batch data for plotting.

        This method performs the expensive shared pre-processing (distributed
        gather, post-processor deep-copy, latlon extraction) exactly once per
        batch.  Subsequent calls with the same ``batch_idx`` return the cached
        payload.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The Lightning module (provides allgather_batch, model, etc.).
        batch : dict[str, torch.Tensor]
            Raw validation batch keyed by dataset name.
        output : tuple
            Raw model output: (loss_scales, preds).  ``preds`` must be a list
            of per-step dicts.
        batch_idx : int
            Current validation batch index, used as cache key.

        Returns
        -------
        PlotPayload
            Gathered, CPU-resident data ready for plotting.
        """
        if self._cached_payload is not None and self._cached_payload.batch_idx == batch_idx:
            return self._cached_payload

        # 1. Gather batch shards across ranks
        gathered_batch = {
            dataset_name: pl_module.allgather_batch(dataset_tensor, dataset_name)
            for dataset_name, dataset_tensor in batch.items()
        }

        # 2. Gather prediction shards
        preds = output[1]
        if not isinstance(preds, list):
            msg = f"output[1] must be a list of per-step dicts, got {type(preds).__name__}"
            raise TypeError(msg)
        gathered_outputs: tuple[Any, list[dict[str, torch.Tensor]]] = (
            output[0],
            [
                {
                    dataset_name: pl_module.allgather_batch(dataset_pred, dataset_name)
                    for dataset_name, dataset_pred in pred.items()
                }
                for pred in preds
            ],
        )

        # 3. Deep-copy post-processors, gather nan_locations, move to CPU
        post_processors = copy.deepcopy(pl_module.model.post_processors)
        for dataset_name in post_processors:
            for post_processor in post_processors[dataset_name].processors.values():
                if hasattr(post_processor, "nan_locations"):
                    post_processor.nan_locations = pl_module.allgather_batch(
                        post_processor.nan_locations,
                        dataset_name,
                    )
            post_processors[dataset_name] = post_processors[dataset_name].cpu()

        # 4. Extract latlons (radians -> degrees)
        latlons: dict[str, np.ndarray] = {}
        for dataset_name in gathered_batch:
            latlons[dataset_name] = np.rad2deg(pl_module.model.model._graph_data[dataset_name].x.detach().cpu().numpy())

        self._cached_payload = PlotPayload(
            batch_idx=batch_idx,
            batch=gathered_batch,
            outputs=gathered_outputs,
            post_processors=post_processors,
            latlons=latlons,
        )
        return self._cached_payload

    @abstractmethod
    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        """Yield (x, y_true, y_pred, tag_suffix) or (sample, recon, tag) per plot sample."""
        ...


class ForecasterPlotAdapter(BasePlotAdapter):
    """Plot Adapter to adapt plots to the rollout set-up of the Forecaster Task.

    Handles multiple loss plots, n_step_output targets per step, multi-step iter.
    """

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
    """Plot Adapter for TemporalDownscaler Task.

    Handles squeezing (1, n_step_output, ...) -> (n_step_output, ...).
    """

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
    """Plot Adapter for Autoencoder Task: single (sample, recon, tag) yield."""

    def iter_plot_samples(self, data: Any, output_tensor: Any) -> Iterator[tuple[Any, Any, Any, str]]:
        sample = data[0, ...].squeeze()
        recon = output_tensor[0, ...].squeeze()
        yield sample, sample, recon, "recon"


class EnsemblePlotAdapterWrapper(BasePlotAdapter):
    """Wraps any task-specific adapter, adding ensemble member handling.

    This adapter decorates an inner (task-specific) adapter to handle the
    extra ensemble dimension present in ensemble training outputs.
    Batch shape convention: (B, T, E, G, V) where E is ensemble members.
    """

    def __init__(self, inner: BasePlotAdapter) -> None:
        self._inner = inner
        self._task = inner._task
        self._cached_payload: PlotPayload | None = None

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
