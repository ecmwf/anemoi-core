# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from operator import itemgetter
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.models.interface import AnemoiModelInterface
    from anemoi.training.config_types import Settings

LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

    task_type = "time-interpolator"

    def __init__(
        self,
        *,
        config: Settings,
        graph_data: HeteroData,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        output_masks: dict,
        grid_indices: dict,
        scalers: dict,
        updating_scalars: dict,
        losses: dict,
        metrics: dict,
        val_metric_ranges: dict,
        optimizer_builder: Callable[[Iterable[torch.nn.Parameter], float], torch.optim.Optimizer] | None = None,
        model_interface: AnemoiModelInterface,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : Settings
            Job configuration
        graph_data : HeteroData
            Graph object
        data_indices : dict[str, IndexCollection]
            Indices of the training data,
        metadata : dict
            Provenance information
        output_masks : dict
            Pre-built output masks keyed by dataset name.
        grid_indices : dict
            Pre-built grid indices keyed by dataset name.
        scalers : dict
            Pre-built scalers keyed by dataset name.
        updating_scalars : dict
            Pre-built updating scalers keyed by dataset name.
        losses : dict
            Pre-built losses keyed by dataset name.
        metrics : dict
            Pre-built metrics keyed by dataset name.
        val_metric_ranges : dict
            Pre-computed validation metric ranges keyed by dataset name.
        optimizer_builder : Callable, optional
            Callable that builds the optimizer from params and lr.

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            data_indices=data_indices,
            metadata=metadata,
            output_masks=output_masks,
            grid_indices=grid_indices,
            scalers=scalers,
            updating_scalars=updating_scalars,
            losses=losses,
            metrics=metrics,
            val_metric_ranges=val_metric_ranges,
            optimizer_builder=optimizer_builder,
            model_interface=model_interface,
        )
        target_forcing_config = get_multiple_datasets_config(config.training.target_forcing)
        self.target_forcing_indices, self.use_time_fraction = {}, {}
        for dataset_name in self.dataset_names:
            data_config = target_forcing_config[dataset_name]["data"]
            if len(data_config) >= 1:
                self.target_forcing_indices[dataset_name] = itemgetter(*data_config)(
                    data_indices[dataset_name].data.input.name_to_index,
                )
                if isinstance(self.target_forcing_indices[dataset_name], int):
                    self.target_forcing_indices[dataset_name] = [self.target_forcing_indices[dataset_name]]
            else:
                self.target_forcing_indices[dataset_name] = []

            self.use_time_fraction[dataset_name] = target_forcing_config[dataset_name]["time_fraction"]

        self.num_tfi = {name: len(idxs) for name, idxs in self.target_forcing_indices.items()}

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}
        self.multi_step = 1
        self.rollout = 1

    def get_target_forcing(self, batch: dict[str, torch.Tensor], interp_step: int) -> dict[str, torch.Tensor]:
        batch_size = next(iter(batch.values())).shape[0]
        ens_size = next(iter(batch.values())).shape[2]
        grid_size = next(iter(batch.values())).shape[3]
        batch_type = next(iter(batch.values())).dtype

        target_forcing = {}
        for dataset_name, num_tfi in self.num_tfi.items():
            target_forcing[dataset_name] = torch.empty(
                batch_size,
                ens_size,
                grid_size,
                num_tfi + self.use_time_fraction[dataset_name],
                device=self.device,
                dtype=batch_type,
            )

            # get the forcing information for the target interpolation time:
            if num_tfi >= 1:
                target_forcing[dataset_name][..., :num_tfi] = batch[dataset_name][
                    :,
                    self.imap[interp_step],
                    :,
                    :,
                    self.target_forcing_indices[dataset_name],
                ]

            if self.use_time_fraction[dataset_name]:
                target_forcing[dataset_name][..., -1] = (interp_step - self.boundary_times[-2]) / (
                    self.boundary_times[-1] - self.boundary_times[-2]
                )

        return target_forcing

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x_bound = {}
        for dataset_name in self.dataset_names:
            x_bound[dataset_name] = batch[dataset_name][:, itemgetter(*self.boundary_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, time, ens, latlon, nvar)

        for interp_step in self.interp_times:
            target_forcing = self.get_target_forcing(batch, interp_step)

            y_pred = self(x_bound, target_forcing)
            y = {}
            for dataset_name, dataset_batch in batch.items():
                y[dataset_name] = dataset_batch[
                    :,
                    self.imap[interp_step],
                    :,
                    :,
                    self.data_indices[dataset_name].data.output.full,
                ]

            loss_step, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=interp_step - 1,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            loss += loss_step
            metrics.update(metrics_next)
            y_preds.append(y_pred)

        loss *= 1.0 / len(self.interp_times)
        return loss, metrics, y_preds

    def forward(self, x: torch.Tensor, target_forcing: torch.Tensor) -> torch.Tensor:
        return super().forward(x, target_forcing=target_forcing)
