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
from omegaconf import DictConfig
from omegaconf import open_dict
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

    task_type = "time-interpolator"

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        assert self.n_step_output == 1, "For multiple outputs, use GraphMultiOutInterpolator"

        target_forcing_config = get_multiple_datasets_config(config.training.target_forcing)
        self.target_forcing_indices, self.use_time_fraction = {}, {}
        for dataset_name in self.dataset_names:
            if len(target_forcing_config[dataset_name].data) >= 1:
                self.target_forcing_indices[dataset_name] = itemgetter(*target_forcing_config[dataset_name].data)(
                    data_indices[dataset_name].data.input.name_to_index,
                )
                if isinstance(self.target_forcing_indices[dataset_name], int):
                    self.target_forcing_indices[dataset_name] = [self.target_forcing_indices[dataset_name]]
            else:
                self.target_forcing_indices[dataset_name] = []

            self.use_time_fraction[dataset_name] = target_forcing_config[dataset_name].time_fraction

        self.num_tfi = {name: len(idxs) for name, idxs in self.target_forcing_indices.items()}

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}
        self.n_step_input = 1
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
                    None,
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


class GraphMultiOutInterpolator(BaseGraphModule):
    """Graph neural network interpolator with multiple output steps for PyTorch Lightning."""

    task_type = "time-interpolator"

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : dict[str, HeteroData]
            Graph objects keyed by dataset name
        statistics : dict
            Statistics of the training data
        data_indices : dict[str, IndexCollection]
            Indices of the training data
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        with open_dict(config.training):
            config.training.multistep_output = len(config.training.explicit_times.target)
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        self.n_step_output = len(self.interp_times)
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.n_step_input = 1
        self.rollout = 1

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        x_bound = {}
        y = {}
        for dataset_name, dataset_batch in batch.items():
            x_bound[dataset_name] = dataset_batch[:, itemgetter(*self.boundary_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, time, ens, latlon, nvar)

            y[dataset_name] = dataset_batch[:, itemgetter(*self.interp_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]

        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)

        y_pred = self(x_bound)

        loss, metrics, _ = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred
