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
        self.multi_out = len(self.interp_times)
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.multi_step = 1
        self.rollout = 1

    def get_target_forcing(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target_forcing = {}
        for dataset_name, num_tfi in self.num_tfi.items():
            dataset_batch = batch[dataset_name]
            batch_size = dataset_batch.shape[0]
            ens_size = dataset_batch.shape[2]
            grid_size = dataset_batch.shape[3]

            interp_indices = torch.as_tensor(
                [self.imap[interp_step] for interp_step in self.interp_times],
                device=dataset_batch.device,
            )
            forcing_steps = dataset_batch.index_select(1, interp_indices)

            # get the forcing information for the target interpolation time:
            if num_tfi >= 1:
                forcing_indices = torch.as_tensor(
                    self.target_forcing_indices[dataset_name],
                    device=dataset_batch.device,
                )
                forcing = forcing_steps.index_select(-1, forcing_indices)
            else:
                forcing = forcing_steps[..., :0]

            if self.use_time_fraction[dataset_name]:
                time_fractions = torch.tensor(
                    [
                        (interp_step - self.boundary_times[-2]) / (self.boundary_times[-1] - self.boundary_times[-2])
                        for interp_step in self.interp_times
                    ],
                    device=dataset_batch.device,
                    dtype=dataset_batch.dtype,
                )
                time_fractions = time_fractions.view(1, -1, 1, 1, 1).expand(batch_size, -1, ens_size, grid_size, 1)
                forcing = torch.cat((forcing, time_fractions), dim=-1)

            target_forcing[dataset_name] = forcing

        return target_forcing

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        x_bound = {}
        for dataset_name in self.dataset_names:
            x_bound[dataset_name] = batch[dataset_name][:, itemgetter(*self.boundary_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, time, ens, latlon, nvar)

        target_forcing = self.get_target_forcing(batch)

        y_pred = self(x_bound, target_forcing=target_forcing)
        y = {}
        for dataset_name, dataset_batch in batch.items():
            interp_indices = torch.as_tensor(
                [self.imap[interp_step] for interp_step in self.interp_times],
                device=dataset_batch.device,
            )
            y[dataset_name] = dataset_batch.index_select(1, interp_indices)[
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]

        loss, metrics, _ = self.compute_loss_metrics(
            y_pred,
            y,
            validation_mode=validation_mode,
        )

        y_preds = []
        for step in range(len(self.interp_times)):
            y_step = {name: pred[:, step] for name, pred in y_pred.items()}
            y_preds.append(y_step)

        return loss, metrics, y_preds
