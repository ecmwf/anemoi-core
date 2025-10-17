# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from operator import itemgetter

import torch
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
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
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )
        if len(config.training.target_forcing.data) >= 1:
            self.target_forcing_indices = itemgetter(*config.training.target_forcing.data)(
                data_indices.data.input.name_to_index,
            )
            if isinstance(self.target_forcing_indices, int):
                self.target_forcing_indices = [self.target_forcing_indices]
        else:
            self.target_forcing_indices = []

        self.use_time_fraction = config.training.target_forcing.time_fraction

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.rollout = 1

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x_bound = batch[:, itemgetter(*self.boundary_times)(self.imap)][
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, time, ens, latlon, nvar)

        num_tfi = len(self.target_forcing_indices)
        target_forcing = torch.empty(
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
            num_tfi + self.use_time_fraction,
            device=self.device,
            dtype=batch.dtype,
        )
        for interp_step in self.interp_times:
            # get the forcing information for the target interpolation time:
            if num_tfi >= 1:
                target_forcing[..., :num_tfi] = batch[:, self.imap[interp_step], :, :, self.target_forcing_indices]
            if self.use_time_fraction:
                target_forcing[..., -1] = (interp_step - self.boundary_times[-2]) / (
                    self.boundary_times[-1] - self.boundary_times[-2]
                )

            y_pred = self(x_bound, target_forcing)
            y = batch[:, self.imap[interp_step], :, :, self.data_indices.data.output.full]

            loss_step, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                interp_step - 1,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            loss += loss_step
            metrics.update(metrics_next)
            y_preds.append(y_pred)

        loss *= 1.0 / len(self.interp_times)
        return loss, metrics, y_preds

    def forward(self, x: torch.Tensor, target_forcing: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
            target_forcing=target_forcing,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )
