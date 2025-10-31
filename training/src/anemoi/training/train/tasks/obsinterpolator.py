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
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class ObsGraphInterpolator(BaseGraphModule):
    """ObsInterpolator: A graph neural network that leverages surface observations to inform interpolation between NWP states for fine-scale, high-frequency nowcasts of atmospheric variables (see https://arxiv.org/abs/2509.00017)
    """

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
        self.known_future_variables = list(itemgetter(*config.training.known_future_variables)(
            data_indices.data.input.name_to_index,
        )) if len(config.training.known_future_variables) else []
        if isinstance(self.known_future_variables, int):
            self.known_future_variables = [self.known_future_variables]
        self.multi_step = getattr(self.config["training"], "multistep_input", 1)
        boundary_times = config.training.explicit_times.input
        self.boundary_times = [t + self.multi_step - 1 for t in boundary_times]
        interp_times = config.training.explicit_times.target
        self.interp_times = [t + self.multi_step - 1 for t in interp_times]
        sorted_indices = sorted(
            set(range(self.multi_step)).union(
                self.boundary_times,
                self.interp_times,
            ),
        )
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        del batch_idx
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        batch = self.model.pre_processors(batch)
        present, future = itemgetter(*self.boundary_times)(self.imap)
        obs = set([var.item() for var in self.data_indices.data.input.full]).difference(
            set(self.known_future_variables)
        )
        x_init = batch[:, : self.multi_step][..., list(obs)]
        x_init_nwp = batch[:, 1][..., self.known_future_variables]
        x_init = rearrange(x_init, "batch time ens grid var -> batch ens grid (var time)")
        x_future = batch[:, future][..., self.known_future_variables]  # adding future known vars to the input
        x_bound = torch.cat([x_init, x_init_nwp, x_future], dim=-1)
        target_forcing = torch.empty(
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
            len(self.known_future_variables) + 1,
            device=self.device,
            dtype=batch.dtype,
        )
        for interp_step in self.interp_times:
            # get the forcing information for the target interpolation time:
            target_forcing[..., : len(self.known_future_variables)] = batch[
                :, self.imap[interp_step], :, :, self.known_future_variables
            ]
            target_forcing[..., -1] = (interp_step - future) / (future - present)
            x_with_intermediate_forcings = torch.cat([x_bound, target_forcing], dim=-1).unsqueeze(dim=1)
            y_pred = self(x_with_intermediate_forcings)
            y = batch[:, self.imap[interp_step], ...]
            loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    interp_step - 1,
                )
            metrics.update(metrics_next)
            y_preds.extend(y_pred)

        loss *= 1.0 / len(self.interp_times)
        return loss, metrics, y_preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )