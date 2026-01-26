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
from torch import Tensor
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class ObsGraphInterpolator(BaseGraphModule):
    """ObsInterpolator: Interpolates between NWP states using surface observations.

    A graph neural network that leverages surface observations to inform interpolation between NWP states
    for fine-scale, high-frequency nowcasts of atmospheric variables
    (see https://arxiv.org/abs/2509.00017).
    """

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
        self.known_future_variables = (
            list(
                itemgetter(*config.training.known_future_variables)(
                    data_indices.data.input.name_to_index,
                ),
            )
            if len(config.training.known_future_variables)
            else []
        )
        if isinstance(self.known_future_variables, int):
            self.known_future_variables = [self.known_future_variables]
        self.multi_step = self.config.training.multistep_input
        boundary_times = config.training.explicit_times.input
        self.boundary_times = [t + self.multi_step - 1 for t in boundary_times]
        interp_times = config.training.explicit_times.target
        self.multi_out = len(interp_times)
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
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping[str, Tensor], Tensor]:
        batch = self.model.pre_processors(batch)
        b, _, e, g, _ = batch.shape
        present, future = itemgetter(*self.boundary_times)(self.imap)

        obs = {var.item() for var in self.data_indices.data.input.full}.difference(
            set(self.known_future_variables),
        )
        x_init = batch[:, : self.multi_step, ..., list(obs)]  # here only past steps are used for observed vars
        x_init_nwp = batch[:, itemgetter(*self.boundary_times)(self.imap)][
            ...,
            self.known_future_variables,
        ]  # bounds are derived from variables we know in the future
        x_init = rearrange(x_init, "b t e g v -> b e g (v t)")
        x_init_nwp = rearrange(x_init_nwp, "b t e g v -> b e g (v t)")
        x_bound = torch.cat([x_init, x_init_nwp], dim=-1)
        # time-ratio forcing for each interp time
        num_interp = len(self.interp_times)
        ratios = torch.tensor(
            [(t - present) / (future - present) for t in self.interp_times],
            device=batch.device,
            dtype=batch.dtype,
        )
        ratios = ratios.reshape(1, 1, 1, num_interp).expand(b, e, g, num_interp)  # broadcast to (b,e,g,num_interp)

        x_full = torch.cat(
            [
                x_bound,  # static wrt interp
                ratios,  # normalized delta-time forcing
            ],
            dim=-1,
        ).unsqueeze(
            1,
        )  # fake time dimension

        y_pred = self(x_full)
        y = batch[:, itemgetter(*self.interp_times)(self.imap)]
        loss = self._compute_loss(
            y_pred,
            y,
            model_comm_group=self.model_comm_group,
            grid_shard_slice=self.grid_shard_slice,
        )

        metrics = {}
        if validation_mode:
            metrics = self._compute_metrics(y_pred, y)

        return loss, metrics, y_pred
