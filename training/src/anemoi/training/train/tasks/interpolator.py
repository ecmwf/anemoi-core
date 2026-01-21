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

from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig
    from torch import Tensor
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

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
        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        config.training.multistep_input = len(self.boundary_times)
        config.training.multistep_output = len(self.interp_times)
        LOGGER.info(
            "Interpolator: overwriting config entries 'multistep_input' to number of input times (%s)"
            " and 'multistep_output' to number of target times (%s).",
            len(self.boundary_times),
            len(self.interp_times),
        )

        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

    def _step(
        self,
        batch: Tensor,
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping[str, Tensor], Tensor]:
        """Training / validation step."""
        metrics = {}
        batch = self.model.pre_processors(batch, in_place=not validation_mode)

        # Scalers which are delayed need to be initialized after the pre-processors
        if self.is_first_step:
            self.update_scalers(callback=AvailableCallbacks.ON_TRAINING_START)
            self.is_first_step = False
        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)

        x_bound = batch[:, itemgetter(*self.boundary_times)(self.imap)][
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, time, ens, latlon, nvar)

        y_pred = self(x_bound)  # has shape (bs, time, ens, latlon, nvar)
        y = batch[:, itemgetter(*self.interp_times)(self.imap)][..., self.data_indices.data.output.full]
        loss = self._compute_loss(
            y_pred,
            y,
            model_comm_group=self.model_comm_group,
            grid_shard_slice=self.grid_shard_slice,
        )
        metrics = {}
        if validation_mode:
            metrics = self._compute_metrics(
                y_pred=y_pred,
                y=y,
                grid_shard_slice=self.grid_shard_slice,
            )
        return loss, metrics, y_pred
