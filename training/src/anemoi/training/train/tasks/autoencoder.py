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
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.schemas.base_schema import BaseSchema

from .forecaster import GraphForecaster

LOGGER = logging.getLogger(__name__)


class GraphAutoEncoder(GraphForecaster):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
        truncation_data: Optional[dict] = None,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : BaseSchema
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
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
            truncation_data=truncation_data,
        )

        assert self.rollout == 1, "Rollout must be 1 for autoencoder"
        assert self.rollout_epoch_increment == 0, "Rollout epoch increment must be 1 for autoencoder"

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        del batch_idx
        batch = self.allgather_batch(batch)

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        # for validation not normalized in-place because remappers cannot be applied in-place
        batch = self.model.pre_processors(batch, in_place=not validation_mode)

        if not self.updated_loss_mask:
            # update loss scalar after first application and initialization of preprocessors
            self.training_weights_for_imputed_variables(batch)

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)

        y_pred = self(x)
        y = batch[:, 0, ..., self.data_indices.internal_data.output.full]

        loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

        metrics_next = {}
        if validation_mode:
            metrics_next = self.calculate_val_metrics(
                y_pred,
                y,
                rollout_step=0,
            )

        metrics.update(metrics_next)
        y_preds.extend(y_pred)

        return loss, metrics, y_preds
