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
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDownscaler(BaseGraphModule):
    """Graph neural network downscaler for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: dict[str, HeteroData],
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : BaseSchema
            Configuration object
        graph_data : dict[str, HeteroData]
            Dictionary of graph data for each dataset
        truncation_data : dict
            Truncation configuration
        statistics : dict
            Training statistics
        statistics_tendencies : dict
            Tendency statistics
        data_indices : dict[str, IndexCollection]
            Data indices for each dataset
        metadata : dict
            Metadata
        supporting_arrays : dict
            Supporting arrays

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=None,  #statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        # Multi-dataset setup (always expect dict inputs now)
        self.dataset_names = list(graph_data.keys())
        LOGGER.info("Forecaster initialized with datasets: %s", self.dataset_names)

    def rollout_step(
        self,
        batch: dict,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the forecaster.

        Parameters
        ----------
        batch : dict
            Dictionary batch to use for rollout (assumed to be already preprocessed)
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        # start rollout of preprocessed batch
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, multi_step, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested multi_step length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {self.multi_step}"
            )
            assert dataset_batch.shape[1] >= self.multi_step, msg

        # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
        y_pred = self(x)

        y = {}
        for dataset_name, dataset_batch in batch.items():
            y[dataset_name] = dataset_batch[
                :,
                self.multi_step,
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]
        # y includes the auxiliary variables, so we must leave those out when computing the loss
        # Compute loss for each dataset and sum them up
        total_loss = None
        metrics_next = {}

        for dataset_name in batch:
            dataset_loss, dataset_metrics = checkpoint(
                self.compute_loss_metrics,
                y_pred[dataset_name],
                y[dataset_name],
                rollout,
                validation_mode,
                dataset_name,
                use_reentrant=False,
            )

            # should the loss of 2 empty tensors of size (2, 1, 40320, 0) be None or 0.0?? 
            dataset_loss = dataset_loss if dataset_loss is not None else 0.0 

            # Add to total loss
            total_loss = dataset_loss if total_loss is None else total_loss + dataset_loss

            # Store metrics with dataset prefix
            for metric_name, metric_value in dataset_metrics.items():
                metrics_next[f"{dataset_name}_{metric_name}"] = metric_value

        loss = total_loss

        yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: dict,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        batch_dtype = next(iter(batch.values())).dtype
        loss = torch.zeros(1, dtype=batch_dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next in self.rollout_step(batch, 0, validation_mode=validation_mode):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        return loss, metrics, y_preds
