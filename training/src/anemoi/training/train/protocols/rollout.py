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
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.protocols.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.train.training_task.base import BaseTask


LOGGER = logging.getLogger(__name__)


class BaseRolloutGraphModule(BaseGraphModule, ABC):
    """Base class for rollout tasks."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        task: BaseTask,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        task : BaseTask
            Training task
        graph_data : dict[str, HeteroData]
            Graph objects keyed by dataset name
        statistics : dict
            Statistics of the training data
        data_indices : dict[str, IndexCollection]
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            task=task,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

    def _compute_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        step: int | None = None,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute validation metrics.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training

        Returns
        -------
        dict[str, torch.Tensor]
            Computed metrics
        """
        return self.calculate_val_metrics(
            y_pred,
            y,
            step=step,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        train_loss = super().training_step(batch, batch_idx)
        self.task.log_extra(logger=self.log, logger_enabled=self.logger_enabled)
        return train_loss

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict, list]:
        """Training / validation step."""
        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)

        for step in self.task.steps():
            y_pred = self(x)
            y = self.task.get_targets(batch, data_indices=self.data_indices, step=step)

            loss_next, metrics_next, y_preds_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            # Advance input state for each dataset
            x = self.task.advance_input(
                x,
                y_preds_next,
                batch,
                data_indices=self.data_indices,
                step=step
            )

            loss = loss + loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / len(self.task.steps())
        return loss, metrics, y_preds

    def on_train_epoch_end(self) -> None:
        self.task.on_train_epoch_end(current_epoch=self.current_epoch)


# task.steps()