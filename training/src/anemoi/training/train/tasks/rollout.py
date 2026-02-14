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
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.index_space import IndexSpace
    from anemoi.training.schemas.base_schema import BaseSchema


LOGGER = logging.getLogger(__name__)


class BaseRolloutGraphModule(BaseGraphModule, ABC):
    """Base class for rollout tasks."""

    def __init__(
        self,
        *,
        config: BaseSchema,
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
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        dataset_name: str,
        rollout_step: int = 0,
    ) -> torch.Tensor:
        """Default implementation used by simple rollout tasks.

        Supports model outputs shaped like:
        - (B, T, E, G, V)
        """
        keep_steps = min(self.n_step_output, self.n_step_input)

        x = x.roll(-keep_steps, dims=1)

        # TODO(dieter): see if we can replace for loop with tensor operations
        for i in range(keep_steps):
            # Get prognostic variables
            x[:, -(i + 1), ..., self.data_indices[dataset_name].model.input.prognostic] = y_pred[
                :,
                -(i + 1),
                ...,
                self.data_indices[dataset_name].model.output.prognostic,
            ]

            batch_time_index = self.n_step_input + (rollout_step + 1) * self.n_step_output - (i + 1)

            x[:, -(i + 1)] = self.output_mask[dataset_name].rollout_boundary(
                x[:, -(i + 1)],
                batch[:, batch_time_index],
                self.data_indices[dataset_name],
                grid_shard_slice=self.grid_shard_slice[dataset_name],
            )

            # get new "constants" needed for time-varying fields
            x[:, -(i + 1), ..., self.data_indices[dataset_name].model.input.forcing] = batch[
                :,
                batch_time_index,
                ...,
                self.data_indices[dataset_name].data.input.forcing,
            ]
        return x

    def _advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int,
    ) -> dict[str, torch.Tensor]:
        for dataset_name in x:
            x[dataset_name] = self._advance_dataset_input(
                x[dataset_name],
                y_pred[dataset_name],
                batch[dataset_name],
                rollout_step=rollout_step,
                dataset_name=dataset_name,
            )
        return x

    def _compute_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        step: int | None = None,
        grid_shard_slice: slice | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
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
            pred_layout=pred_layout,
            target_layout=target_layout,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        train_loss = super().training_step(batch, batch_idx)
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
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

        for loss_next, metrics_next, y_preds_next in self._rollout_step(
            batch,
            rollout=self.rollout,
            validation_mode=validation_mode,
        ):
            loss = loss + loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    @abstractmethod
    def _rollout_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        pass

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)
