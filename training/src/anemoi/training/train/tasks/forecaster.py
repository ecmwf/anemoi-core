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

from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch


LOGGER = logging.getLogger(__name__)


class GraphForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster for PyTorch Lightning."""

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
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        # Multi-dataset setup (always expect dict inputs now)
        self.dataset_names = list(graph_data.keys())
        LOGGER.info("Forecaster initialized with datasets: %s", self.dataset_names)

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
        data_indices,  # type: ignore[misc]
        output_mask,  # type: ignore[misc]
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, data_indices.model.input.prognostic] = y_pred[
            ...,
            data_indices.model.output.prognostic,
        ]

        x[:, -1] = output_mask.rollout_boundary(
            x[:, -1],
            batch[:, self.multi_step + rollout_step],
            data_indices,
            grid_shard_slice=self.grid_shard_slice,
        )

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            data_indices.data.input.forcing,
        ]
        return x

    def _rollout_step(
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
                f", {dataset_batch.shape[1]} !>= {rollout + self.multi_step}"
            )
            assert dataset_batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x)

            y = {}
            for dataset_name, dataset_batch in batch.items():
                y[dataset_name] = dataset_batch[
                    :,
                    self.multi_step + rollout_step,
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
                    rollout_step,
                    validation_mode,
                    dataset_name,
                    use_reentrant=False,
                )

                # Add to total loss
                total_loss = dataset_loss if total_loss is None else total_loss + dataset_loss

                # Store metrics with dataset prefix
                for metric_name, metric_value in dataset_metrics.items():
                    metrics_next[f"{dataset_name}_{metric_name}"] = metric_value

            # Advance input state for each dataset
            for dataset_name in batch:
                x[dataset_name] = self.advance_input(
                    x[dataset_name],
                    y_pred[dataset_name],
                    batch[dataset_name],
                    rollout_step,
                    self.data_indices[dataset_name],
                    self.output_mask[dataset_name],
                )

            loss = total_loss

            yield loss, metrics_next, y_pred
