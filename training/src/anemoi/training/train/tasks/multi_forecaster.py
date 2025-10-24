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

from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.base import BaseGraphModule

from anemoi.training.utils.masks import NoOutputMask #TODO: remove when boundary handling for multi-step output is implemented

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphMultiForecaster(BaseGraphModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

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
        LOGGER.info("Instantiating the multi-output-step forecaster")
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
        self.multi_out = config.training.multistep_output
        LOGGER.info(f"multi-in: {self.multi_step}")
        LOGGER.info(f"multi-out: {self.multi_out}")
        
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

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
    ) -> torch.Tensor:
        x = x.roll(-self.multi_out, dims=1)
        #TODO: see if we can replace for loop with tensor operations
        for i in range(self.multi_out):
            # Get prognostic variables
            x[:, -(i+1), :, :, self.data_indices.model.input.prognostic] = y_pred[
                :,
                -(i+1),
                ...,
                self.data_indices.model.output.prognostic,
            ]

            # TODO: handle boundary conditions for multi-step output
            assert isinstance(self.output_mask, NoOutputMask), "Boundary rollout not implemented for multi-step output!"
            # x[:, -1] = self.output_mask.rollout_boundary(
            #     x[:, -1],
            #     batch[:, self.multi_step + rollout_step],
            #     self.data_indices,
            #     grid_shard_slice=self.grid_shard_slice,
            # )

            # get new "constants" needed for time-varying fields
            x[:, -(i+1), :, :, self.data_indices.model.input.forcing] = batch[
                :,
                self.multi_step + (rollout_step+1)*self.multi_out-(i+1),
                :,
                :,
                self.data_indices.data.input.forcing,
            ]
        return x

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        ############
        if rollout is None:
            rollout = self.rollout
        assert rollout == 1, "Only rollout_step=1 is currently supported for multi-step output"
        ### TODO: extend to multi-step output with rollout, then remove above.

        batch = self.model.pre_processors(batch)  # normalized in-place

        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.update_scalers(callback=AvailableCallbacks.ON_TRAINING_START)
            self.is_first_step = False

        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        
        required_time_steps = rollout*self.multi_out + self.multi_step
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {required_time_steps}"
        )
        assert batch.shape[1] >= required_time_steps, msg


        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            
            y_pred = self(x)
            fc_times =[self.multi_step + rollout_step*self.multi_out + i for i in range(self.multi_out) ]
            y = batch[:, fc_times, ...]
            y = y[..., self.data_indices.data.output.full]
            
            
            loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                rollout_step,
                training_mode,
                validation_mode,
                use_reentrant=False,
            )

            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds
