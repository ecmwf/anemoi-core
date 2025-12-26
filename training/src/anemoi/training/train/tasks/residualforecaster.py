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

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)

class GraphResidualForecaster(BaseGraphModule):
    """Graph neural-network forecaster that predicts *normalized residuals*
    Δx_norm = (x_{t+1} - x_t) / std_Δx  (mean difference assumed zero).
    
    
    Matches the GraphCast framework.
    """

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
        
        # rollout control
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        LOGGER.info("Residual prediction mode enabled.")
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
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[
            ...,
            self.data_indices.model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(
            x[:, -1],
            batch[:, self.multi_step + rollout_step],
            self.data_indices,
            grid_shard_slice=self.grid_shard_slice,
        )

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
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
        """Rollout step for the residual forecaster (GraphCast-style).

        The model predicts *normalized residuals*:
            Δx_norm = (y_true - x_last) / σ_Δx
        where y_true and x_last are in physical space.     
            
        The loss is computed in this normalized residual space.

        At each step, the next-state prediction is reconstructed as:
            x̂_{t+1} = x_last + Δx̂_norm * σ_Δx

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
        batch = self.model.pre_processors(batch)  # normalized in-place (no cloning)

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
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            
            # forward prediction (normalized residual), shape = (bs, latlon, nvar)
            # Note: For input to self, x needs to be normalized. 
            # At this point, we want to run the 
            # self.model.pre_processors as otherwise, we 
            # want an "x" in physical space for the remaining computations. 
            Δx̂_norm = self(x)
                
            # true next-state tensor
            # Retrieve y_true and x_last in *physical* space
            # Undo the preprocessor normalization (mean/std) before computing Δx.
            # Caution: introducing an extra GPU memory hit :( 
            # Efficient: only unnormalize the current + next states
            x_last = self.model.post_processors(x[:, -1, ..., self.data_indices.data.output.prognostic], in_place=False)
            y_true = self.model.post_processors(
                    batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.prognostic],
                    in_place=False
            )
            
            # Compute normalized true residual Δx_true_norm = (y_true - x_last) / σ_Δx
            Δx_true_norm = self.model.residual_normalizer.transform(x_last, y_true)#, in_place=True)
            
            # loss in normalized residual space
            loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                Δx̂_norm,
                Δx_true_norm,
                rollout_step,
                validation_mode,
                use_reentrant=False,
            )

            # reconstruct next-state prediction  x_{t+1} = x_t + Δx̂_norm * std_Δx
            # and then re-normalize it so it can be appended onto the normalized "x" for rollout. 
            y_pred = self.model.residual_normalizer.inverse_transform(x_last, Δx̂_norm)
            y_pred = self.model.pre_processors(y_pred, data_index=self.data_indices.model.input.prognostic)
                        
            # feed next-state prediction back into input window
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
