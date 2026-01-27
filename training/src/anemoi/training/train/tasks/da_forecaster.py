# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Data Assimilation Graph Forecaster.

This module implements a forecaster that performs DA cycling before starting
the standard autoregressive rollout. DA cycling allows the model to assimilate
sparse observations and create an analysis state for forecasting.

The key insight is that the model architecture expects `multistep_input` timesteps
as input, so we always maintain that window size. DA cycling iteratively advances
the window by predicting, blending with observations, and rolling forward.

Training flow:
    1. Start with first `multistep_input` frames
    2. For each DA cycle:
        a. Run forward pass to get prediction for next timestep
        b. Blend prediction with observations (use obs where available, pred where NaN)
        c. Roll window forward and append blended state
    3. After DA cycles complete, run standard autoregressive rollout

Configuration:
    training.multistep_input: Model input window size (unchanged from standard)
    training.da_cycles: Number of DA cycles before forecast starts (default: 0)
    training.da_loss_weight: Weight for loss during DA cycles (default: 0.0)

Batch requirements:
    Batch must have at least: multistep_input + da_cycles + rollout timesteps
    The datamodule automatically includes da_cycles in relative_date_indices,
    so batches will have the correct length when da_cycles is configured.

IMPORTANT - Imputer Configuration:
    The InputOnlyImputer should be configured with multi_step = multistep_input.
    DA target timesteps (at indices multistep_input, multistep_input+1, ...) should
    retain NaNs in prognostic variables for DA blending to work. This happens
    naturally since the imputer only fills the first multi_step timesteps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema


LOGGER = logging.getLogger(__name__)


class DAGraphForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster with Data Assimilation cycling.

    This forecaster performs DA cycling to assimilate observations before
    starting the standard autoregressive rollout. Each DA cycle:
    1. Predicts the next timestep
    2. Blends prediction with sparse observations (NaN = missing)
    3. Advances the input window

    This creates an "analysis" state that serves as the initial condition
    for forecasting, informed by both model dynamics and observations.

    Example:
        With multistep_input=2, da_cycles=3, rollout=4:
        - Batch: [t0, t1, t2, t3, t4, t5, t6, t7, t8]
        - Initial input: [t0, t1]
        - DA cycle 0: predict t2, blend with obs(t2) -> [t1, analysis0]
        - DA cycle 1: predict t3, blend with obs(t3) -> [analysis0, analysis1]
        - DA cycle 2: predict t4, blend with obs(t4) -> [analysis1, analysis2]
        - Rollout starts from analysis state, targets are t5, t6, t7, t8
    """

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize the DA Graph Forecaster.

        Parameters
        ----------
        config : BaseSchema
            Job configuration containing DA settings
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        statistics_tendencies : dict
            Statistics for tendencies
        data_indices : IndexCollection
            Indices of the training data
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

        # DA cycling configuration
        training_config = config.training

        self.da_cycles = getattr(training_config, "da_cycles", 0)
        self.da_loss_weight = getattr(training_config, "da_loss_weight", 0.0)

        if self.da_cycles > 0:
            LOGGER.info(
                "DA Forecaster initialized: da_cycles=%d, multistep_input=%d, da_loss_weight=%.3f",
                self.da_cycles,
                self.multi_step,
                self.da_loss_weight,
            )
            LOGGER.info(
                "Batch must have at least multistep_input + da_cycles + rollout = %d + %d + rollout timesteps",
                self.multi_step,
                self.da_cycles,
            )
        else:
            LOGGER.info(
                "DA Forecaster initialized with no DA cycles (standard forecasting)"
            )

    def _da_blend(
        self,
        y_pred: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Blend model prediction with observations using simple replacement.

        For each grid point and variable:
        - If observation is available (not NaN): use observation
        - If observation is NaN: use model prediction

        Parameters
        ----------
        y_pred : torch.Tensor
            Model prediction, shape (bs, grid, var_out)
        obs : torch.Tensor
            Observations (may contain NaNs), shape (bs, grid, var_in)

        Returns
        -------
        torch.Tensor
            Blended state ready for next input, shape (bs, grid, var_in)
        """
        # Clone obs to get forcing variables (already imputed by preprocessor)
        # and preserve tensor structure - more efficient than zeros + copy
        blended = obs.clone()

        # Blend prognostic variables: use obs where available, pred where NaN
        prog_idx_data = self.data_indices.data.input.prognostic
        prog_idx_out = self.data_indices.model.output.prognostic

        # In-place update of prognostic variables with blended values
        # torch.where creates the blend, then we assign to the correct positions
        blended[..., prog_idx_data] = torch.where(
            torch.isnan(obs[..., prog_idx_data]),
            y_pred[..., prog_idx_out],
            obs[..., prog_idx_data],
        )

        return blended

    def _prepare_input_with_da(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Prepare input by cycling through DA steps.

        Parameters
        ----------
        batch : torch.Tensor
            Full normalized batch with shape (bs, time, grid, var)
        validation_mode : bool
            Whether to compute validation metrics during DA

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict]
            - Prepared input of shape (bs, multi_step, grid, var_in)
            - Accumulated DA loss
            - DA metrics dictionary
        """
        da_loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        da_metrics = {}

        # Start with first multi_step frames (standard input window)
        x = batch[
            :, 0 : self.multi_step, ..., self.data_indices.data.input.full
        ]

        if self.da_cycles == 0:
            return x, da_loss, da_metrics

        # Perform DA cycles
        for cycle in range(self.da_cycles):
            # Forward pass to get prediction
            y_pred = self(x)  # (bs, grid, var_out)

            # Target observation index: multi_step + cycle
            target_idx = self.multi_step + cycle

            # Get observations at target time (may have NaNs in prognostic vars)
            obs = batch[:, target_idx, ..., self.data_indices.data.input.full]

            # Optionally compute loss for DA cycle
            if self.da_loss_weight > 0:
                y_target = batch[
                    :, target_idx, ..., self.data_indices.data.output.full
                ]
                cycle_loss, cycle_metrics, _ = checkpoint(
                    self.compute_loss_metrics,
                    y_pred,
                    y_target,
                    step=cycle,
                    validation_mode=validation_mode,
                    use_reentrant=False,
                )
                da_loss = da_loss + cycle_loss * self.da_loss_weight
                for k, v in cycle_metrics.items():
                    da_metrics[f"da_cycle_{cycle}/{k}"] = v

            # Blend prediction with observations
            blended = self._da_blend(y_pred, obs)

            # Roll the input window and append the blended state
            x = x.roll(-1, dims=1)
            x[:, -1, ...] = blended

        return x, da_loss, da_metrics

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step with DA-prepared input.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized batch to use for rollout
        rollout : int | None, optional
            Number of rollout steps
        validation_mode : bool, optional
            Whether in validation mode

        Yields
        ------
        Generator[tuple[torch.Tensor | None, dict, list], None, None]
            Loss value, metrics, and predictions (per step)
        """
        rollout = rollout or self.rollout

        # Validate batch length
        # Need: multi_step (initial) + da_cycles (DA targets) + rollout (forecast targets)
        min_batch_len = self.multi_step + self.da_cycles + rollout
        msg = (
            f"Batch length not sufficient for DA forecaster! "
            f"batch.shape[1]={batch.shape[1]} must be >= "
            f"multistep_input ({self.multi_step}) + da_cycles ({self.da_cycles}) + rollout ({rollout}) = {min_batch_len}"
        )
        assert batch.shape[1] >= min_batch_len, msg

        # Prepare input through DA cycling
        x, da_loss, da_metrics = self._prepare_input_with_da(batch, validation_mode)

        # Track if this is the first step (to include DA loss)
        first_step = True

        # Forecast targets start after multi_step + da_cycles
        forecast_start_idx = self.multi_step + self.da_cycles

        for rollout_step in range(rollout):
            # Prediction at this rollout step
            y_pred = self(x)

            # Target from batch
            target_idx = forecast_start_idx + rollout_step
            y = batch[:, target_idx, ..., self.data_indices.data.output.full]

            # Compute loss and metrics
            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            # Add DA loss to first step (so it's included in backward pass)
            if first_step and self.da_loss_weight > 0:
                loss = loss + da_loss
                metrics_next.update(da_metrics)
                first_step = False

            # Advance input using standard method from parent
            x = self._advance_input(x, y_pred, batch, rollout_step, forecast_start_idx)

            yield loss, metrics_next, y_pred

    def _advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
        forecast_start_idx: int,
    ) -> torch.Tensor:
        """Advance the input tensor for the next rollout step.

        Parameters
        ----------
        x : torch.Tensor
            Current input tensor, shape (bs, multi_step, grid, var_in)
        y_pred : torch.Tensor
            Model prediction, shape (bs, grid, var_out)
        batch : torch.Tensor
            Full batch tensor
        rollout_step : int
            Current rollout step index
        forecast_start_idx : int
            Starting batch index for forecast targets

        Returns
        -------
        torch.Tensor
            Updated input tensor for next step
        """
        x = x.roll(-1, dims=1)

        # Get prognostic variables from prediction
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[
            ..., self.data_indices.model.output.prognostic
        ]

        # Batch index for forcing/boundary
        batch_idx = forecast_start_idx + rollout_step

        # Apply boundary mask
        x[:, -1] = self.output_mask.rollout_boundary(
            x[:, -1],
            batch[:, batch_idx],
            self.data_indices,
            grid_shard_slice=self.grid_shard_slice,
        )

        # Get forcing from batch
        x[:, -1, :, :, self.data_indices.model.input.forcing] = batch[
            :, batch_idx, :, :, self.data_indices.data.input.forcing
        ]

        return x
