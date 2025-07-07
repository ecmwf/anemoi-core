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

import einops
import torch
from torch.utils.checkpoint import checkpoint

from .forecaster import GraphForecaster

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDiffusionForecaster(GraphForecaster):
    """Graph neural network forecaster for diffusion."""

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

        self.rho = config.model.model.diffusion.noise.rho

    def forward(self, x: torch.Tensor, y_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.model.model.fwd_with_preconditioning(
            x,
            y_noised,
            sigma,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        weights: torch.Tensor = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the diffusion loss with noise weighting.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        weights : torch.Tensor
            Noise weights for diffusion loss computation
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed loss with noise weighting applied
        """
        msg = "Diffusion loss requires 'weights' parameter"
        if weights is None:
            raise ValueError(msg)

        fact = weights**0.5  # factor for mse loss
        return self.loss(
            y_pred * fact,
            y * fact,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
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
        batch = self.model.pre_processors(batch)  # normalized in-place

        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, ens, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):

            # get noise level and associated loss weights
            sigma, noise_weights = self._get_noise_level(
                shape=(x.shape[0],) + (1,) * (x.ndim - 2),
                sigma_max=self.model.model.sigma_max,
                sigma_min=self.model.model.sigma_min,
                sigma_data=self.model.model.sigma_data,
                device=x.device,
            )

            # get targets and noised targets
            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
            y_noised = self._noise_target(y, sigma)

            # prediction, fwd_with_preconditioning
            y_pred = self(
                x,
                y_noised,
                sigma,
            )  # shape is (bs, ens, latlon, nvar)

            # Use checkpoint for compute_loss_metrics
            loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                rollout_step,
                training_mode,
                validation_mode,
                weights=noise_weights,
                use_reentrant=False,
            )

            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred, y_noised

    def _noise_target(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to the state."""
        return x + torch.randn_like(x) * sigma

    def _get_noise_level(
        self,
        shape: torch.shape,
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        device: torch.device,
    ) -> tuple[torch.Tensor]:
        rnd_uniform = torch.rand(shape, device=device)
        sigma = (
            sigma_max ** (1.0 / self.rho)
            + rnd_uniform * (sigma_min ** (1.0 / self.rho) - sigma_max ** (1.0 / self.rho))
        ) ** self.rho
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
        return sigma, weight


class GraphDiffusionTendForecaster(GraphDiffusionForecaster):
    """Graph neural network forecaster for diffusion tendency prediction."""

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

    def compute_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
        training_mode: bool = True,
        validation_mode: bool = False,
        y_pred_state: torch.Tensor = None,
        y_state: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Compute loss on tendencies and metrics on states.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tendencies
        y : torch.Tensor
            Target tendencies
        rollout_step : int
            Current rollout step
        training_mode : bool
            Whether to compute training loss
        validation_mode : bool
            Whether to compute validation metrics
        y_pred_state : torch.Tensor, optional
            Predicted states (for validation metrics)
        y_state : torch.Tensor, optional
            Target states (for validation metrics)
        **kwargs
            Additional arguments (including weights for diffusion)

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor]]
            Loss (if training_mode) and metrics dictionary (if validation_mode)
        """
        # Prepare tendencies for loss computation
        tendency_pred_full, tendency_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred,
            y,
            training_mode,
            validation_mode,
        )

        # Compute loss on tendencies if in training mode
        loss = None
        if training_mode:
            loss = self._compute_loss(tendency_pred_full, tendency_full, grid_shard_slice, **kwargs)

        # Compute metrics on states if in validation mode
        metrics_next = {}
        if validation_mode and y_pred_state is not None and y_state is not None:
            # Prepare states for metrics computation
            y_pred_state_full, y_state_full, grid_shard_slice_metrics = self._prepare_tensors_for_loss(
                y_pred_state,
                y_state,
                False,
                validation_mode,
            )
            metrics_next = self._compute_metrics(
                y_pred_state_full,
                y_state_full,
                rollout_step,
                grid_shard_slice_metrics,
            )

        return loss, metrics_next

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        """Rollout step for the tendency-based flow forecaster.

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
        # Get batch tendencies from non processed batch
        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        if pre_processors_tendencies is None:
            msg = (
                "pre_processors_tendencies not found. This is required for tendency-based diffusion models. "
                "Ensure that statistics_tendencies is provided during model initialization."
            )
            raise AttributeError(msg)

        batch_size, nsteps, ensemble_size, _, _ = batch.shape
        batch_trunc = einops.rearrange(batch, "batch step ensemble grid vars -> (batch step ensemble) grid vars")
        batch_trunc = self.model.model._apply_truncation(
            batch_trunc,
            grid_shard_shapes=self.grid_shard_shapes,
            model_comm_group=self.model_comm_group,
        )
        batch_trunc = einops.rearrange(
            batch_trunc,
            "(batch step ensemble) grid vars -> batch step ensemble grid vars",
            batch=batch_size,
            step=nsteps,
            ensemble=ensemble_size,
        )

        batch_tendency = self.model.model.compute_tendency(
            batch[:, self.multi_step : self.multi_step + self.rollout, ...],
            batch_trunc[:, self.multi_step - 1 : self.multi_step + self.rollout - 1, ...],
            self.model.pre_processors,
            pre_processors_tendencies,
        )  # SL TODO: possible to do this in-place? double check that gradient flow works for out of place

        # Delayed scalers need to be initialized after the pre-processors once,
        # compute_tendency does run pre-processors
        if self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        # start rollout
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, ens, latlon, nvar)

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):

            # target tendency
            tendency_target = batch_tendency[:, rollout_step, ...]

            # normalise inputs
            x_norm = self.model.pre_processors(x, in_place=False, data_index=self.data_indices.data.input.full)

            # get noise level and associated loss weights
            sigma, noise_weights = self._get_noise_level(
                shape=(x.shape[0],) + (1,) * (x.ndim - 2),
                sigma_max=self.model.model.sigma_max,
                sigma_min=self.model.model.sigma_min,
                sigma_data=self.model.model.sigma_data,
                device=x.device,
            )

            # add noise to tendency
            tendency_target_noised = self._noise_target(tendency_target, sigma)

            # prediction, fwd_with_preconditioning
            tendency_pred = self(
                x_norm,
                tendency_target_noised,
                sigma,
            )  # shape is (bs, ens, latlon, nvar)

            # re-construct predicted state
            y_pred = self.model.model.add_tendency_to_state(
                x_norm[:, -1, ...],
                tendency_pred,
                self.model.post_processors,
                self.model.post_processors_tendencies,
            )  # SL TODO: make explicit that this is not in-place

            # Prepare states for validation if needed
            y_pred_norm = None
            y_norm = None
            y_noised = None
            if validation_mode:
                # metrics calculation and plotting expects normalised states
                # SL TODO: some could be in-place
                y_pred_norm = self.model.pre_processors(
                    y_pred,
                    in_place=False,
                    data_index=self.data_indices.data.output.full,
                )
                y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
                y_norm = self.model.pre_processors(y, in_place=False, data_index=self.data_indices.data.output.full)

                # Compute noised state for visualization
                y_noised = self.model.model.add_tendency_to_state(
                    x_norm[:, -1, ...],
                    tendency_target_noised,
                    self.model.post_processors,
                    self.model.post_processors_tendencies,
                )
                y_noised = self.model.pre_processors(
                    y_noised,
                    in_place=False,
                    data_index=self.data_indices.data.output.full,
                )

            # Use checkpoint for compute_loss_metrics
            loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                tendency_pred,  # predicted tendencies
                tendency_target,  # target tendencies
                rollout_step,
                training_mode,
                validation_mode,
                y_pred_norm,  # predicted states for metrics
                y_norm,  # target states for metrics
                weights=noise_weights,  # noise weights for loss
                use_reentrant=False,
            )

            # advance input using non-processed x, y_pred and batch
            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred_norm, y_noised
