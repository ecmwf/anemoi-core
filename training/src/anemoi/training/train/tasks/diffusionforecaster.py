# (C) Copyright 2025 Anemoi contributors.
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

        self.rho = config.model.model.diffusion.rho

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
        weights: torch.Tensor,
        grid_shard_slice: slice | None = None,
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
        return self.loss(
            y_pred,
            y,
            weights=weights,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, torch.Tensor], None, None]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized batch to use for rollout (assumed to be already preprocessed).
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, torch.Tensor], None, None]
            Loss value, metrics, and predictions (per step)

        """
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
                rho=self.rho,
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
                validation_mode,
                weights=noise_weights,
                use_reentrant=False,
            )

            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred

    def _noise_target(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to the state."""
        return x + torch.randn_like(x) * sigma

    def _get_noise_level(
        self,
        shape: torch.shape,
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        rho: float,
        device: torch.device,
    ) -> tuple[torch.Tensor]:
        rnd_uniform = torch.rand(shape, device=device)
        sigma = (sigma_max ** (1.0 / rho) + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
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
            Loss and metrics dictionary (if validation_mode)
        """
        # Prepare tendencies for loss computation
        tendency_pred_full, tendency_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred,
            y,
            validation_mode,
        )

        # Compute loss on tendencies
        loss = self._compute_loss(
            y_pred=tendency_pred_full,
            y=tendency_full,
            grid_shard_slice=grid_shard_slice,
            **kwargs,
        )

        # Compute metrics on states if in validation mode
        metrics_next = {}
        if validation_mode and y_pred_state is not None and y_state is not None:
            # Prepare states for metrics computation
            y_pred_state_full, y_state_full, grid_shard_slice_metrics = self._prepare_tensors_for_loss(
                y_pred_state,
                y_state,
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
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, torch.Tensor], None, None]:
        """Rollout step for the tendency-based diffusion forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized batch to use for rollout (assumed to be already preprocessed).
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, torch.Tensor], None, None]
            Loss value, metrics, and predictions (per step)

        """
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        if pre_processors_tendencies is None:
            msg = (
                "pre_processors_tendencies not found. This is required for tendency-based diffusion models. "
                "Ensure that statistics_tendencies is provided during model initialization."
            )
            raise AttributeError(msg)

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, ens, latlon, nvar)

        for rollout_step in range(rollout or self.rollout):

            assert rollout_step < 1, "multi-step rollout not supported"

            x_ref = batch[:, self.multi_step + rollout_step - 1, ...]
            x_ref = self.model.model.apply_reference_state_truncation(
                x_ref,
                self.grid_shard_shapes,
                self.model_comm_group,
            )

            tendency_target = self.model.model.compute_tendency(
                batch[:, self.multi_step + rollout_step, ...],
                x_ref,
                self.model.pre_processors,
                self.model.pre_processors_tendencies,
                input_post_processor=self.model.post_processors,
            )

            # get noise level and associated loss weights
            sigma, noise_weights = self._get_noise_level(
                shape=(x.shape[0],) + (1,) * (x.ndim - 2),
                sigma_max=self.model.model.sigma_max,
                sigma_min=self.model.model.sigma_min,
                sigma_data=self.model.model.sigma_data,
                rho=self.rho,
                device=x.device,
            )

            tendency_target_noised = self._noise_target(tendency_target, sigma)

            # prediction, fwd_with_preconditioning
            tendency_pred = self(
                x,
                tendency_target_noised,
                sigma,
            )  # shape is (bs, ens, latlon, nvar)

            # re-construct predicted state, de-normalised
            y_pred = self.model.model.add_tendency_to_state(
                x_ref[..., self.data_indices.data.input.full],
                tendency_pred,
                self.model.post_processors,
                self.model.post_processors_tendencies,
                output_pre_processor=self.model.pre_processors,
            )

            y = None
            if validation_mode:
                # metrics calculation and plotting expects normalised states
                y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]

            # compute_loss_metrics
            loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                tendency_pred,
                tendency_target,
                rollout_step,
                validation_mode,
                y_pred,
                y,
                weights=noise_weights,
                use_reentrant=False,
            )

            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
