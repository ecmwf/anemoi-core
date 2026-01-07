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

from .base import BaseGraphModule

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class BaseDiffusionForecaster(BaseGraphModule):
    """Base class for diffusion forecasters."""

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

        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.rho = config.model.model.diffusion.rho

    def get_input(self, batch: torch.Tensor) -> torch.Tensor:
        """Get input tensor shape for diffusion model."""
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, ens, latlon, nvar)
        msg = f"Batch length not sufficient for requested multi_step length!, {batch.shape[1]} !>= {self.multi_step}"
        assert batch.shape[1] >= self.multi_step, msg
        LOGGER.debug("SHAPE: x.shape = %s", list(x.shape))
        return x

    def get_target(self, batch: torch.Tensor) -> torch.Tensor:
        """Get target tensor shape for diffusion model."""
        y = batch[:, self.multi_step, ..., self.data_indices.data.output.full]
        LOGGER.debug("SHAPE: y.shape = %s", list(y.shape))
        return y

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
        weights: torch.Tensor | None = None,
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
        assert weights is not None, f"{self.__class__.__name__} must be provided for diffusion loss computation."
        return self.loss(
            y_pred,
            y,
            weights=weights,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _noise_target(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to the state."""
        return x + torch.randn_like(x) * sigma

    def _get_noise_level(
        self,
        shape: tuple[int],
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        rho: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rnd_uniform = torch.rand(shape, device=device)
        sigma = (sigma_max ** (1.0 / rho) + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
        return sigma, weight


class GraphDiffusionForecaster(BaseDiffusionForecaster):
    """Graph neural network forecaster for diffusion."""

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Step for the forecaster.

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

        Returns
        -------
        tuple[torch.Tensor, dict, torch.Tensor]
            Loss value, metrics, and predictions (per step)
        """
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)

        x = self.get_input(batch)  # (bs, multi_step, ens, latlon, nvar)
        y = self.get_target(batch)  # (bs, ens, latlon, nvar)

        # get noise level and associated loss weights
        sigma, noise_weights = self._get_noise_level(
            shape=(x.shape[0],) + (1,) * (x.ndim - 2),
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=x.device,
        )

        # get noised targets
        y_noised = self._noise_target(y, sigma)

        # prediction, fwd_with_preconditioning
        y_pred = self(x, y_noised, sigma)  # shape is (bs, ens, latlon, nvar)

        # Use checkpoint for compute_loss_metrics
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        return loss, metrics, y_pred


class GraphDiffusionTendForecaster(BaseDiffusionForecaster):
    """Graph neural network forecaster for diffusion tendency prediction."""

    def compute_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
        y_pred_state: torch.Tensor | None = None,
        y_state: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        """Compute loss and metrics for the given predictions and targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        step : int, optional
            Current step
        validation_mode : bool, optional
            Whether to compute validation metrics
        y_pred_state : torch.Tensor, optional
            Predicted states (for validation metrics) if they differ from y_pred (e.g., tendency-based models)
        y_state : torch.Tensor, optional
            Target states (for validation metrics) if they differ from y (e.g., tendency-based models)
        **kwargs
            Additional arguments to pass to loss computation

        Returns
        -------
        tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]
            Loss, metrics dictionary (if validation_mode), and full predictions
        """
        # Prepare tensors for loss/metrics computation
        y_pred_full, y_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred,
            y,
            validation_mode,
        )

        loss = self._compute_loss(y_pred_full, y_full, grid_shard_slice=grid_shard_slice, **kwargs)

        # Compute metrics if in validation mode
        metrics_next = {}
        if validation_mode:
            assert y_pred_state is not None, "y_pred_state must be provided for tendency-based diffusion models."
            assert y_state is not None, "y_state must be provided for tendency-based diffusion models."

            # Prepare states for metrics computation
            y_pred_state_full, y_state_full, grid_shard_slice = self._prepare_tensors_for_loss(
                y_pred_state,
                y_state,
                validation_mode,
            )

            metrics_next = self._compute_metrics(
                y_pred_state_full,
                y_state_full,
                grid_shard_slice=grid_shard_slice,
                **kwargs,
            )

        return loss, metrics_next, y_pred_state_full if validation_mode else None

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Step for the tendency-based diffusion forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized batch to use for rollout (assumed to be already preprocessed).
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Returns
        -------
        tuple[torch.Tensor, dict, torch.Tensor]
            Loss value, metrics, and predictions (per step)
        """
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)

        x = self.get_input(batch)  # (bs, multi_step, ens, latlon, nvar)
        y = self.get_target(batch)  # (bs, ens, latlon, nvar)

        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        if pre_processors_tendencies is None:
            msg = (
                "pre_processors_tendencies not found. This is required for tendency-based diffusion models. "
                "Ensure that statistics_tendencies is provided during model initialization."
            )
            raise AttributeError(msg)

        x_ref = self.model.model.apply_reference_state_truncation(
            x,
            self.grid_shard_shapes,
            self.model_comm_group,
        )

        tendency_target = self.model.model.compute_tendency(
            y,
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
        tendency_pred = self(x, tendency_target_noised, sigma)  # shape is (bs, ens, latlon, nvar)

        y_pred = None
        if validation_mode:
            # re-construct predicted state, de-normalised
            y_pred = self.model.model.add_tendency_to_state(
                x_ref,
                tendency_pred,
                self.model.post_processors,
                self.model.post_processors_tendencies,
                output_pre_processor=self.model.pre_processors,
            )

        # compute_loss_metrics
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            tendency_pred,
            tendency_target,
            y_pred_state=y_pred,
            y_state=y,
            validation_mode=validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        return loss, metrics, y_pred
