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
import time
from typing import TYPE_CHECKING
import time
import torch
from torch.utils.checkpoint import checkpoint
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.utils.enums import TensorDim
from anemoi.training.train.tasks.base import BaseGraphModule
from hydra.utils import instantiate
from omegaconf import OmegaConf

if TYPE_CHECKING:

    from collections.abc import Mapping
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDiffusionDownscaler(BaseGraphModule):
    """Graph neural network downscaler for diffusion."""

    task_type = "downscaler"

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        # truncation_data: dict,
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
            statistics_tendencies=None,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.rho = config.model.model.diffusion.rho
        self.lognormal_mean = config.model.model.diffusion.log_normal_mean
        self.lognormal_std = config.model.model.diffusion.log_normal_std
        self.training_approach = getattr(config.training, "training_approach", "probabilistic_low_noise")
        self.x_in_matching_channel_indices = self._match_tensor_channels(
            data_indices["in_lres"].name_to_index,
            data_indices["out_hres"].name_to_index,
        )
        reader_group_size = self.config.dataloader.read_group_size

        fields_direct_prediction = getattr(config.data, "direct_prediction", None)
        self.indices_direct_prediction = ...

    def forward(
        self,
        x_in_lres_interp_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Note: All inputs are at HRES resolution at this point:
        - x_in_lres_interp_hres: upsampled from lres to hres
        - x_in_hres: native hres forcings
        - y_noised: native hres target (noised)
        """
        # Wrap inputs as dicts for model interface
        x_dict = {"in_lres": x_in_lres_interp_hres, "in_hres": x_in_hres}
        y_dict = {"out_hres": y_noised}
        sigma_dict = {"out_hres": sigma}

        result = self.model.model.fwd_with_preconditioning(
            x_dict,
            y_dict,
            sigma_dict,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=None,
        )

        return result["out_hres"]

    def _prepare_tensors_for_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, slice | None]:
        """Prepare tensors for loss computation, squeezing time dimension.

        Override base method to squeeze time dimension (always 1) before loss computation,
        since scalers expect 4D tensors (batch, ensemble, grid, vars).

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values with shape (batch, time=1, ensemble, grid, vars)
        y : torch.Tensor
            Target values with shape (batch, time=1, ensemble, grid, vars)
        validation_mode : bool
            Whether in validation mode
        dataset_name : str | None
            Dataset name for multi-dataset setups

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, slice | None]
            Prepared y_pred (4D), y (4D), and grid_shard_slice
        """
        # Call base class to handle sharding
        y_pred_full, y_full, grid_shard_slice = super()._prepare_tensors_for_loss(
            y_pred, y, validation_mode, dataset_name
        )

        # Squeeze time dimension (always 1) to get 4D tensors for loss computation
        # (batch, time=1, ensemble, grid, vars) -> (batch, ensemble, grid, vars)
        assert y_pred_full.shape[2] == 1, f"Expected ensemble dimension to be 1, got {y_pred_full.shape[2]}"
        assert y_full.shape[2] == 1, f"Expected ensemble dimension to be 1, got {y_full.shape[2]}"

        y_pred_full = y_pred_full.squeeze(2)  # Remove ensemble dimension
        y_full = y_full.squeeze(2)  # Remove ensemble dimension

        return y_pred_full, y_full, grid_shard_slice

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
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
        weights : dict[str, torch.Tensor]
            Noise weights for diffusion loss computation (per dataset)
        dataset_name : str | None
            Dataset name for multi-dataset setups
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed loss with noise weighting applied
        """
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets"
        assert (
            weights is not None
        ), f"weights must be provided for diffusion loss computation in {self.__class__.__name__}"

        # Extract weights for this dataset (handle both dict and tensor for backwards compatibility)
        dataset_weights = weights[dataset_name] if isinstance(weights, dict) else weights

        # Handle both per-dataset losses (ModuleDict) and single loss function
        loss_fn = self.loss[dataset_name] if isinstance(self.loss, torch.nn.ModuleDict) else self.loss

        # No transpose needed - loss expects (..., grid, vars) format
        return loss_fn(
            y_pred,
            y,
            weights=dataset_weights,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _step(
        self,
        batch: list[torch.Tensor],
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Process batch size of len 3 with each item of dimensions:
        [batch_size, dates, ensemble, gridpoints, variables].
        """
        x_in_lres = batch["in_lres"]
        x_in_hres = batch["in_hres"]
        y = batch["out_hres"]

        if x_in_lres.ndim != 5:
            raise ValueError(f"Expected x_in_lres to have 5 dimensions, got {x_in_lres.ndim}")
        if x_in_hres.ndim != 5:
            raise ValueError(f"Expected x_in_hres to have 5 dimensions, got {x_in_hres.ndim}")
        if y.ndim != 5:
            raise ValueError(f"Expected y to have 5 dimensions, got {y.ndim}")

        # Interpolate in_lres from lres to hres resolution
        x_in_lres_upsampled = self.model.model.residual["in_lres"](
            x_in_lres,  # (batch, multistep, ensemble, grid, features)
            grid_shard_shapes=None,
            model_comm_group=self.model_comm_group,
        )[
            :, :, None, :, :
        ]  # Add ensemble back: (batch, time, ensemble=1, grid, features)

        # Compute residuals: high-res target minus upsampled low-res input
        # Select only the matching channels from upsampled lres
        resid = y - x_in_lres_upsampled[..., self.x_in_matching_channel_indices.to(x_in_lres_upsampled.device)]

        x_in_lres_upsampled = self.model.pre_processors["in_lres"](x_in_lres_upsampled)
        x_in_hres = self.model.pre_processors["in_hres"](x_in_hres)
        resid = self.model.pre_processors["out_hres"](resid)

        # get noise level and associated loss weights
        sigma, noise_weights = self._get_noise_level(
            shape=(resid.shape[0],) + (1,) * (resid.ndim - 2),
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=resid.device,
        )

        # get targets and noised targets
        resid_noised = self._noise_target(resid, sigma)

        # Validate input dimensions
        assert (
            x_in_lres_upsampled.ndim == x_in_hres.ndim == resid_noised.ndim == 5
        ), f"Expected 5D tensors, got shapes {x_in_lres_upsampled.shape}, {x_in_hres.shape}, {resid_noised.shape}"

        # All inputs keep time dimension for future multi-step support
        y_pred = self(
            x_in_lres_upsampled,
            x_in_hres,
            resid_noised,
            sigma,
        )  # shape is (bs, time, ens, latlon, nvar)

        # Wrap tensors in dictionaries for multi-dataset compute_loss_metrics
        y_pred_dict = {"out_hres": y_pred}
        resid_dict = {"out_hres": resid}
        weights_dict = {"out_hres": noise_weights}

        # Compute loss and metrics with checkpoint (keeping time dimension)
        loss, metrics_next, y_pred_denorm_dict = checkpoint(
            self.compute_loss_metrics,
            y_pred_dict,
            resid_dict,
            validation_mode=validation_mode,
            weights=weights_dict,
            use_reentrant=False,
        )

        # Extract denormalized prediction from dictionary
        y_pred_denorm = y_pred_denorm_dict["out_hres"]  # (bs, time, ens, latlon, nvar)

        # Reconstruct full prediction: baseline (upsampled lres) + predicted residual
        # Denormalize baseline (upsampled lres)
        x_in_lres_upsampled_denorm = self.model.post_processors["in_lres"](x_in_lres_upsampled)

        # Full prediction = baseline + residual
        y_pred_full = (
            x_in_lres_upsampled_denorm[..., self.x_in_matching_channel_indices.to(x_in_lres_upsampled_denorm.device)]
            + y_pred_denorm
        )

        return loss, metrics_next, [y_pred_full, y_pred_denorm]

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
        sigma_override: float | None = None,
    ) -> tuple[torch.Tensor]:
        """Get noise level for diffusion training.

        Parameters
        ----------
        shape : torch.shape
            Shape for the noise tensor
        sigma_max : float
            Maximum noise level
        sigma_min : float
            Minimum noise level
        sigma_data : float
            Data scaling factor
        rho : float
            Distribution parameter
        device : torch.device
            Device for tensor creation
        sigma_override : float | None
            If provided, use this fixed sigma instead of sampling. Useful for testing.

        Returns
        -------
        tuple[torch.Tensor]
            Sigma and weight tensors
        """

        if sigma_override is not None:
            # Use fixed sigma for testing/debugging
            sigma = torch.full(shape, fill_value=sigma_override, device=device)
        elif self.training_approach == "probabilistic_high_noise":
            rnd_uniform = torch.rand(shape, device=device)
            sigma = (
                sigma_max ** (1.0 / rho) + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
            ) ** rho

        elif self.training_approach == "probabilistic_low_noise":
            log_sigma = torch.normal(
                mean=self.lognormal_mean,
                std=self.lognormal_std,
                size=shape,
                device=device,
            )
            sigma = torch.exp(log_sigma)
        elif self.training_approach == "deterministic":
            sigma = torch.full(
                shape,
                fill_value=500000.0,
                device=device,
            )

        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
        return sigma, weight

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int = 0,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted ensemble
        y: torch.Tensor
            Ground truth (target).
        rollout_step: int
            Rollout step
        grid_shard_slice: slice | None
            Grid shard slice for distributed training
        dataset_name: str | None
            Dataset name for multi-dataset setups

        Returns
        -------
        val_metrics : dict[str, torch.Tensor]
            validation metrics and predictions
        """
        metrics = {}

        # Use dataset_name for multi-dataset support
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"

        y_postprocessed = self.model.post_processors[dataset_name](y, in_place=False)
        y_pred_postprocessed = self.model.post_processors[dataset_name](y_pred, in_place=False)

        metrics_dict = self.metrics[dataset_name]
        val_metric_ranges = self.val_metric_ranges[dataset_name]

        for metric_name, metric in metrics_dict.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(y_pred_postprocessed, y_postprocessed)
                continue

            for mkey, indices in val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metrics[metric_step_name] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                    scaler_indices=[..., indices],
                    grid_shard_slice=grid_shard_slice,
                    group=self.model_comm_group,
                )

        return metrics

    def _match_tensor_channels(self, input_name_to_index, output_name_to_index):
        """Reorder and select channels from input tensor to match output tensor structure.

        Parameters
        ----------
        input_name_to_index : dict
            Mapping from variable names to indices in input tensor
        output_name_to_index : dict
            Mapping from variable names to indices in output tensor

        Returns
        -------
        torch.Tensor
            Tensor of indices for channel selection
        """
        common_channels = set(input_name_to_index.keys()) & set(output_name_to_index.keys())

        # for each output channel, look for corresponding input channel
        channel_mapping = []
        for channel_name in output_name_to_index.keys():
            if channel_name in common_channels:
                input_pos = input_name_to_index[channel_name]
                channel_mapping.append(input_pos)

        # Convert to tensor for indexing
        channel_indices = torch.tensor(channel_mapping)

        return channel_indices

    def on_after_batch_transfer(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.


        Parameters
        ----------
        batch : torch.Tensor
            Batch to transfer

        Returns
        -------
        torch.Tensor
            Batch after transfer
        """
        # Gathering/sharding of batch
        batch = self._setup_batch_sharding(batch)

        # Prepare scalers, e.g. init delayed scalers and update scalers
        self._prepare_loss_scalers()

        return batch
