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

import torch
import numpy as np
from anemoi.utils.spectral import DCT2D, InverseDCT2D
from torch.utils.checkpoint import checkpoint

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.downscaler import GraphDiffusionDownscaler
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from collections.abc import Mapping
    from torch_geometric.data import HeteroData
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)

class EqualSNRGraphDiffusionDownscaler(GraphDiffusionDownscaler):
    """Graph neural network downscaler for Equal SNR diffusion."""

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
            statistics_tendencies=None,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        # Initiates spectral transforms
        self.NX, self.NY = len(np.unique(supporting_arrays["latitudes"])), len(np.unique(supporting_arrays["longitudes"]))
        self.transform = DCT2D(self.NX, self.NY, norm='ortho')
        self.itransform = InverseDCT2D(self.NX, self.NY, norm='ortho')

        # Reads variance
        path_to_variance = config.hardware.paths.variance + config.hardware.files.variance_tensor
        with open(path_to_variance, 'rb') as file:
            self.raw_variance = torch.from_numpy(np.load(file))
        path_to_variance_name_to_index = config.hardware.paths.variance + config.hardware.files.variance_name_to_index
        with open(path_to_variance_name_to_index, 'rb') as file:
            self.variance_name_to_index = np.load(file, allow_pickle=True).item()

        # Creates the model's output name to index dictionnary
        input_name_to_index = self.data_indices.data.input[0].name_to_index
        out_name_to_index = {
            k: v
            for k, v in self.data_indices.data.output.name_to_index.items()
            if v in self.data_indices.data.output.full
        }
        
        # Makes sure variables indices match model's name to index dictionnary
        V = torch.ones(self.NX*self.NY, len(out_name_to_index))
        LOGGER.info(f"out: {out_name_to_index.items()}, V: {self.variance_name_to_index}")
        for v, i in out_name_to_index.items():
            V[:, i] = self.raw_variance[:,self.variance_name_to_index[v]]

        # Makes variance accessible on all GPUs
        self.register_buffer('variance', V)

        # Saves variance to the checkpoint for inference
        self.model.supporting_arrays.update({'variance': np.array(self.variance.detach().cpu())})

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        noise_weights: torch.Tensor,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the diffusion spectral loss with noise and variance weighting.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        noise_weights : torch.Tensor
            Noise weights for diffusion loss computation
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed spectral loss with noise and variance weighting applied
        """
        return self.loss(
            y_pred,
            y,
            transform=self.transform,
            variance=self.variance,
            noise_weights=noise_weights,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Process batch size of len 3 with each item of dimensions:
        [batch_size, dates, ensemble, gridpoints, variables].
        """
        del batch_idx

        x_in, x_in_hres, y = batch

        # interpolate low-res input to high-res
        x_in_interp_to_hres = self.model.model.apply_interpolate_to_high_res(
            x_in[:, 0, ...],
            grid_shard_shapes=self.lres_grid_shard_shapes,
            model_comm_group=self.model_comm_group,
        )[:, None, ...]

        # compute target with residual and non_residual variables
        y_target = self.model.model.compute_residuals(y, x_in_interp_to_hres)

        # normalize inputs and target
        x_in_interp_to_hres_norm = self.model.pre_processors(x_in_interp_to_hres, dataset="input_lres", in_place=False)
        x_in_hres_norm = self.model.pre_processors(x_in_hres, dataset="input_hres", in_place=False)
        y_target_norm = self.model.pre_processors(y_target, dataset="output", in_place=False)

        # Scaler update
        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)

        # get noise level and associated loss weights
        sigma, noise_weights = self._get_noise_level(
            shape=(y_target_norm.shape[0],) + (1,) * (y_target_norm.ndim - 2),
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=y_target_norm.device,
        )

        # get targets and noised targets
        y_target_norm_noised = self._noise_target(y_target_norm, sigma)

        # prediction, fwd_with_preconditioning
        y_pred = self(
            x_in_interp_to_hres_norm,
            x_in_hres_norm,
            y_target_norm_noised,
            sigma,
        )  # shape is (bs, ens, latlon, nvar)

        # Use checkpoint for compute_loss_metrics
        loss, metrics_next = checkpoint(
            self.compute_loss_metrics,
            y_pred=y_pred[:, 0, ...],
            y=y_target_norm[:, 0, ...],  # removing time dim for loss computation,
            transform=self.transform,
            variance=self.variance,
            rollout_step=0,
            training_mode=training_mode,
            validation_mode=validation_mode,
            noise_weights=noise_weights,
            use_reentrant=False,
        )

        # Denormalize output tensors
        y_pred_denorm = self.model.post_processors(y_pred, dataset="output", in_place=False)

        # convert residual predictions to direct predictions
        y_pred_full = self.model.model.compute_direct_predictions(y_pred_denorm, x_in_interp_to_hres)

        # Add predicted residuals to the state
        y_preds = [y_pred_full, y_pred]

        return loss, metrics_next, y_preds

    def _noise_target(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to the state."""
        return x + sigma * self.itransform(torch.randn_like(x) * torch.pow(self.variance, 1/2))

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int = 0,
        grid_shard_slice: slice | None = None,
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

        Returns
        -------
        val_metrics : dict[str, torch.Tensor]
            validation metrics and predictions
        """
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False, dataset="output")
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False, dataset="output")

        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(y_pred_postprocessed, y_postprocessed)
                continue

            for mkey, indices in self.val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)
                if metric_name == "variance_weighted_spectral_mse":
                    metrics[metric_step_name] = metric(
                        y_pred_postprocessed,
                        y_postprocessed,
                        transform=self.transform,
                        variance=self.variance,
                        scaler_indices=[..., indices],
                        grid_shard_slice=grid_shard_slice,
                        group=self.model_comm_group,
                    )
                else:
                    metrics[metric_step_name] = metric(
                        y_pred_postprocessed,
                        y_postprocessed,
                        scaler_indices=[..., indices],
                        grid_shard_slice=grid_shard_slice,
                        group=self.model_comm_group,
                    )
        return metrics