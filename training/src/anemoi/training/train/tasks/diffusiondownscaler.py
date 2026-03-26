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
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDiffusionDownscaler(BaseGraphModule):
    """Graph neural network downscaler for diffusion.

    Follows the same patterns as GraphDiffusionTendForecaster where applicable:
    - Dict-based sigma/weights through the full pipeline
    - Dict-based forward() interface to model
    - Structured residual processor validation

    Key differences from the forecaster (by design):
    - Batch is NOT pre-normalized: compute_residuals needs raw y and x_interp
      to compute y_raw - x_interp_raw before normalizing with residual stats.
      Input normalization happens explicitly in _step after residual computation.
    - Two separate inputs (in_lres upsampled + in_hres) instead of single input
    - Residual prediction (y - interp(x)) instead of tendency prediction (y_t1 - y_t0)
    """

    task_type = "downscaler"

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
        self.lognormal_mean = config.model.model.diffusion.log_normal_mean
        self.lognormal_std = config.model.model.diffusion.log_normal_std
        self.training_approach = getattr(config.training, "training_approach", "probabilistic_low_noise")

        # Residual pairs: read from the model (single source of truth)
        self._residual_pairs = self.model.model._residual_pairs

        # Validate and cache residual processors (mirrors _validate_tendency_processors in forecaster)
        self._residual_pre_processors: dict[str, object] = {}
        self._residual_post_processors: dict[str, object] = {}
        self._validate_residual_processors()

    def _validate_residual_processors(self) -> None:
        """Validate and cache residual/tendency processors for each residual pair.

        Mirrors GraphDiffusionTendForecaster._validate_tendency_processors.
        """
        pre_tend = getattr(self.model, "pre_processors_tendencies", None)
        post_tend = getattr(self.model, "post_processors_tendencies", None)

        for target_ds in self._residual_pairs:
            if pre_tend is not None and target_ds in pre_tend:
                self._residual_pre_processors[target_ds] = pre_tend[target_ds]
                LOGGER.info("Using residual/tendency pre-processor for %s normalization", target_ds)
            else:
                LOGGER.warning(
                    "No residual/tendency pre-processor for %s — falling back to state normalization. "
                    "This may cause training instability with diffusion loss weights.",
                    target_ds,
                )

            if post_tend is not None and target_ds in post_tend:
                self._residual_post_processors[target_ds] = post_tend[target_ds]
                LOGGER.info("Using residual/tendency post-processor for %s denormalization", target_ds)
            else:
                LOGGER.warning("No residual/tendency post-processor for %s", target_ds)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Follows the forecaster pattern: all inputs/outputs are dicts keyed by dataset name.
        """
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
        dataset_name: str,
        weights: dict[str, torch.Tensor] | None = None,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the diffusion loss with noise weighting.

        Signature aligned with BaseDiffusionForecaster._compute_loss.
        """
        assert weights is not None, f"{self.__class__.__name__} requires weights for diffusion loss computation."

        return self.loss[dataset_name](
            y_pred,
            y,
            weights=weights[dataset_name],
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Process batch with dimensions [batch_size, dates, ensemble, gridpoints, variables].

        Note: Unlike the forecaster, the batch is NOT pre-normalized here.
        compute_residuals needs raw y and x_interp to compute the raw residual
        before normalizing with residual-specific statistics. Input normalization
        happens explicitly after residual computation.

        Handles mixed residual/direct prediction:
        - Prognostic output channels (in both source and target): residual prediction
        - Diagnostic output channels (only in target): direct prediction
        """
        x_in_lres = batch["in_lres"]
        x_in_hres = batch["in_hres"]
        y = batch["out_hres"]

        target_ds = self.model.model._decoder_datasets[0]  # e.g. "out_hres"
        source_ds = self._residual_pairs.get(target_ds)  # e.g. "in_lres", or None

        # Interpolate source dataset to target resolution
        x_in_lres_upsampled = self.model.model.residual["in_lres"](
            x_in_lres,
            grid_shard_shapes=None,
            model_comm_group=self.model_comm_group,
        )[
            :, :, None, :, :,
        ]  # Add ensemble dim: (batch, time, ensemble=1, grid, features)

        # Compute training target on raw data
        if source_ds is not None:
            channel_indices = self.model.model.get_matching_channel_indices(target_ds).to(
                x_in_lres_upsampled.device,
            )
            target = self.model.model.compute_residuals(
                y=y,
                x_interp=x_in_lres_upsampled[..., channel_indices],
                pre_processors_state=self.model.pre_processors[target_ds],
                pre_processors_tendencies=self._residual_pre_processors.get(target_ds),
                target_dataset=target_ds,
            )
        else:
            target = self.model.pre_processors[target_ds](y, in_place=False)

        # Normalize inputs (after residual computation which needs raw data)
        x_in_lres_upsampled = self.model.pre_processors["in_lres"](x_in_lres_upsampled, in_place=False)
        x_in_hres = self.model.pre_processors["in_hres"](x_in_hres, in_place=False)

        # Wrap as dicts for the rest of the pipeline (aligned with forecaster)
        target_dict = {target_ds: target}
        x_dict = {"in_lres": x_in_lres_upsampled, "in_hres": x_in_hres}

        # Get noise level and loss weights (dict-based, like forecaster)
        shapes = {k: v.shape for k, v in target_dict.items()}
        sigma, noise_weights = self._get_noise_level(
            shape=shapes,
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=target.device,
        )

        # Add noise to targets (dict-based, like forecaster)
        y_noised = self._noise_target(target_dict, sigma)

        # Forward pass with preconditioning
        y_pred = self(x_dict, y_noised, sigma)  # dict: {target_ds: (bs, time, ens, latlon, nvar)}

        # Compute loss and metrics
        loss, metrics_next, y_pred_out = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            target_dict,
            validation_mode=validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        # Reconstruct full prediction (denorm + add interpolated source)
        if source_ds is not None:
            y_pred_full = self.model.model.add_interp_to_state(
                state_inp=x_in_lres_upsampled,
                model_output=y_pred[target_ds],
                post_processors_state=self.model.post_processors,
                post_processors_tendencies=(
                    dict(self.model.post_processors_tendencies)
                    if hasattr(self.model, "post_processors_tendencies")
                    and self.model.post_processors_tendencies is not None
                    else None
                ),
                target_dataset=target_ds,
                source_dataset=source_ds,
            )
        else:
            y_pred_full = y_pred_out[target_ds]

        return loss, metrics_next, [y_pred_full]

    def _noise_target(
        self,
        x: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add noise to the state (dict-based, aligned with forecaster)."""
        return {name: x[name] + torch.randn_like(x[name]) * sigma[name] for name in x}

    def _get_noise_level(
        self,
        shape: dict[str, tuple[int, ...]],
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        rho: float,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Get noise level for diffusion training.

        Dict-based interface aligned with BaseDiffusionForecaster._get_noise_level.
        Extends the forecaster with support for multiple noise schedules
        (probabilistic_low_noise, probabilistic_high_noise, deterministic).
        """
        dataset_names = list(shape.keys())
        ref_shape = shape[dataset_names[0]]
        assert len(ref_shape) == 5, "Expected 5D tensor shape (batch, time, ensemble, grid, vars) for diffusion noise."
        batch_size = ref_shape[0]
        ensemble_size = ref_shape[2]

        base_shape = (batch_size, ensemble_size)

        if self.training_approach == "probabilistic_high_noise":
            rnd_uniform = torch.rand(base_shape, device=device)
            sigma_base = (
                sigma_max ** (1.0 / rho) + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
            ) ** rho
        elif self.training_approach == "probabilistic_low_noise":
            log_sigma = torch.normal(
                mean=self.lognormal_mean,
                std=self.lognormal_std,
                size=base_shape,
                device=device,
            )
            sigma_base = torch.exp(log_sigma)
        elif self.training_approach == "deterministic":
            sigma_base = torch.full(base_shape, fill_value=500000.0, device=device)
        else:
            raise ValueError(f"Unknown training_approach: {self.training_approach}")

        weight_base = (sigma_base**2 + sigma_data**2) / (sigma_base * sigma_data) ** 2

        # Reshape to broadcast: (batch, 1_time, ensemble, 1_grid, 1_vars)
        sigma_base = sigma_base[:, None, :, None, None]
        weight_base = weight_base[:, None, :, None, None]

        sigma, weight = {}, {}
        for dataset_name in shape:
            sigma[dataset_name] = sigma_base
            weight[dataset_name] = weight_base
        return sigma, weight

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], _: int) -> dict[str, torch.Tensor]:
        """Prepare batch after GPU transfer.

        Unlike the forecaster, we intentionally skip _normalize_batch here.
        compute_residuals needs raw y and x_interp to compute y_raw - x_interp_raw
        before normalizing with residual-specific statistics. Input normalization
        is done explicitly in _step after residual computation.
        """
        batch = self._setup_batch_sharding(batch)
        self._prepare_loss_scalers()
        return batch
