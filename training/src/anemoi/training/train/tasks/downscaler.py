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
from icecream import ic
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
import pandas as pd
if TYPE_CHECKING:

    from collections.abc import Mapping
    from collections.abc import Generator

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDiffusionDownscaler(BaseGraphModule):
    """Graph neural network downscaler for diffusion."""

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

        self.rho = config.model.model.diffusion.rho
        self.lognormal_mean = config.model.model.diffusion.log_normal_mean
        self.lognormal_std = config.model.model.diffusion.log_normal_std
        self.training_approach = getattr(
            config.training, "training_approach", "probabilistic_low_noise"
        )
        reader_group_size = self.config.dataloader.read_group_size
        self.lres_grid_indices = instantiate(
            self.config.model_dump(by_alias=True).dataloader.lres_grid_indices,
            reader_group_size=reader_group_size,
        )
        self.lres_grid_indices.setup(graph_data)
        self.grid_shard_shapes_in_lres = self.lres_grid_indices.shard_shapes

        self.hres_grid_indices = instantiate(
            self.config.model_dump(by_alias=True).dataloader.hres_grid_indices,
            reader_group_size=reader_group_size,
        )
        self.hres_grid_indices.setup(graph_data)
        self.grid_shard_shapes_in_hres = self.hres_grid_indices.shard_shapes

        self.lres_grid_shard_shapes = None
        self.lres_grid_shard_slice = None
        self.hres_grid_shard_shapes = None
        self.hres_grid_shard_slice = None

    def forward(
        self,
        x_in_lres_interp_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.model.fwd_with_preconditioning(
            x_in_lres_interp_hres,
            x_in_hres,
            y_noised,
            sigma,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.hres_grid_shard_shapes,
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
        x_in_interp_to_hres_norm = self.model.pre_processors(
            x_in_interp_to_hres, dataset="input_lres", in_place=False
        ) 
        x_in_hres_norm = self.model.pre_processors(
            x_in_hres, dataset="input_hres", in_place=False
        ) 
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
            rollout_step=0,
            training_mode=training_mode,
            validation_mode=validation_mode,
            weights=noise_weights,
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

        if self.training_approach == "probabilistic_high_noise":
            rnd_uniform = torch.rand(shape, device=device)
            sigma = (
                sigma_max ** (1.0 / rho)
                + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
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

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """

        return batch  # already have the full grid

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : torch.Tensor
            Validation batch
        batch_idx : int
            Batch inces

        """
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(
                batch, batch_idx, training_mode=True, validation_mode=True
            )

        self.log(
            "val_" + self.loss.name + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch[0].shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(
            batch, batch_idx, training_mode=True, validation_mode=False
        )
        self.log(
            "train_" + self.loss.name + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
            sync_dist=True,
        )

        return train_loss

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
        y_postprocessed = self.model.post_processors(
            y, in_place=False, dataset="output"
        )
        y_pred_postprocessed = self.model.post_processors(
            y_pred, in_place=False, dataset="output"
        )

        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(
                    y_pred_postprocessed, y_postprocessed
                )
                continue

            for mkey, indices in self.val_metric_ranges.items():
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

        if self.keep_batch_sharded and self.model_comm_group_size > 1:
            self.lres_grid_shard_shapes = self.lres_grid_indices.shard_shapes
            self.hres_grid_shard_shapes = self.hres_grid_indices.shard_shapes
            self.grid_shard_shapes = self.grid_indices.shard_shapes
            self.lres_grid_shard_slice = self.lres_grid_indices.get_shard_slice(
                self.reader_group_rank
            )
            self.hres_grid_shard_slice = self.hres_grid_indices.get_shard_slice(
                self.reader_group_rank
            )
            self.grid_shard_slice = self.grid_indices.get_shard_slice(
                self.reader_group_rank
            )
        else:
            batch = self.allgather_batch(batch)
            self.lres_grid_shard_shapes, self.lres_grid_shard_slice = None, None
            self.hres_grid_shard_shapes, self.hres_grid_shard_slice = None, None
        return batch

    def on_fit_start(self):
        self.bw_last = 0.0
        self.opt_last = 0.0

    def on_train_batch_start(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_before_backward(self, loss):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._tb = time.perf_counter()

    def on_after_backward(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.bw_last = time.perf_counter() - self._tb

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure=None, *a, **k
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        optimizer.step(closure=optimizer_closure)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.opt_last = time.perf_counter() - t

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        it_s = 1.0 / dt
        """
        self.log_dict(
            {"it_s": it_s, "bw_s": self.bw_last, "opt_s": self.opt_last},
            on_step=True,
            prog_bar=True,
            logger=False,
        )

        print(
            {
                "step": self.global_step,
                "it_s": it_s,
                "bw_s": self.bw_last,
                "opt_s": self.opt_last,
            }
        )
        """
