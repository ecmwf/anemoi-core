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
        self.x_in_matching_channel_indices = match_tensor_channels(
            self.data_indices.data.input[0].name_to_index,
            {
                k: v
                for k, v in self.data_indices.data.output.name_to_index.items()
                if v in self.data_indices.data.output.full
            },
        )
        reader_group_size = self.config.dataloader.read_group_size
        """
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
        """

        fields_direct_prediction = getattr(config.data, "direct_prediction", None)
        self.indices_direct_prediction = ...

    

    def get_inputs(self, batch: dict, sample_length: int) -> dict:
        # start rollout of preprocessed batch
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, multi_step, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested multi_step length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {sample_length}"
            )
            assert dataset_batch.shape[1] >= sample_length, msg
            LOGGER.info("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_targets(self, batch: dict, lead_step: int) -> dict:
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y[dataset_name] = dataset_batch[
                :,
                lead_step,
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]
            LOGGER.info("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y        

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
        x = self.get_inputs(batch, sample_length=self.multi_step)  # (bs, multi_step, ens, latlon, nvar)
        y = self.get_targets(batch, lead_step=self.multi_step - 1)  # (bs, multi_step, ens, latlon, nvar)

        assert len(x) == 2, "Expected x to contain two elements: [low_res, high_res]"

        x_lres, x_hres = x 

        x_lres_upsampled = self.model.model.apply_interpolate_to_high_res(
            x_lres[:, 0, ...],
            grid_shard_shapes=self.lres_grid_shard_shapes,
            model_comm_group=self.model_comm_group,
        )[:, None, ...]

        resid = self.model.model.compute_residuals(
            y,
            x_lres_upsampled[..., self.x_in_matching_channel_indices.to(
            x_lres_upsampled.device
        )],
        )

        x_lres_upsampled = self.model.pre_processors(
            x_lres_upsampled, dataset="input_lres"
        )  
        x_hres = self.model.pre_processors(
            x_hres, dataset="input_hres"
        )
        resid = self.model.pre_processors(resid, dataset="output")

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

        y_pred = self(
            x_lres_upsampled,
            x_hres,
            resid_noised,
            sigma,
        )  # shape is (bs, ens, latlon, nvar)


        # Use checkpoint for compute_loss_metrics
        loss, metrics_next = checkpoint(
            self.compute_loss_metrics,
            y_pred=y_pred[:, 0, ...],
            y=resid[:, 0, ...],  # removing time dim for loss computation,

            training_mode=training_mode,
            validation_mode=validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        denorm_x_lres_upsampled = self.model.post_processors(
            x_lres_upsampled, dataset="input_lres"
        )
        denorm_y_pred = self.model.post_processors(
            y_pred, dataset="output"
        )

        y_preds = [denorm_x_lres_upsampled + denorm_y_pred, denorm_y_pred]

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



def match_tensor_channels(input_name_to_index, output_name_to_index):
    """
    Reorders and selects channels from input tensor to match output tensor structure.
    x_in: Input tensor of shape [batch, n_grid_points, channels]
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
