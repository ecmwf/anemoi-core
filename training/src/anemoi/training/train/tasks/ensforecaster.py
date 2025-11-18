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

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    from omegaconf import DictConfig
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphEnsForecaster(BaseGraphModule):
    """Graph neural network forecaster for ensembles for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
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

        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)

        # num_gpus_per_ensemble >= 1 and num_gpus_per_ensemble >= num_gpus_per_model (as per the DDP strategy)
        self.model_comm_group_size = config.hardware.num_gpus_per_model
        assert config.hardware.num_gpus_per_ensemble % config.hardware.num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            f"{config.hardware.num_gpus_per_ensemble} mod {config.hardware.num_gpus_per_model} != 0.\
            If you would like to run in deterministic mode, please use aifs-train"
        )

        self.num_gpus_per_model = config.hardware.num_gpus_per_model
        self.num_gpus_per_ensemble = config.hardware.num_gpus_per_ensemble

        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_ensemble
        )

        LOGGER.info("Base (config) learning rate: %e -- Effective learning rate: %e", config.training.lr.rate, self.lr)

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = (
            config.training.ensemble_size_per_device * self.num_gpus_per_ensemble // config.hardware.num_gpus_per_model
        )
        LOGGER.info("Ensemble size: per device = %d, per ens-group = %d", self.nens_per_device, self.nens_per_group)

        # lazy init ensemble group info, will be set by the DDPEnsGroupStrategy:
        self.ens_comm_group = None
        self.ens_comm_group_id = None
        self.ens_comm_group_rank = None
        self.ens_comm_num_groups = None
        self.ens_comm_group_size = None

    def forward(self, x: torch.Tensor, fcstep: int) -> torch.Tensor:
        return self.model(
            x,
            fcstep=fcstep,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )

    def set_ens_comm_group(
        self,
        ens_comm_group: ProcessGroup,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def set_ens_comm_subgroup(
        self,
        ens_comm_subgroup: ProcessGroup,
        ens_comm_subgroup_id: int,
        ens_comm_subgroup_rank: int,
        ens_comm_subgroup_num_groups: int,
        ens_comm_subgroup_size: int,
    ) -> None:
        self.ens_comm_subgroup = ens_comm_subgroup
        self.ens_comm_subgroup_id = ens_comm_subgroup_id
        self.ens_comm_subgroup_rank = ens_comm_subgroup_rank
        self.ens_comm_subgroup_num_groups = ens_comm_subgroup_num_groups
        self.ens_comm_subgroup_size = ens_comm_subgroup_size

    def gather_and_compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        loss: torch.nn.Module,
        ens_comm_subgroup_size: int,
        ens_comm_subgroup: ProcessGroup,
        model_comm_group: ProcessGroup,
        return_pred_ens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Gather the ensemble members from all devices in my group.

        Eliminate duplicates (if any) and compute the loss.

        Args:
            y_pred: torch.Tensor
                Predicted state tensor, calculated on self.device
            y: torch.Tensor
                Ground truth
            loss: torch.nn.Module
                Loss function
            ens_comm_group_size: int
                Size of the ensemble communication group
            ens_comm_subgroup: ProcessGroup
                Ensemble communication subgroup
            model_comm_group: ProcessGroup
                Model communication group
            return_pred_ens: bool
                Validation flag: if True, we return the predicted ensemble (post-gather)

        Returns
        -------
            loss_inc:
                Loss
            y_pred_ens:
                Predictions if validation mode
        """
        # gather ensemble members,
        # full ensemble is only materialised on GPU in checkpointed region
        y_pred_ens = gather_tensor(
            y_pred.clone(),  # for bwd because we checkpoint this region
            dim=1,
            shapes=[y_pred.shape] * ens_comm_subgroup_size,
            mgroup=ens_comm_subgroup,
        )

        # compute the loss
        loss_inc = loss(y_pred_ens, y, squash=True, grid_shard_slice=self.grid_shard_slice, group=model_comm_group)

        return loss_inc, y_pred_ens if return_pred_ens else None

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
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the forecaster.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized batch to use for rollout (assumed to be already preprocessed)
        rollout : int, optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[torch.Tensor | None, dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        Returns
        -------
        None
            None
        """
        # Stack the analysis nens_per_device times along an ensemble dimension
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, ms, ens_dummy, latlon, nvar)

        x = torch.cat([x] * self.nens_per_device, dim=2)  # shape == (bs, ms, nens_per_device, latlon, nvar)
        LOGGER.debug("Shapes: x.shape = %s", list(x.shape))

        assert len(x.shape) == 5, f"Expected a 5-dimensional tensor and got {len(x.shape)} dimensions, shape {x.shape}!"
        assert (x.shape[1] == self.multi_step) and (x.shape[2] == self.nens_per_device), (
            "Shape mismatch in x! "
            f"Expected ({self.multi_step}, {self.nens_per_device}), "
            f"got ({x.shape[1]}, {x.shape[2]})!"
        )

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x, rollout_step)
            y = batch[
                :,
                self.multi_step + rollout_step,
                0,
                :,
                self.data_indices.data.output.full,
            ]  # self.data_indices.data.output.full
            LOGGER.debug("SHAPE: y.shape = %s", list(y.shape))

            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss, y_pred_ens_group = checkpoint(
                self.gather_and_compute_loss,
                y_pred,
                y,
                self.loss,
                self.ens_comm_subgroup_size,
                self.ens_comm_subgroup,
                self.model_comm_group,
                validation_mode,
                use_reentrant=False,
            )

            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred_ens_group,
                    y,
                    rollout_step,
                    grid_shard_slice=self.grid_shard_slice,
                )
            yield loss, metrics_next, y_pred_ens_group if validation_mode else [], x if validation_mode else None

    def _step(
        self,
        batch: tuple[torch.Tensor, ...],
        validation_mode: bool = False,
    ) -> tuple:
        """Training / validation step."""
        LOGGER.debug("SHAPES: batch.shape = %s", list(batch.shape))

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next, _ens_ic in self.rollout_step(
            batch,
            rollout=self.rollout,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds, _ens_ic

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor | dict:
        """Run one training step.

        Args:
            batch: tuple
                Batch data. tuple of length 1 or 2.
                batch[0]: analysis, shape (bs, multi_step + rollout, nvar, latlon)
                batch[1] (optional with ensemble): EDA perturbations, shape (multi_step, nens_per_device, nvar, latlon)
            batch_idx: int
                Training batch index

        Returns
        -------
            train_loss:
                Training loss
        """
        del batch_idx

        train_loss, _, _, _ = self._step(batch)

        self.log(
            "train_" + self.loss.name,
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
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

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a validation step.

        Parameters
        ----------
        batch: tuple
            Batch data. tuple of length 1 or 2.
            batch[0]: analysis, shape (bs, multi_step + rollout, nvar, latlon)
            batch[1] (optional): EDA perturbations, shape (nens_per_device, multi_step, nvar, latlon)
        batch_idx: int
            Validation batch index

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the validation loss, the predictions, and the ensemble initial conditions
        """
        del batch_idx

        with torch.no_grad():
            val_loss, metrics, y_preds, ens_ic = self._step(batch, validation_mode=True)
        self.log(
            "val_" + self.loss.name,
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
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
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds, ens_ic
