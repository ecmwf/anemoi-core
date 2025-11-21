#yes# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from operator import itemgetter
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.utils.inicond import EnsembleInitialConditions

if TYPE_CHECKING:

    from omegaconf import DictConfig
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphEnsInterpMulti(BaseGraphModule):
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

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.rollout = 1

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

        self.ensemble_ic_generator = EnsembleInitialConditions(config=config, data_indices=data_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
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

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple:
        """Training / validation step."""
        LOGGER.debug(
            "SHAPES: batch[0].shape = %s, batch[1].shape == %s",
            list(batch[0].shape),
            list(batch[1].shape) if len(batch) == 2 else "n/a",
        )

        loss = torch.zeros(1, dtype=batch[0].dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        # New code for ensemble interpolator: (no rollout loop, instead loops through interp targets)

        batch = self.model.pre_processors(batch[0], in_place=not validation_mode)  # don't use EDA for interpolator
        x = self.ensemble_ic_generator(batch, None)  # no EDA for interpolator

        # Scalers which are delayed need to be initialized after the pre-processors
        if self.is_first_step:
            self.update_scalers(callback=AvailableCallbacks.ON_TRAINING_START)
            self.is_first_step = False
        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)

        x_bound = batch[:, itemgetter(*self.boundary_times)(self.imap)][
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, time, ens, latlon, nvar)


        y_pred = self(x_bound) # has shape (bs, time, ens, latlon, nvar)
        y = batch[:, itemgetter(*self.interp_times)(self.imap)][:,:, 0, :, self.data_indices.data.output.full]

        for interp_step in self.interp_times:
            y_pred_step = y_pred[:, interp_step - 1, ...]  # (bs, ens, latlon, nvar)
            y_step = y[:, interp_step - 1, ...]  # (bs, latlon, nvar)

            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss_next, y_pred_ens_group = checkpoint(
                self.gather_and_compute_loss,
                y_pred_step,
                y_step,
                self.loss,
                self.ens_comm_subgroup_size,
                self.ens_comm_subgroup,
                self.model_comm_group,
                validation_mode,
                use_reentrant=False,
            )
            if not validation_mode:
                y_pred_ens_group = []

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred_ens_group,
                    y_step,
                    interp_step - 1,
                    grid_shard_slice=self.grid_shard_slice,
                )

            loss += loss_next
            metrics.update(metrics_next)
            y_preds.append(y_pred_ens_group)

        _ens_ic = x_bound if validation_mode else None

        loss *= 1.0 / len(self.interp_times)
        return loss, metrics, y_preds, _ens_ic

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch[0] = super().allgather_batch(batch[0])
        if len(batch) == 2:
            batch[1] = super().allgather_batch(batch[1])
        return batch

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
            batch_size=batch[0].shape[0],
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
        # if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
        #     self.rollout += 1
        #     LOGGER.debug("Rollout window length: %d", self.rollout)
        # self.rollout = min(self.rollout, self.rollout_max)
        pass

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

        return val_loss, y_preds, ens_ic
