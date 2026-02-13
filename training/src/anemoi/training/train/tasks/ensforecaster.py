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
from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from collections.abc import Generator

    from omegaconf import DictConfig
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphEnsForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster for ensembles for PyTorch Lightning."""

    task_type = "forecaster"

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: dict[str, HeteroData],
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
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        # num_gpus_per_ensemble >= 1 and num_gpus_per_ensemble >= num_gpus_per_model (as per the DDP strategy)
        self.model_comm_group_size = config.system.hardware.num_gpus_per_model
        num_gpus_per_model = config.system.hardware.num_gpus_per_model
        num_gpus_per_ensemble = config.system.hardware.num_gpus_per_ensemble

        assert num_gpus_per_ensemble % num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            f"{num_gpus_per_ensemble} mod {num_gpus_per_model} != 0.\
            If you would like to run in deterministic mode, please use aifs-train"
        )

        self.lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.lr.rate
            / num_gpus_per_ensemble
        )
        LOGGER.info("Base (config) learning rate: %e -- Effective learning rate: %e", config.training.lr.rate, self.lr)

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = self.nens_per_device * num_gpus_per_ensemble // num_gpus_per_model
        LOGGER.info("Ensemble size: per device = %d, per ens-group = %d", self.nens_per_device, self.nens_per_group)

        # lazy init ensemble group info, will be set by the DDPEnsGroupStrategy:
        self.ens_comm_group = None
        self.ens_comm_group_id = None
        self.ens_comm_group_rank = None
        self.ens_comm_num_groups = None
        self.ens_comm_group_size = None

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

    def compute_dataset_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        step: int | None = None,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        y_pred_ens = gather_tensor(
            y_pred.clone(),  # for bwd because we checkpoint this region
            dim=TensorDim.ENSEMBLE_DIM,
            shapes=[y_pred.shape] * self.ens_comm_subgroup_size,
            mgroup=self.ens_comm_subgroup,
        )

        loss = self._compute_loss(
            y_pred_ens,
            y,
            grid_shard_slice=self.grid_shard_slice[dataset_name],
            grid_dim=self.grid_dim,
            grid_shard_shape=self.grid_shard_shapes,
            dataset_name=dataset_name,
        )

        # Compute metrics if in validation mode
        metrics_next = {}
        if validation_mode:
            metrics_next = self._compute_metrics(
                y_pred_ens,
                y,
                step=step,
                dataset_name=dataset_name,
                grid_shard_slice=self.grid_shard_slice[dataset_name],
            )

        return loss, metrics_next, y_pred_ens

    def _rollout_step(
        self,
        batch: dict[str, torch.Tensor],
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
        rollout_steps = rollout or self.rollout
        required_time_steps = rollout_steps * self.n_step_output + self.n_step_input

        # Stack the analysis nens_per_device times along an ensemble dimension
        # start rollout of preprocessed batch
        x = {}
        for dataset_name, dataset_batch in batch.items():
            dataset_batch = dataset_batch[
                :,
                0 : self.n_step_input,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, n_step_input, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested n_step_input length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg

            x[dataset_name] = torch.cat(
                [dataset_batch] * self.nens_per_device,
                dim=2,
            )  # shape == (bs, ms, nens_per_device, latlon, nvar)
            LOGGER.debug("Shapes: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))

            assert (
                len(x[dataset_name].shape) == 5
            ), f"Expected a 5-D tensor and got {len(x[dataset_name].shape)} dimensions, shape {x[dataset_name].shape}!"
            assert (x[dataset_name].shape[1] == self.n_step_input) and (
                x[dataset_name].shape[2] == self.nens_per_device
            ), (
                "Shape mismatch in x! "
                f"Expected ({self.n_step_input}, {self.nens_per_device}), "
                f"got ({x[dataset_name].shape[1]}, {x[dataset_name].shape[2]})!"
            )

        for rollout_step in range(rollout_steps):
            # prediction at rollout step rollout_step, shape = (bs, n_step_output, ens_size, latlon, nvar)
            y_pred = self(x, fcstep=rollout_step)
            y = {}
            [self.multi_step + rollout_step * self.multi_out + i for i in range(self.multi_out)]
            for dataset_name, dataset_batch in batch.items():
                start = self.n_step_input + rollout_step * self.n_step_output
                y_time = dataset_batch.narrow(1, start, self.n_step_output)[:, :, 0, :, :]
                var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
                y[dataset_name] = y_time.index_select(-1, var_idx)
                LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))

            loss, metrics_next, y_pred_ens_group = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred_ens_group
