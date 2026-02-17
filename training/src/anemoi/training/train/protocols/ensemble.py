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
from anemoi.training.train.protocols.base import BaseGraphModule
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.training.train.training_task.base import BaseTask

LOGGER = logging.getLogger(__name__)


class EnsembleProtocol(BaseGraphModule):
    """Graph neural network forecaster for ensembles for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        task: BaseTask,
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
        task : BaseTask
            Training task
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
        super().__init__(
            config=config,
            task=task,
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

    def _expand_ens_dim(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expand the ensemble dimension in the input batch by stacking the data nens_per_device times."""
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch.tile(1, 1, self.nens_per_device, 1, 1)
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))

        return x

    def _collapse_ens_dim(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Collapse the ensemble dimension in the input batch by taking the first element along the ensemble dimension."""
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y[dataset_name] = dataset_batch[:, :, 0, ...]
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))

        return y

    def compute_dataset_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        rollout_step: int | None = None,
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
                rollout_step=rollout_step,
                dataset_name=dataset_name,
                grid_shard_slice=self.grid_shard_slice[dataset_name],
            )

        return loss, metrics_next, y_pred_ens

    def forward(self, x: dict[str, torch.Tensor], rollout_step: int | None = None, **kwargs) -> dict[str, torch.Tensor]:
        """Forward method.

        This method calls the model's forward method with the appropriate
        communication group and sharding information.
        """
        if rollout_step is not None:
            kwargs["fcstep"] = rollout_step
        else:
            kwargs["fcstep"] = 0 # TODO(Mario,Simon): set the conditioning on the step optional

        return self.model(
            x,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
            **kwargs,
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict, list]:
        """Training / validation step."""
        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        x = self._expand_ens_dim(x)

        for task_kwargs in self.task.steps:
            y_pred = self(x, **task_kwargs)
            y = self.task.get_targets(batch, data_indices=self.data_indices, **task_kwargs)
            y = self._collapse_ens_dim(y)

            loss_next, metrics_next, y_preds_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                **task_kwargs,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            # Advance input state for each dataset
            x = self.task.advance_input(
                x,
                y_preds_next,
                batch,
                **task_kwargs,
                data_indices=self.data_indices,
            )

            loss = loss + loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / self.task.num_steps
        return loss, metrics, y_preds
