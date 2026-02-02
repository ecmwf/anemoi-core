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

from torch.utils.checkpoint import checkpoint

from anemoi.training.train.objectives import DirectPredictionObjective
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphAutoEncoder(BaseGraphModule):
    """Graph neural network autoencoder for PyTorch Lightning."""

    task_type = "autoencoder"
    supported_objectives = (DirectPredictionObjective,)

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

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

        assert self.multi_step == self.multi_out, "Autoencoders must have the same number of input and output steps."

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        required_time_steps = max(self.multi_step, self.multi_out)
        x = {}

        for dataset_name, dataset_batch in batch.items():
            msg = (
                f"Batch length not sufficient for requested multi_step/multi_out for {dataset_name}!"
                f" {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg
            # Conditioning input: input variables for the reconstruction window.
            x[dataset_name] = dataset_batch[
                :,
                0:required_time_steps,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]

        y = {}

        for dataset_name, dataset_batch in batch.items():
            y_time = dataset_batch.narrow(1, 0, self.multi_out)
            var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            # Target output state: full output variables for the same window.
            y[dataset_name] = y_time.index_select(-1, var_idx)

        shapes = {k: y_.shape for k, y_ in y.items()}
        model_impl = self.model.model
        # Objective provides schedule (e.g., sigma or time) per batch.
        # Unused in direct prediction objective.
        schedule = self.objective.sample_schedule(
            shape=shapes,
            device=next(iter(batch.values())).device,
            model=model_impl,
        )
        # Objective builds (conditioning, target) pair for the training loss.
        # y_cond is None for direct prediction objective.
        y_cond, target = self.objective.build_training_pair(y, schedule)
        # Forward pass in objective space (e.g., denoising or velocity).
        y_pred = self.objective.forward(
            model_impl,
            x,
            y_cond,
            schedule,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )
        # Optional pre-loss weights (e.g., diffusion noise weighting).
        # None for direct prediction objective.
        pre_loss_weights = self.objective.pre_loss_weights(schedule, model=model_impl)

        # Clean prediction/target in normalized state space for metrics.
        # Autoencoder: only used in validation, no rollout.
        # y_pred_clean == y_pred, y == target for direct prediction objective.
        y_pred_clean, y_clean = self.objective.clean_pred_target_pair(y_pred, y, y_cond, schedule)

        metrics_y_pred = y_pred_clean if validation_mode else None
        metrics_y = y_clean if validation_mode else None
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            target,
            rollout_step=0,
            training_mode=True,
            validation_mode=validation_mode,
            metrics_y_pred=metrics_y_pred,
            metrics_y=metrics_y,
            pre_loss_weights=pre_loss_weights,
            use_reentrant=False,
        )

        return loss, metrics, y_pred

    def on_train_epoch_end(self) -> None:
        pass
