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

from anemoi.training.train.objectives import DiffusionObjective
from anemoi.training.train.objectives import DirectPredictionObjective
from anemoi.training.train.objectives import FlowObjective
from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch

LOGGER = logging.getLogger(__name__)


class GraphForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    task_type = "forecaster"
    supported_objectives = (DirectPredictionObjective, DiffusionObjective, FlowObjective)

    def _rollout_step(
        self,
        batch: dict,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the forecaster.

        Parameters
        ----------
        batch : dict
            Dictionary batch to use for rollout (assumed to be already preprocessed)
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        assert self.objective is not None, "GraphForecaster requires a training.objective configuration to be set."
        # start rollout of preprocessed batch
        rollout_steps = rollout or self.rollout
        required_time_steps = rollout_steps * self.multi_out + self.multi_step
        x = {}
        for dataset_name, dataset_batch in batch.items():
            # Conditioning input: last `multi_step` states (input variables only).
            x[dataset_name] = dataset_batch[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, multi_step, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested multi_step length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg

        for rollout_step in range(rollout_steps):
            y = {}
            for dataset_name, dataset_batch in batch.items():
                start = self.multi_step + rollout_step * self.multi_out
                y_time = dataset_batch.narrow(1, start, self.multi_out)
                var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
                # Target output window: full output variables for this rollout step.
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
            # Note: y is the clean target state; target is the objective-space
            # loss target (can differ for flow objectives).
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

            # Clean prediction/target in normalized state space for metrics/rollout.
            # y_pred_clean == y_pred, y == target for direct prediction objective.
            y_pred_clean, y_clean = self.objective.clean_pred_target_pair(y_pred, y, y_cond, schedule)

            metrics_y_pred = y_pred_clean if validation_mode else None
            metrics_y = y_clean if validation_mode else None
            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                target,
                step=rollout_step,
                validation_mode=validation_mode,
                metrics_y_pred=metrics_y_pred,
                metrics_y=metrics_y,
                pre_loss_weights=pre_loss_weights,
                use_reentrant=False,
            )

            # Advance input state for each dataset
            x = self._advance_input(x, y_pred_clean, batch, rollout_step=rollout_step)

            yield loss, metrics_next, y_pred_clean
