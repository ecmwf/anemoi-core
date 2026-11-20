# (C) Copyright 2026- Anemoi contributors.
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

from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    import torch

    from anemoi.training.train.step_output import TrainingStepOutput

LOGGER = logging.getLogger(__name__)


class LatentRolloutTraining(SingleTraining):
    """Deterministic training with the rollout done in latent space.

    The inputs are encoded once per batch with ``model.encode``; every rollout
    step then calls ``model.step_latent``, which predicts from the latent state
    and returns the next one, so the model's own predictions are not re-encoded.
    The input state is still advanced as in :class:`SingleTraining`, because the
    decoder uses the current forcings and the residual connection.

    Needs a model that supports latent rollout (``supports_latent_rollout``),
    e.g. ``anemoi.models.models.AnemoiModelEncProcDec``: the processor then
    advances its own output at every step.
    """

    def _latent_model(self) -> torch.nn.Module:
        model = self.model.model
        if not getattr(model, "supports_latent_rollout", False):
            msg = f"{type(self).__name__} needs a model that supports latent rollout, got {type(model).__name__}."
            raise TypeError(msg)
        return model

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Training / validation step with latent rollout."""
        model = self._latent_model()
        sharding = {"model_comm_group": self.model_comm_group, "grid_shard_sizes": self.grid_shard_sizes}
        step_losses, step_metrics, y_preds = [], [], []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        state = model.encode(x, **sharding)

        task_steps = self.task.steps("training" if not validation_mode else "validation")
        for i, task_kwargs in enumerate(task_steps):
            y_pred, state = model.step_latent(state, x, **sharding)

            y = self.task.get_targets(batch, **task_kwargs)

            loss_next, metrics_next, y_preds_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                **task_kwargs,
                validation_mode=validation_mode,
                pred_layout=IndexSpace.MODEL_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
                use_reentrant=False,
            )

            # Advance the input state (forcings, boundaries, residual) if another step follows
            if i < len(task_steps) - 1:
                x = self.task.advance_input(
                    x,
                    y_preds_next,
                    batch,
                    **task_kwargs,
                    data_indices=self.data_indices,
                    output_mask=self.output_mask,
                    grid_shard_slice=self.grid_shard_slice,
                )

            step_losses.append(loss_next)
            step_metrics.append(metrics_next)
            y_preds.append(y_preds_next)

        return self._combine_loss_and_metrics(step_losses, step_metrics, y_preds)
