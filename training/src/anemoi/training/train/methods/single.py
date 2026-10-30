# (C) Copyright 2024-2026 Anemoi contributors.
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

from anemoi.models.data import Batch
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from anemoi.training.train.step_output import TrainingStepOutput

LOGGER = logging.getLogger(__name__)


class SingleTraining(BaseTrainingModule):
    """Base class for deterministic prediction tasks."""

    def _step(
        self,
        batch: Batch,
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Training / validation step."""
        step_losses, step_metrics, y_preds, step_weights = [], [], [], []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        x = self.preprocess_inputs(x)
        # The task may replace the input as the residual base (e.g. a DA forecaster's background).
        skip_input = self.task.build_skip_input(x, data_indices=self.data_indices)

        task_steps = self.task.steps("training" if not validation_mode else "validation")
        for step_index, task_kwargs in enumerate(task_steps):
            # the full target slice used for the loss, what the model predicts at the target nodes, and the
            # output-time forcing variables that condition the decoder.
            raw_targets, target_template, target_forcings = self.task.get_targets(
                batch,
                data_indices=self.data_indices,
                **task_kwargs,
            )
            y = self.preprocess_targets(raw_targets)

            # the target forcings are consumed by the decoder, so they are model *inputs* and go through
            # the input processors (so NaNs get imputed, etc.)
            target_forcings = self.preprocess_inputs(target_forcings)

            y_pred = self._forward_step(
                x,
                use_checkpoint=not validation_mode and self.task.step_uses_checkpoint(**task_kwargs),
                target_forcings=target_forcings,
                target_template=target_template,
                **({} if skip_input is None else {"skip_input": skip_input}),
            )

            weight = self.task.step_loss_weight(**task_kwargs)
            # The prediction of the target datasets, which advances the input state: always the raw one.
            raw_y_preds = Batch({name: y_pred[name] for name in self.target_dataset_names if name in y_pred})
            if weight > 0:
                y_pred_loss = self._loss_prediction(y_pred, y)
                loss_next, metrics_next, y_preds_next = checkpoint(
                    self.compute_loss_metrics,
                    y_pred_loss,
                    y,
                    **task_kwargs,
                    validation_mode=validation_mode,
                    pred_layout=IndexSpace.MODEL_OUTPUT,
                    target_layout=IndexSpace.DATA_FULL,
                    use_reentrant=False,
                )
                if y_pred_loss is not y_pred:
                    y_preds_next = raw_y_preds
            else:
                # A step without loss (e.g. a DA cycle): only its prediction is kept.
                loss_next, metrics_next, y_preds_next = None, {}, raw_y_preds

            # advance input state for each dataset, except on the final step
            if step_index < len(task_steps) - 1:
                observed = (
                    {"observed": self.preprocess_observations(raw_targets)}
                    if self.task.advance_needs_observations(**task_kwargs)
                    else {}
                )
                x = self.task.advance_input(
                    x,
                    y_preds_next,
                    self.preprocess_inputs(raw_targets),
                    **task_kwargs,
                    data_indices=self.data_indices,
                    output_mask=self.output_mask,
                    # The LAM boundary refill indexes the local grid shard.
                    grid_shard_slice={name: self._grid_shard_slice(view) for name, view in x.items()},
                    **observed,
                )
                skip_input = self.task.build_skip_input(x, y_preds_next, data_indices=self.data_indices, **task_kwargs)

            step_losses.append(loss_next)
            step_metrics.append(metrics_next)
            y_preds.append(y_preds_next)
            step_weights.append(weight)

        return self._combine_loss_and_metrics(step_losses, step_metrics, y_preds, step_weights)
