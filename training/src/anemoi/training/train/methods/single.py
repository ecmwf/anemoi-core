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

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from anemoi.models.data import Batch

LOGGER = logging.getLogger(__name__)


class SingleTraining(BaseTrainingModule):
    """Base class for deterministic prediction tasks."""

    def _step(
        self,
        batch: Batch,
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Training / validation step."""
        first_payload = next(iter(batch.data.values()))
        dtype = first_payload[0].dtype if isinstance(first_payload, list) else first_payload.dtype
        loss = torch.zeros(1, dtype=dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x = self.preprocess_inputs(self.task.get_inputs(batch, data_indices=self.data_indices))

        task_steps = self.task.steps("training" if not validation_mode else "validation")
        for step_index, task_kwargs in enumerate(task_steps):
            # get_targets returns (targets, target_forcings): the full target slice used for the
            # loss, and the output-time forcing variables that condition the decoder.
            raw_targets, target_forcings = self.task.get_targets(
                batch,
                data_indices=self.data_indices,
                **task_kwargs,
            )
            y = self.preprocess_targets(raw_targets)
            # the target forcings are consumed by the decoder, so they are model *inputs* and go through
            # the input processors (so NaNs get imputed, etc.)
            target_forcings = self.preprocess_inputs(target_forcings)

            y_pred = self(x, target=target_forcings)

            loss_next, metrics_next, y_preds_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                **task_kwargs,
                validation_mode=validation_mode,
                num_task_steps=len(task_steps),
                pred_layout=IndexSpace.MODEL_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
                use_reentrant=False,
            )

            # advance input state for each dataset, except on the final step
            if step_index < len(task_steps) - 1:
                x = self.task.advance_input(
                    x,
                    y_preds_next,
                    self.preprocess_inputs(raw_targets),
                    **task_kwargs,
                    data_indices=self.data_indices,
                    output_mask=self.output_mask,
                )

            loss = loss + loss_next
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)

        loss *= 1.0 / len(task_steps)
        return TrainingStepOutput(loss=loss, metrics=metrics, predictions=y_preds)
