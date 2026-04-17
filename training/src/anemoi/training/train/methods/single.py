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

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


class SingleTraining(BaseTrainingModule):
    """Base class for deterministic prediction tasks."""

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

        task_steps = self.task.steps("training" if not validation_mode else "validation")
        for task_kwargs in task_steps:
            y_pred = self(x)
            y = self.task.get_targets(batch, data_indices=self.data_indices, **task_kwargs)

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

        loss *= 1.0 / len(task_steps)
        return loss, metrics, y_preds
