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

from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping


LOGGER = logging.getLogger(__name__)


class GraphAutoEncoder(BaseGraphModule):
    """Graph neural network autoencoder for PyTorch Lightning."""

    task_type = "autoencoder"

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
            x[dataset_name] = dataset_batch[
                :,
                0:required_time_steps,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]

        y_pred = self(x)

        y = {}

        for dataset_name, dataset_batch in batch.items():
            time_idx = torch.arange(self.multi_out, device=dataset_batch.device)
            y_time = dataset_batch.index_select(1, time_idx)
            var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = y_time.index_select(-1, var_idx)

        # y includes the auxiliary variables, so we must leave those out when computing the loss
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            rollout_step=0,
            training_mode=True,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred

    def on_train_epoch_end(self) -> None:
        pass
