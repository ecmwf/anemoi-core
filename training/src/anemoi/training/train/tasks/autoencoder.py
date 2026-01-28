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

from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


LOGGER = logging.getLogger(__name__)


class GraphAutoEncoder(BaseGraphModule):
    """Graph neural network autoencoder for PyTorch Lightning."""

    task_type = "autoencoder"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout = 1 # required for plotting callbacks rollout loops

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        x = {}

        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]

        y_pred = self(x)

        y = {}

        for dataset_name, dataset_batch in batch.items():
            print('dataset_batch', dataset_batch.shape)
            y[dataset_name] = dataset_batch[:, 0, ..., self.data_indices[dataset_name].data.output.full]
            print('y_pred',y_pred[dataset_name].shape)
            print('y',y[dataset_name].shape)
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
        return loss, metrics, [y_pred]

    def on_train_epoch_end(self) -> None:
        pass
