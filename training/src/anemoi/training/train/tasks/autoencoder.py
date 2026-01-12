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
    """Graph neural network forecaster for PyTorch Lightning."""

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        x = batch[
            ...,
            self.data_indices.data.input.full,
        ]

        y_pred = self(x)
        y = batch[:, 0, ..., self.data_indices.data.output.full]

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
