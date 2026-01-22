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


class GraphDownscaler(BaseGraphModule):
    """Graph neural network downscaler for PyTorch Lightning."""

    def get_inputs(self, batch: dict, sample_length: int) -> dict:
        # start rollout of preprocessed batch
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, multi_step, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested multi_step length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {sample_length}"
            )
            assert dataset_batch.shape[1] >= sample_length, msg
        return x

    def get_targets(self, batch: dict, lead_step: int) -> dict:
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y[dataset_name] = dataset_batch[
                :,
                lead_step,
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]
        return y

    def _step(
        self,
        batch: dict,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        x = self.get_inputs(batch, sample_length=self.multi_step)
        y = self.get_targets(batch, lead_step=self.multi_step - 1)

        y_pred = self(x)

        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred
