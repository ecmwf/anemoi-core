# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class BaseSingleStepGraphModule(BaseGraphModule, ABC):
    """Graph neural network autoencoder for PyTorch Lightning."""

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        required_time_steps = max(self.n_step_input, self.n_step_output)
        x = {}
        for dataset_name, dataset_batch in batch.items():
            msg = (
                f"Batch length not sufficient for requested n_step_input/n_step_output for {dataset_name}!"
                f" {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg
            x[dataset_name] = dataset_batch[
                :,
                0:required_time_steps,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, multi_step, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested multi_step length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {sample_length}"
            )
            assert dataset_batch.shape[1] >= sample_length, msg
        return x

    def get_targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y_time = dataset_batch.narrow(1, 0, self.n_step_output)
            var_indices = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = y_time.index_select(-1, var_indices)
        return y

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[dict[str, torch.Tensor], Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        x = self.get_inputs(batch, sample_length=self.multi_step)
        y = self.get_targets(batch, lead_step=self.multi_step - 1)

        y_pred = self(x)

        # y includes the auxiliary variables, so we must leave those out when computing the loss
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred


class GraphDownscaler(BaseSingleStepGraphModule):
    """Graph neural network downscaler for PyTorch Lightning."""

    task_type = "downscaler"


class GraphAutoEncoder(BaseSingleStepGraphModule):
    """Graph neural network autoencoder for PyTorch Lightning."""

    task_type = "autoencoder"

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        assert (
            self.n_step_input == self.n_step_output
        ), "Autoencoders must have the same number of input and output steps."
