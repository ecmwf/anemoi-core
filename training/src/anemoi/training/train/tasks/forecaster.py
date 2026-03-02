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

from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch

LOGGER = logging.getLogger(__name__)


class GraphForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    task_type = "forecaster"

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
        # start rollout of preprocessed batch
        dataset_contexts = self._build_dataset_contexts()  # static only used here
        rollout_steps = rollout or self.rollout
        required_time_steps = rollout_steps * self.n_step_output + self.n_step_input
        x = {}
        for dataset_ctx in dataset_contexts.values():
            dataset_name = dataset_ctx.static.name
            dataset_batch = batch[dataset_name]
            x[dataset_name] = dataset_batch[
                :,
                0 : self.n_step_input,
                ...,
                dataset_ctx.static.data_indices.data.input.full,
            ]  # (bs, multi_step, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested n_step_input length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg

        for rollout_step in range(rollout_steps):
            y_pred = self(x)
            y = {}
            for dataset_ctx in dataset_contexts.values():
                dataset_name = dataset_ctx.static.name
                dataset_batch = batch[dataset_name]
                start = self.n_step_input + rollout_step * self.n_step_output
                y_time = dataset_batch.narrow(1, start, self.n_step_output)
                var_idx = dataset_ctx.static.data_indices.data.output.full.to(device=dataset_batch.device)
                y[dataset_name] = y_time.index_select(-1, var_idx)
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            # Compute loss for each dataset and sum them up
            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            # Advance input state for each dataset
            x = self._advance_input(x, y_pred, batch, rollout_step=rollout_step, dataset_contexts=dataset_contexts)

            yield loss, metrics_next, y_pred
