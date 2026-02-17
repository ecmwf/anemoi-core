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

    @property
    def _include_future_forcing(self) -> bool:
        return getattr(self.config.training, "include_future_forcing", False)

    def _get_future_forcing(self, batch: torch.Tensor, rollout_step: int) -> torch.Tensor | None:
        """Extract future forcing fields from batch at the target time step.

        Parameters
        ----------
        batch : torch.Tensor
            Full batch tensor (bs, time, ensemble, grid, vars)
        rollout_step : int
            Current rollout step index

        Returns
        -------
        torch.Tensor | None
            Future forcing tensor (bs, 1, ensemble, grid, n_forcing) or None if disabled
        """
        if not self._include_future_forcing:
            return None
        # Extract forcing fields at the target time step (t+6 relative to current input)
        future_forcing = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.data.input.forcing,
        ]  # (bs, ensemble, grid, n_forcing)
        return future_forcing.unsqueeze(1)  # (bs, 1, ensemble, grid, n_forcing)

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        # ... (docstring unchanged) ...

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            future_forcing = self._get_future_forcing(batch, rollout_step)  # <-- NEW
            y_pred = self(x, future_forcing=future_forcing)  # <-- CHANGED

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
            # ... rest unchanged ...
            LOGGER.debug("SHAPE: y.shape = %s", list(y.shape))
            # y includes the auxiliary variables, so we must leave those out when computing the loss

            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
