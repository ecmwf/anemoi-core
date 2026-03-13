# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Concrete loading strategy implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.base import LoadingStrategy

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class WeightsOnlyLoader(LoadingStrategy):
    """Load only model weights, discarding optimizer and scheduler state.

    This is the simplest loading strategy: extract the state dict from
    checkpoint data, load it into the model, and explicitly discard any
    optimizer/scheduler state. Useful for cold-start scenarios where you
    want pretrained weights but a fresh optimizer.

    Parameters
    ----------
    strict : bool, optional
        Whether to require an exact match between checkpoint keys and
        model keys (default: False)
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load weights into model, discard optimizer/scheduler.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded and optimizer/scheduler cleared.
        """
        state_dict = self._extract_state_dict(context)

        try:
            context.model.load_state_dict(state_dict, strict=self.strict)
        except RuntimeError as e:
            msg = f"Failed to load state dict into model: {e}"
            raise CheckpointLoadError(msg) from e

        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        # Discard optimizer/scheduler — weights-only means fresh training state
        context.optimizer = None
        context.scheduler = None

        context.metadata["loading_strategy"] = "weights_only"

        LOGGER.info("Loaded weights only (strict=%s), optimizer/scheduler discarded", self.strict)

        return context
