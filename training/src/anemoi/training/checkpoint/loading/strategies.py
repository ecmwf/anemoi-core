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
    optimizer/scheduler state.

    Behavior
    --------
    - Loads weights with ``strict=self.strict`` (default ``True``)
    - Clears ``context.optimizer`` and ``context.scheduler`` to ``None``
    - **Leaves training-progress metadata untouched** (``epoch``,
      ``global_step``, ``best_metric``). A prior pipeline stage that set
      these values keeps them. If you want explicit zero-reset semantics,
      use :class:`ColdStartLoader`.

    Composes naturally inside larger pipelines where another stage owns
    training-progress state. For top-level "fresh training from pretrained
    weights" use :class:`ColdStartLoader` instead.

    Parameters
    ----------
    strict : bool, optional
        Whether to require an exact match between checkpoint keys and
        model keys (default: True). Missing keys raise ``CheckpointLoadError``.
    """

    def __init__(self, strict: bool = True) -> None:
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
        self._apply_format_migrations(context)
        self._refresh_checkpoint_processors(context)
        self._apply_trainable_edge_perm_migration(context)

        state_dict = self._extract_state_dict(context)

        try:
            context.model.load_state_dict(state_dict, strict=self.strict)
        except RuntimeError as e:
            raise CheckpointLoadError(context.checkpoint_path or "<in-memory checkpoint>", e) from e

        self._warn_on_hparams_divergence(context)
        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._extract_variables_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        # Discard optimizer/scheduler — weights-only means fresh training state
        context.optimizer = None
        context.scheduler = None

        context.metadata["loading_strategy"] = "weights_only"

        LOGGER.info("Loaded weights only (strict=%s), optimizer/scheduler discarded", self.strict)

        return context


class TransferLearningLoader(LoadingStrategy):
    """Flexible loading for transfer learning scenarios.

    Filters the source state dict to only include keys compatible with the
    target model (matching key names and tensor shapes), then loads the
    filtered weights. Keys that are missing in the target or have shape
    mismatches are skipped rather than raising an error.

    The filter is non-mutating: it builds a new dict and never modifies the
    original ``checkpoint_data["state_dict"]``.

    Parameters
    ----------
    skip_mismatched : bool, optional
        Whether to skip keys with mismatched shapes (default: True).
        If False, shape mismatches raise ``CheckpointIncompatibleError``.
    """

    def __init__(self, skip_mismatched: bool = True) -> None:
        self.skip_mismatched = skip_mismatched

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Filter and load compatible weights from checkpoint.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with compatible weights loaded and metadata updated.
        """
        from anemoi.training.checkpoint.loading.utils import filter_state_dict

        self._apply_format_migrations(context)
        self._refresh_checkpoint_processors(context)
        self._apply_trainable_edge_perm_migration(context)

        source_state = self._extract_state_dict(context)
        target_state = context.model.state_dict()

        filtered, skipped = filter_state_dict(source_state, target_state)

        if not self.skip_mismatched:
            shape_skipped = {k: v for k, v in skipped.items() if "Shape mismatch" in v}
            if shape_skipped:
                from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

                msg = f"Shape mismatches found and skip_mismatched=False: {shape_skipped}"
                raise CheckpointIncompatibleError(msg)

        try:
            context.model.load_state_dict(filtered, strict=False)
        except RuntimeError as e:
            raise CheckpointLoadError(context.checkpoint_path or "<in-memory checkpoint>", e) from e

        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._extract_variables_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        # Discard optimizer/scheduler — transfer learning means fresh training state
        context.optimizer = None
        context.scheduler = None

        context.metadata["loading_strategy"] = "transfer_learning"
        context.metadata["transferred_params"] = list(filtered.keys())
        context.metadata["skipped_params"] = skipped

        LOGGER.info(
            "Transfer learning: loaded %d params, skipped %d",
            len(filtered),
            len(skipped),
        )

        return context


class WarmStartLoader(LoadingStrategy):
    """Resume an interrupted training run on the same architecture.

    Loads model weights (``strict=True`` — an exact architecture match is
    expected when resuming) and applies the parity steps, exactly like the other
    strategies. The pipeline runs at model-build, before the optimizer, scheduler
    and Lightning fit-loop exist, so it cannot restore optimizer/scheduler/loop
    progress here.

    That runtime-state restore is owned by Lightning's ``ckpt_path`` resume, which
    runs at ``trainer.fit()`` once those objects exist. :attr:`restores_training_state`
    is ``True`` so the trainer keeps ``ckpt_path`` for this strategy (and only this
    one); for the weights-only / transfer / cold-start strategies ``ckpt_path`` is
    suppressed, since they start fresh training state.
    """

    restores_training_state = True

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load model weights (strict) and apply the parity steps.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded; optimizer/scheduler/loop progress are
            restored later by Lightning's ``ckpt_path`` (see the class docstring).
        """
        from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

        self._apply_format_migrations(context)
        self._refresh_checkpoint_processors(context)
        self._apply_trainable_edge_perm_migration(context)

        # Model weights (strict — exact match expected for resume)
        state_dict = self._extract_state_dict(context)
        try:
            context.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            msg = f"WarmStart requires exact model match: {e}"
            raise CheckpointIncompatibleError(msg) from e

        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._extract_variables_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)
        context.metadata["loading_strategy"] = "warm_start"

        LOGGER.info("Warm start: loaded weights; optimizer/epoch restore deferred to Lightning ckpt_path")

        return context


class ColdStartLoader(WeightsOnlyLoader):
    """Start fresh training from pretrained weights.

    Extends :class:`WeightsOnlyLoader` with the explicit "fresh training"
    contract that top-level callers usually want.

    When to use
    -----------
    Use this when you want pretrained weights but a clean training run:
    ``epoch`` and ``global_step`` are reset to zero regardless of what
    was in the checkpoint, and ``pretrained_from`` is recorded in metadata
    so downstream tooling (loggers, checkpoint naming) can trace the
    provenance.

    Use :class:`WeightsOnlyLoader` directly when composing pipelines where
    another stage owns training-progress metadata and you do not want to
    overwrite it.
    """

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load weights and reset training state to zero.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded and training state reset.
        """
        context = await super().process(context)

        context.metadata["epoch"] = 0
        context.metadata["global_step"] = 0
        context.metadata["loading_strategy"] = "cold_start"
        context.metadata["pretrained_from"] = str(context.checkpoint_path) if context.checkpoint_path else None

        LOGGER.info("Cold start: training state reset, pretrained from %s", context.checkpoint_path)

        return context
