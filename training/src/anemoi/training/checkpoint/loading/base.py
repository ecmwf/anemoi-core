# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract base class for checkpoint loading strategies.

Loading strategies are responsible for applying checkpoint data to a
model. Different strategies handle different use cases: warm start
(resume training), cold start (fresh optimiser), transfer learning
(partial weight loading), and weights-only loading.

Wiring to the Lightning trainer
-------------------------------
This module defines the strategy contract; the actual
trainer-to-pipeline wiring (replacing the legacy
``model.load_from_checkpoint(...)`` call site and the
``AnemoiLightningModule.on_load_checkpoint`` hook) lands in Phase 3 of
the checkpoint refactor (Issue #495 / CheckpointManager). Until that
ships, strategies still need to perform every step that
``on_load_checkpoint`` did, so that an integration that simply forwards
the loaded model into Lightning does not silently regress.
``_apply_format_migrations`` bridges that gap: it mirrors the
``chunking_fix_migration(checkpoint)`` call in
``anemoi.training.utils.checkpoint.transfer_learning_loading``, and runs
at the top of every strategy's ``process()`` so the loaded state dict
matches what the legacy path would have produced.

Example
-------
>>> class WeightsOnlyLoader(LoadingStrategy):
...     async def process(self, context: CheckpointContext) -> CheckpointContext:
...         state_dict = self._extract_state_dict(context)
...         context.model.load_state_dict(state_dict, strict=False)
...         self._mark_weights_loaded(context.model)
...         return context
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    import torch.nn as nn

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class LoadingStrategy(PipelineStage):
    """Abstract base class for all checkpoint loading strategies.

    Loading strategies form the orchestration layer of the checkpoint
    pipeline. They receive a context with loaded checkpoint data and
    apply it to the model (and optionally optimiser/scheduler) according
    to a specific strategy.

    Subclasses must implement the ``process`` method. Several convenience
    methods are provided for common operations:

    - ``_extract_state_dict``: Extract model state dict from checkpoint data
    - ``_preserve_anemoi_metadata``: Preserve Anemoi-specific model metadata
    - ``_mark_weights_loaded``: Flag the model as having loaded weights

    Examples
    --------
    >>> class TransferLearningLoader(LoadingStrategy):
    ...     async def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         state_dict = self._extract_state_dict(context)
    ...         # ... filter and apply compatible weights ...
    ...         self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
    ...         self._mark_weights_loaded(context.model)
    ...         return context
    """

    @abstractmethod
    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Apply checkpoint data to the model using this strategy.

        Implementations should:
        1. Validate that required context fields are present
           (``checkpoint_data``, ``model``)
        2. Extract the state dict from checkpoint data
        3. Apply weights to the model according to the strategy
        4. Optionally restore optimiser/scheduler state
        5. Preserve Anemoi metadata and mark weights as loaded
        6. Update context metadata with loading results

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` populated by
            a prior source stage and ``model`` set to the target model.

        Returns
        -------
        CheckpointContext
            Context with model weights applied and metadata updated.

        Raises
        ------
        CheckpointLoadError
            If weights cannot be loaded into the model
        CheckpointIncompatibleError
            If the checkpoint is incompatible with the model architecture
        """

    def _extract_state_dict(self, context: CheckpointContext) -> dict[str, Any]:
        """Extract the model state dict from checkpoint data in context.

        Delegates to :func:`anemoi.training.checkpoint.formats.extract_state_dict`
        which handles various checkpoint structures (Lightning, PyTorch,
        raw state dicts).

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` populated

        Returns
        -------
        dict[str, Any]
            Extracted model state dictionary

        Raises
        ------
        CheckpointValidationError
            If no valid state dict can be found in checkpoint data
        """
        from anemoi.training.checkpoint.formats import extract_state_dict

        return extract_state_dict(context.checkpoint_data)

    def _preserve_anemoi_metadata(
        self,
        model: nn.Module,
        checkpoint_data: dict[str, Any],
    ) -> None:
        """Restore Anemoi-specific metadata from checkpoint onto model.

        Sets ``model._ckpt_model_name_to_index``, which maps variable names
        to their tensor indices and is consumed by diagnostics callbacks
        (sanity checks) and downstream inference.

        Two checkpoint shapes are supported, matching the production hook
        in ``anemoi.training.train.tasks.base.AnemoiLightningModule.on_load_checkpoint``:

        - **Multi-dataset** (current): ``hyper_parameters["data_indices"]``
          is a ``dict[str, IndexCollection]`` keyed by dataset name. The
          attribute becomes ``{dataset_name: ic.name_to_index for ...}``,
          which matches what ``DiagnosticsSanityCallback`` indexes by
          dataset name.
        - **Single-dataset** (legacy): ``hyper_parameters["data_indices"]``
          is a single ``IndexCollection`` with a ``.name_to_index``
          attribute. The flat mapping is assigned unchanged.

        If neither shape is recognised, the attribute is left alone and
        a debug message is logged.

        Parameters
        ----------
        model : nn.Module
            Target model to attach metadata to
        checkpoint_data : dict
            Full checkpoint data dictionary (not just the state dict)
        """
        hyper_params = checkpoint_data.get("hyper_parameters", {})
        data_indices = hyper_params.get("data_indices")

        if isinstance(data_indices, dict) and data_indices:
            try:
                model._ckpt_model_name_to_index = {name: ic.name_to_index for name, ic in data_indices.items()}
            except AttributeError:
                LOGGER.debug(
                    "Multi-dataset data_indices entries lack .name_to_index; skipping restoration",
                )
                return
            LOGGER.debug(
                "Restored multi-dataset _ckpt_model_name_to_index for %d datasets",
                len(data_indices),
            )
            return

        if data_indices is not None and hasattr(data_indices, "name_to_index"):
            model._ckpt_model_name_to_index = data_indices.name_to_index
            LOGGER.debug("Restored single-dataset _ckpt_model_name_to_index from checkpoint hyper_parameters")
            return

        LOGGER.debug(
            "Checkpoint does not contain hyper_parameters.data_indices.name_to_index; "
            "skipping _ckpt_model_name_to_index restoration",
        )

    def _apply_format_migrations(self, context: CheckpointContext) -> None:
        """Run anemoi-models checkpoint-format migrations against ``context.checkpoint_data``.

        Mirrors the legacy ``chunking_fix_migration(checkpoint)`` call in
        ``anemoi.training.utils.checkpoint.transfer_learning_loading`` so
        old checkpoints with the pre-chunking attention head layout get
        rewritten before any ``load_state_dict`` attempt. Without this,
        :func:`~anemoi.training.checkpoint.loading.utils.filter_state_dict`
        silently drops the affected weights as shape mismatches.

        Importing the migration is best-effort: anemoi-models versions
        that predate the migration module are tolerated as a no-op, so
        the pipeline does not hard-require a specific anemoi-models
        version.

        Mutates ``context.checkpoint_data`` in place (the migration
        returns the rewritten dict; we reassign it onto the context).

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` set
        """
        if context.checkpoint_data is None:
            return

        migrate = _load_chunking_fix_migration()
        if migrate is None:
            return

        try:
            context.checkpoint_data = migrate(context.checkpoint_data)
        except (KeyError, AttributeError) as exc:
            # The migration reads ``ckpt["hyper_parameters"]["config"].model.processor``
            # unconditionally. Test fixtures and minimally-built checkpoints (raw
            # state_dict saves) lack that tree, so the access raises KeyError or
            # AttributeError. Treat that as "nothing to migrate" rather than
            # propagating — real anemoi checkpoints always have the structure.
            LOGGER.debug("chunking_fix migration skipped: checkpoint shape incomplete (%s)", exc)
            return
        LOGGER.debug("Applied chunking_fix migration to checkpoint data")

    def _mark_weights_loaded(self, model: nn.Module) -> None:
        """Mark the model as having successfully loaded weights.

        Sets ``model.weights_initialized = True``. Downstream checks
        (``anemoi.training.checkpoint.pipeline.CheckpointPipeline`` at
        line 501 and ``anemoi.training.checkpoint.validation`` at
        line 297) read this attribute to decide whether to warn that
        a source stage ran without a loading strategy ever applying
        weights.

        The flag is a **hint, not a gate**: ``getattr(model,
        "weights_initialized", False)`` is treated as "no loading
        strategy executed" and only triggers a warning log. Code that
        forgets to call ``_mark_weights_loaded`` still runs, it just
        produces noisier output. Tests that bypass strategies entirely
        therefore do not need to set the attribute.

        Parameters
        ----------
        model : nn.Module
            Model to mark as weight-loaded
        """
        model.weights_initialized = True
        LOGGER.debug("Marked model weights as initialized")


# Candidate import paths for the chunking_fix migration. Try the friendly
# dotted name first; fall back to the timestamp-prefixed module that
# anemoi-models currently ships
# (``1762857428_chunking_fix``, also used by the legacy import in
# ``anemoi.training.utils.checkpoint``). Both resolve to a ``migrate(ckpt)``
# function. Returns ``None`` if neither path is importable, which we treat
# as "no chunking migration needed in this anemoi-models version".
_CHUNKING_FIX_PATHS = (
    "anemoi.models.migrations.scripts.chunking_fix",
    "anemoi.models.migrations.scripts.1762857428_chunking_fix",
)


def _load_chunking_fix_migration() -> Any | None:
    """Resolve the ``chunking_fix.migrate`` callable from anemoi-models, or ``None``."""
    import importlib

    for path in _CHUNKING_FIX_PATHS:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        migrate = getattr(module, "migrate", None)
        if migrate is not None:
            return migrate
    LOGGER.debug("chunking_fix migration not available in anemoi-models; skipping")
    return None
