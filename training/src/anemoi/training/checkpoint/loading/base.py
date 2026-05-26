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

    @staticmethod
    def _processor_prefixes_from_config(context: CheckpointContext) -> tuple[str, ...]:
        """Return the processor key prefixes to refresh, based on context.config.

        Reads ``config.training.update_ds_stats_on_ckpt_load.{states,tendencies}``
        defensively (any missing layer yields an empty tuple, i.e. no refresh).
        """
        update_cfg = getattr(
            getattr(getattr(context, "config", None), "training", None),
            "update_ds_stats_on_ckpt_load",
            None,
        )
        if update_cfg is None:
            return ()

        prefixes: tuple[str, ...] = ()
        if bool(getattr(update_cfg, "states", False)):
            prefixes += ("model.pre_processors.", "model.post_processors.")
        if bool(getattr(update_cfg, "tendencies", False)):
            prefixes += ("model.pre_processors_tendencies.", "model.post_processors_tendencies.")
        return prefixes

    def _refresh_checkpoint_processors(self, context: CheckpointContext) -> None:
        """Replace pre/post processor weights in the checkpoint with the current model's.

        Mirrors
        ``anemoi.training.train.tasks.base.AnemoiLightningModule._update_checkpoint_state_dict_for_load``
        so pipeline-based loading honours
        ``config.training.update_ds_stats_on_ckpt_load.{states,tendencies}``.
        Without this, users with the default ``tendencies: True`` config
        would load stale processor stats from the checkpoint instead of
        rebuilding them from the current dataset.

        Mutates ``context.checkpoint_data["state_dict"]`` in place (the
        whole point: the checkpoint payload is rewritten so the
        subsequent ``load_state_dict`` call picks up fresh processor
        weights). No-op when ``context.config`` is missing or the flags
        are both false.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data``, ``model`` and
            optionally ``config`` set
        """
        prefixes = self._processor_prefixes_from_config(context)
        if not prefixes:
            return

        state_dict = context.checkpoint_data.get("state_dict") if context.checkpoint_data else None
        if not isinstance(state_dict, dict):
            return

        removed = _drop_keys_with_prefix(state_dict, prefixes)
        injected = _inject_model_weights(state_dict, context.model, prefixes) if context.model is not None else 0

        LOGGER.debug(
            "Refreshed checkpoint processors: removed %d stale entries, injected %d from current model",
            removed,
            injected,
        )

    def _mark_weights_loaded(self, model: nn.Module) -> None:
        """Mark the model as having successfully loaded weights.

        Sets ``model.weights_initialized = True`` which downstream
        components can check to determine whether checkpoint weights
        have been applied.

        Parameters
        ----------
        model : nn.Module
            Model to mark as weight-loaded
        """
        model.weights_initialized = True
        LOGGER.debug("Marked model weights as initialized")


def _drop_keys_with_prefix(state_dict: dict[str, Any], prefixes: tuple[str, ...]) -> int:
    """Remove every key in ``state_dict`` starting with one of ``prefixes``; return count."""
    to_remove = [key for key in state_dict if key.startswith(prefixes)]
    for key in to_remove:
        del state_dict[key]
    return len(to_remove)


def _inject_model_weights(
    state_dict: dict[str, Any],
    model: nn.Module,
    prefixes: tuple[str, ...],
) -> int:
    """Copy model parameters into ``state_dict`` under ``model.<key>``; return count injected."""
    injected = 0
    for key, value in model.state_dict().items():
        full_key = f"model.{key}"
        if full_key.startswith(prefixes):
            state_dict[full_key] = value
            injected += 1
    return injected
