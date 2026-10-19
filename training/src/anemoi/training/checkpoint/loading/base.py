# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract base class for checkpoint loading strategies.

A loading strategy applies checkpoint data to a model.

Wiring to the Lightning trainer
-------------------------------
Only a resume goes through Lightning. On all other paths the strategy applies
the load corrections itself. Both use ``apply_checkpoint_corrections``.

Example
-------
>>> class WeightsOnlyLoader(LoadingStrategy):
...     def process(self, context: CheckpointContext) -> CheckpointContext:
...         state_dict = self._extract_state_dict(context)
...         context.model.load_state_dict(state_dict, strict=False)
...         context.model.weights_initialized = True
...         return context
"""

from __future__ import annotations

import importlib
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

    Subclasses implement ``process``. Use ``_extract_state_dict`` to get the
    model state dict. After the load, set ``model.weights_initialized = True``:
    the pipeline raises if it is not set (except on resume).
    """

    #: If ``True``, Lightning does the full load through ``Trainer.fit(ckpt_path=)``.
    #: If ``False``, the strategy loads the weights and training state starts fresh.
    restores_training_state: bool = False

    @abstractmethod
    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Apply checkpoint data to the model using this strategy.

        Implementations should:
        1. Validate that required context fields are present
           (``checkpoint_data``, ``model``)
        2. Extract the state dict from checkpoint data
        3. Apply weights to the model according to the strategy
        4. Preserve Anemoi metadata and mark weights as loaded
        5. Update context metadata with loading results

        Parameters
        ----------
        context : CheckpointContext
            Context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with model weights applied and metadata updated.

        Raises
        ------
        CheckpointLoadError
            If weights cannot be loaded into the model.
        CheckpointIncompatibleError
            If the checkpoint is incompatible with the model architecture.
        """

    def _extract_state_dict(self, context: CheckpointContext) -> dict[str, Any]:
        """Extract the model state dict from ``context.checkpoint_data``.

        Parameters
        ----------
        context : CheckpointContext
            Context with ``checkpoint_data`` set.

        Returns
        -------
        dict[str, Any]
            Extracted model state dictionary.

        Raises
        ------
        CheckpointLoadError
            If the source deferred its error to rank 0.
        CheckpointValidationError
            If no valid state dict can be found in checkpoint data.
        """
        from anemoi.training.checkpoint.formats import extract_state_dict

        self._abort_if_source_deferred(context)
        return extract_state_dict(context.checkpoint_data)

    @staticmethod
    def _abort_if_source_deferred(context: CheckpointContext) -> None:
        """Raise a clear error if the source deferred its error to rank 0.

        Without this, the other ranks report a misleading "corrupted checkpoint" error.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context.

        Raises
        ------
        CheckpointLoadError
            If the source deferred.
        """
        if not context.metadata.get("source_deferred"):
            return

        from anemoi.training.checkpoint.exceptions import CheckpointLoadError

        reason = context.metadata.get("source_deferred_reason", "the checkpoint could not be acquired")
        msg = (
            f"The checkpoint source could not provide a checkpoint on this rank ({reason}), so there is "
            "nothing to load. This is not a corrupted checkpoint: the source deferred the error to rank 0, "
            "which reports the actionable message. Look at rank 0's log."
        )
        LOGGER.error(msg)
        raise CheckpointLoadError(context.checkpoint_path or "<deferred to rank 0>", RuntimeError(msg))


_chunking_fix = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate
_trainable_edge_perm_fix = importlib.import_module(
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
).migrate


def _chunking_fix_applicable(checkpoint: dict[str, Any]) -> bool:
    """Return ``True`` if ``chunking_fix`` can run on the checkpoint.

    ``chunking_fix`` needs a state dict and integer ``num_layers`` and ``num_chunks > 0``.
    Processors such as ``NoOpProcessor`` do not have them.
    """
    try:
        processor = checkpoint["hyper_parameters"]["config"].model.processor
    except (KeyError, AttributeError, TypeError):
        return False
    num_layers = getattr(processor, "num_layers", None)
    num_chunks = getattr(processor, "num_chunks", None)
    return "state_dict" in checkpoint and isinstance(num_layers, int) and isinstance(num_chunks, int) and num_chunks > 0


def _inject_model_weights(
    state_dict: dict[str, Any],
    model: nn.Module,
    prefixes: tuple[str, ...],
) -> int:
    """Copy model parameters into ``state_dict`` under ``model.<key>``; return count.

    Also copy all ``model_output_idx`` buffers from the live model.
    """
    model_state_dict = model.state_dict()
    effective_prefixes = prefixes + tuple(f"model.{key}" for key in model_state_dict if "model_output_idx" in key)
    injected = 0
    for key, value in model_state_dict.items():
        full_key = f"model.{key}"
        if full_key.startswith(effective_prefixes):
            state_dict[full_key] = value
            injected += 1
    return injected


def apply_checkpoint_corrections(checkpoint: dict[str, Any], model: nn.Module | None, config: Any) -> None:
    """Make a checkpoint agree with the installed anemoi-models and the live model, in place.

    The steps run in this order:

    1. ``chunking_fix``, if the processor config supports it.
    2. ``trainable_edge_perm_fix``.
    3. ``refresh_checkpoint_processors``.

    The migrations are idempotent. They change only old-format entries.

    Parameters
    ----------
    checkpoint : dict
        The loaded checkpoint.
    model : nn.Module or None
        The training module. The refresh uses its inner ``.model``.
    config : Any
        The run config. Supplies ``training.update_ds_stats_on_ckpt_load``.
    """
    if _chunking_fix_applicable(checkpoint):
        _chunking_fix(checkpoint)
    _trainable_edge_perm_fix(checkpoint, model)

    update_cfg = getattr(getattr(config, "training", None), "update_ds_stats_on_ckpt_load", None)
    refresh_checkpoint_processors(
        checkpoint,
        getattr(model, "model", None),
        update_states=bool(getattr(update_cfg, "states", False)),
        update_tendencies=bool(getattr(update_cfg, "tendencies", False)),
    )


def refresh_checkpoint_processors(
    checkpoint: dict[str, Any],
    model: nn.Module | None,
    *,
    update_states: bool,
    update_tendencies: bool,
) -> None:
    """Replace the processor weights in ``checkpoint['state_dict']`` with the model's.

    Changes the state dict in place. If ``model`` is ``None``, nothing is dropped,
    so a strict load still has the processor entries.
    """
    if not (update_states or update_tendencies):
        return
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        return

    prefixes: tuple[str, ...] = ()
    if update_states:
        prefixes += ("model.pre_processors.", "model.post_processors.")
    if update_tendencies:
        prefixes += ("model.pre_processors_tendencies.", "model.post_processors_tendencies.")

    if model is None:
        LOGGER.warning(
            "Processor refresh requested (update_ds_stats_on_ckpt_load) but no model was given to "
            "re-inject from; keeping the checkpoint's processor entries unchanged",
        )
        return

    stale = [key for key in state_dict if key.startswith(prefixes)]
    for key in stale:
        del state_dict[key]
    injected = _inject_model_weights(state_dict, model, prefixes)
    if stale and not injected:
        LOGGER.warning(
            "Processor refresh dropped %d checkpoint entries under %s but the model has none to "
            "re-inject; the loaded model keeps its own processor statistics",
            len(stale),
            prefixes,
        )
        return
    LOGGER.debug(
        "Refreshed checkpoint processors: removed %d stale entries, injected %d from current model",
        len(stale),
        injected,
    )


def preserve_anemoi_metadata(model: nn.Module, checkpoint_data: dict[str, Any]) -> None:
    """Set ``model._ckpt_model_name_to_index`` and ``model._ckpt_variables_metadata`` from the checkpoint.

    Raise ``TypeError`` for a single-dataset (pre-multi-dataset) checkpoint.
    """
    from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

    data_indices = checkpoint_data.get("hyper_parameters", {}).get("data_indices")

    if isinstance(data_indices, dict) and data_indices:
        try:
            name_to_index = {name: ic.name_to_index for name, ic in data_indices.items()}
        except AttributeError:
            LOGGER.debug("Multi-dataset data_indices entries lack .name_to_index; skipping restoration")
            return
        model._ckpt_model_name_to_index = name_to_index
        model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(checkpoint_data, name_to_index)
        LOGGER.debug("Restored multi-dataset _ckpt_model_name_to_index for %d datasets", len(data_indices))
        return

    if data_indices is not None and hasattr(data_indices, "name_to_index"):
        # A flat mapping silently breaks the dataset-keyed lookups, so reject it.
        msg = (
            "Checkpoint hyper_parameters.data_indices is a single-dataset IndexCollection "
            "from a pre-multi-dataset anemoi-core, incompatible with the current loaders. "
            "Run `anemoi-models migration sync <checkpoint>` to upgrade it to the "
            "multi-dataset format, or re-export it with current anemoi-core."
        )
        raise TypeError(msg)

    LOGGER.debug(
        "Checkpoint does not contain hyper_parameters.data_indices.name_to_index; "
        "skipping _ckpt_model_name_to_index restoration",
    )


def warn_on_hparams_divergence(checkpoint_data: dict[str, Any], run_config: Any) -> None:
    """Warn if the checkpoint model config is different from the run config.

    Best-effort. Matching tensor shapes can hide a different model config.
    """
    if run_config is None:
        return

    hyper_params = checkpoint_data.get("hyper_parameters")
    if not isinstance(hyper_params, dict):
        return
    ckpt_config = hyper_params.get("config")
    if ckpt_config is None:
        return

    from omegaconf import OmegaConf

    try:
        ckpt_model = OmegaConf.to_container(OmegaConf.create(ckpt_config), resolve=True).get("model")
        run_model = OmegaConf.to_container(OmegaConf.create(run_config), resolve=True).get("model")
    except (ValueError, TypeError, AttributeError):
        return

    if ckpt_model is not None and ckpt_model != run_model:
        LOGGER.warning(
            "Checkpoint hparams differ from the run config (checkpoint "
            "hyper_parameters.config.model != training config model); fill-model loading "
            "keeps the current architecture. Verify the checkpoint matches this run.",
        )
