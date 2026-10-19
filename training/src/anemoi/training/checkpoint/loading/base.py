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

    Subclasses implement ``process``, with some helpers.

    - ``_extract_state_dict``: Extract model state dict from checkpoint data
    - ``_mark_weights_loaded``: Flag the model as having loaded weights

    Examples
    --------
    >>> class TransferLearningLoader(LoadingStrategy):
    ...     async def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         state_dict = self._extract_state_dict(context)
    ...         # ... filter and apply compatible weights ...
    ...         preserve_anemoi_metadata(context.model, context.checkpoint_data)
    ...         self._mark_weights_loaded(context.model)
    ...         return context
    """

    #: If ``True``, Lightning does the full load through ``Trainer.fit(ckpt_path=)``.
    #: If ``False``, the strategy loads the weights and training state starts fresh.
    restores_training_state: bool = False

    @abstractmethod
    async def process(self, context: CheckpointContext) -> CheckpointContext:
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

    def _apply_corrections(self, context: CheckpointContext) -> None:
        """Run ``apply_checkpoint_corrections`` on ``context.checkpoint_data``."""
        if context.checkpoint_data is None:
            return
        context.checkpoint_data = apply_checkpoint_corrections(
            context.checkpoint_data,
            context.model,
            context.config,
            checkpoint_path=context.checkpoint_path,
        )

    def _mark_weights_loaded(self, model: nn.Module) -> None:
        """Set ``model.weights_initialized = True``.

        The pipeline raises if this flag is not set after loading (except on resume).
        Every strategy must call this.

        Parameters
        ----------
        model : nn.Module
            Model to mark.
        """
        model.weights_initialized = True
        LOGGER.debug("Marked model weights as initialized")


# Candidate import paths for the chunking_fix migration. Try the friendly
# dotted name first; fall back to the timestamp-prefixed module that
# anemoi-models currently ships
# (``1762857428_chunking_fix``). Both resolve to a ``migrate(ckpt)``
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


def _migrator() -> Any | None:
    """Return an anemoi-models ``Migrator``, or ``None`` if not available.

    The constructor imports all migration scripts, so it is inside the guard too.
    """
    try:
        from anemoi.models.migrations import Migrator

        return Migrator()
    except ImportError as exc:
        LOGGER.debug("anemoi-models migration ledger unavailable (%s); falling back to the in-memory path", exc)
        return None


def _has_migration_ledger(checkpoint: dict[str, Any]) -> bool:
    """Return ``True`` if the checkpoint has a non-empty migration ledger.

    Checkpoints older than migration tracking have no ledger.
    """
    try:
        return bool(checkpoint.get("migrations"))
    except AttributeError:  # pragma: no cover - checkpoint is always a mapping here
        return False


def _outstanding_migrations(migrator: Any, checkpoint: dict[str, Any]) -> list[str] | None:
    """Return the names of the migrations the checkpoint does not have, in order.

    Read-only. Compare the full ledger, not one migration name: the ledger records
    the migrations the writer knew, not the ones it applied.

    Parameters
    ----------
    migrator : Any
        An anemoi-models ``Migrator``.
    checkpoint : dict
        The checkpoint to read.

    Returns
    -------
    list[str] or None
        The missing migrations (empty if current), or ``None`` if the ledger
        cannot be read.
    """
    try:
        from anemoi.models.migrations import IncompatibleCheckpointException
    except ImportError:  # pragma: no cover - _migrator() already proved the import works
        IncompatibleCheckpointException = ()  # noqa: N806

    try:
        _setups, ops, _extra = migrator._resolve_migrations(checkpoint, migrator._grouped_migrations[-1])
    except (IncompatibleCheckpointException, AttributeError, KeyError, TypeError, IndexError) as exc:
        # IncompatibleCheckpointException is a BaseException, so name it explicitly.
        LOGGER.debug("Could not resolve the checkpoint migration ledger (%s); treating it as unknown", exc)
        return None
    return [op.migration.name for op in ops]


def _warn_unmigrated_checkpoint(outstanding: list[str], checkpoint_path: Any) -> None:
    """Warn that a checkpoint is behind the installed anemoi-models. Do not block.

    We do not migrate in-process: ``Migrator.sync`` copies the checkpoint many times
    in RAM and changes ``sys.modules``. We do not refuse: most migrations do not
    affect the load, and read-only or remote checkpoints cannot be migrated in place.
    A migration that is necessary still fails later, after this warning.

    Parameters
    ----------
    outstanding : list[str]
        Names of the missing migrations.
    checkpoint_path : Any
        Checkpoint file to name in the remedy, if there is one.
    """
    remedy = (
        f"`anemoi-models migration sync {checkpoint_path}` (note: this rewrites the checkpoint in "
        "place and writes a full-size backup beside it)"
        if checkpoint_path is not None
        else "downloading it, running `anemoi-models migration sync` on the copy, and pointing "
        "training.checkpoint.source at the result"
    )
    LOGGER.warning(
        "Checkpoint is behind the installed anemoi-models by %d migration(s): %s. Loading it anyway; "
        "most migrations do not affect the load. If this run fails with a checkpoint-format error, "
        "migrate it first: %s.",
        len(outstanding),
        ", ".join(outstanding),
        remedy,
    )


def _chunking_fix_applicable(checkpoint: dict[str, Any]) -> bool:
    """Return ``True`` if the processor config has valid ``num_layers`` and ``num_chunks``.

    ``chunking_fix`` needs both. Processors such as ``NoOpProcessor`` do not have them.
    """
    try:
        processor = checkpoint["hyper_parameters"]["config"].model.processor
    except (KeyError, AttributeError, TypeError):
        return False
    num_layers = getattr(processor, "num_layers", None)
    num_chunks = getattr(processor, "num_chunks", None)
    return isinstance(num_layers, int) and isinstance(num_chunks, int) and num_chunks > 0


# Import paths for the trainable_edge_perm_fix migration, ``migrate(ckpt, model)``.
# If neither imports (older anemoi-models), no migration is necessary.
_TRAINABLE_EDGE_PERM_PATHS = (
    "anemoi.models.migrations.scripts.trainable_edge_perm_fix",
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
)


def _load_trainable_edge_perm_migration() -> Any | None:
    """Resolve the ``trainable_edge_perm_fix.migrate`` callable from anemoi-models, or ``None``."""
    import importlib

    for path in _TRAINABLE_EDGE_PERM_PATHS:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        migrate = getattr(module, "migrate", None)
        if migrate is not None:
            return migrate
    LOGGER.debug("trainable_edge_perm migration not available in anemoi-models; skipping")
    return None


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


# ---------------------------------------------------------------------------
# Checkpoint-load corrections, shared by the loading strategies and
# ``on_load_checkpoint``.
# ---------------------------------------------------------------------------


def apply_checkpoint_corrections(
    checkpoint: dict[str, Any] | None,
    model: nn.Module | None,
    config: Any,
    *,
    checkpoint_path: Any = None,
) -> dict[str, Any] | None:
    """Make a checkpoint agree with the installed anemoi-models and the live model.

    The steps run in this order:

    1. ``apply_checkpoint_format_migrations`` (can replace the dict).
    2. ``apply_trainable_edge_perm_migration`` (can rebuild the state dict, so it
       runs before the refresh).
    3. ``refresh_checkpoint_processors``.

    Parameters
    ----------
    checkpoint : dict or None
        The loaded checkpoint. Returned unchanged if ``None``.
    model : nn.Module or None
        The training module. The refresh uses its inner ``.model``.
    config : Any
        The run config. Supplies ``training.update_ds_stats_on_ckpt_load``.
    checkpoint_path : Path or str, optional
        The checkpoint file, if there is one. Used in the migration warning.

    Returns
    -------
    dict or None
        The corrected checkpoint.
    """
    if checkpoint is None:
        return checkpoint

    checkpoint = apply_checkpoint_format_migrations(checkpoint, checkpoint_path)
    checkpoint = apply_trainable_edge_perm_migration(checkpoint, model)

    update_cfg = getattr(getattr(config, "training", None), "update_ds_stats_on_ckpt_load", None)
    refresh_checkpoint_processors(
        checkpoint,
        getattr(model, "model", None),
        update_states=bool(getattr(update_cfg, "states", False)),
        update_tendencies=bool(getattr(update_cfg, "tendencies", False)),
    )
    return checkpoint


def apply_checkpoint_format_migrations(
    checkpoint: dict[str, Any],
    checkpoint_path: Any = None,
) -> dict[str, Any]:
    """Apply format migrations, as the migration ledger requires.

    A checkpoint with a complete ledger is returned unchanged. If the ledger is
    incomplete, log a warning. If the ledger is incomplete or missing, apply
    ``chunking_fix`` in memory.

    Parameters
    ----------
    checkpoint : dict
        The loaded checkpoint. Returned unchanged if ``None``.
    checkpoint_path : Path or str, optional
        The checkpoint file, if there is one. Used in the warning.

    Returns
    -------
    dict
        The (possibly migrated) checkpoint.
    """
    if checkpoint is None:
        return checkpoint

    migrator = _migrator()
    # Only a Lightning training checkpoint with a ledger can be behind.
    # "pytorch-lightning_version" is the key ``Migrator._load_ckpt`` checks.
    if migrator is not None and "pytorch-lightning_version" in checkpoint and _has_migration_ledger(checkpoint):
        outstanding = _outstanding_migrations(migrator, checkpoint)
        if outstanding == []:
            LOGGER.debug("Checkpoint migration ledger is up to date; no format migration applied")
            return checkpoint
        if outstanding:
            _warn_unmigrated_checkpoint(outstanding, checkpoint_path)

    # No ledger, an unreadable ledger, or an incomplete ledger.
    return _apply_chunking_fix_in_memory(checkpoint)


def _apply_chunking_fix_in_memory(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Apply ``chunking_fix`` in memory, if it applies.

    Without it, a non-strict load silently loses the renamed processor keys.
    """
    migrate = _load_chunking_fix_migration()
    if migrate is None:
        return checkpoint
    if not _chunking_fix_applicable(checkpoint):
        LOGGER.info(
            "chunking_fix not applied: the checkpoint declares no usable processor num_layers/num_chunks "
            "(e.g. a NoOpProcessor autoencoder, or a raw state_dict save)",
        )
        return checkpoint

    LOGGER.info("Applying chunking_fix in memory; no checkpoint file available for a full ledger migration")
    try:
        return migrate(checkpoint)
    except (KeyError, AttributeError) as exc:
        # For checkpoints without some other key the migration reads.
        LOGGER.info("chunking_fix skipped: checkpoint shape incomplete (%s)", exc)
        return checkpoint


def apply_trainable_edge_perm_migration(checkpoint: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    """Apply the model-dependent trainable-edge-permutation migration.

    Skip it if the migration, checkpoint or model is missing, or if the checkpoint
    is incomplete.
    """
    if checkpoint is None or model is None:
        return checkpoint
    migrate = _load_trainable_edge_perm_migration()
    if migrate is None:
        return checkpoint
    try:
        return migrate(checkpoint, model)
    except (KeyError, AttributeError) as exc:
        LOGGER.debug("trainable_edge_perm migration skipped: checkpoint shape incomplete (%s)", exc)
        return checkpoint


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
    state_dict = checkpoint.get("state_dict") if checkpoint else None
    if not isinstance(state_dict, dict):
        return

    prefixes: tuple[str, ...] = ()
    if update_states:
        prefixes += ("model.pre_processors.", "model.post_processors.")
    if update_tendencies:
        prefixes += ("model.pre_processors_tendencies.", "model.post_processors_tendencies.")
    if not prefixes:
        return

    if model is None:
        LOGGER.warning(
            "Processor refresh requested (update_ds_stats_on_ckpt_load) but no model was given to "
            "re-inject from; keeping the checkpoint's processor entries unchanged",
        )
        return

    removed = _drop_keys_with_prefix(state_dict, prefixes)
    injected = _inject_model_weights(state_dict, model, prefixes)
    if removed and not injected:
        LOGGER.warning(
            "Processor refresh dropped %d checkpoint entries under %s but the model has none to "
            "re-inject; the loaded model keeps its own processor statistics",
            removed,
            prefixes,
        )
        return
    LOGGER.debug(
        "Refreshed checkpoint processors: removed %d stale entries, injected %d from current model",
        removed,
        injected,
    )


def preserve_anemoi_metadata(model: nn.Module, checkpoint_data: dict[str, Any]) -> None:
    """Restore ``model._ckpt_model_name_to_index`` from the checkpoint ``data_indices``.

    Raise ``TypeError`` for a single-dataset (pre-multi-dataset) checkpoint.
    """
    hyper_params = checkpoint_data.get("hyper_parameters", {})
    data_indices = hyper_params.get("data_indices")

    if isinstance(data_indices, dict) and data_indices:
        try:
            model._ckpt_model_name_to_index = {name: ic.name_to_index for name, ic in data_indices.items()}
        except AttributeError:
            LOGGER.debug("Multi-dataset data_indices entries lack .name_to_index; skipping restoration")
            return
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


def extract_checkpoint_variables_metadata(model: nn.Module, checkpoint_data: dict[str, Any] | None) -> None:
    """Set ``model._ckpt_variables_metadata`` from the checkpoint, if possible."""
    from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

    name_to_index = getattr(model, "_ckpt_model_name_to_index", None)
    if name_to_index is None or checkpoint_data is None:
        return
    model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(checkpoint_data, name_to_index)


def warn_on_hparams_divergence(checkpoint_data: dict[str, Any], run_config: Any) -> None:
    """Warn if the checkpoint model config is different from the run config.

    Best-effort. Matching tensor shapes can hide a different model config.
    """
    if run_config is None or checkpoint_data is None:
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
