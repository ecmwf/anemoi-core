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
This module defines the strategy contract. The trainer-to-pipeline wiring has
shipped (Issue #495): ``AnemoiTrainer.model`` builds and runs the pipeline at
model-construction time, so a configured ``training.checkpoint`` loading
strategy owns weight loading. For the weights-only, transfer-learning and
cold-start paths Lightning never sees the checkpoint (its ``ckpt_path`` restore
is suppressed), so each strategy must itself perform every step that
``AnemoiLightningModule.on_load_checkpoint`` would have done — otherwise the
loaded state dict would silently differ from the legacy path.

Those parity steps live as shared, context-free functions at the bottom of this
module (:func:`apply_checkpoint_format_migrations`,
:func:`apply_trainable_edge_perm_migration`,
:func:`refresh_checkpoint_processors`, :func:`preserve_anemoi_metadata`,
:func:`extract_checkpoint_variables_metadata`,
:func:`warn_on_hparams_divergence`). The ``LoadingStrategy._*`` methods are thin
wrappers over them, and ``AnemoiLightningModule.on_load_checkpoint`` calls the
same functions, so the algorithm lives in exactly one place. Warm start keeps
Lightning's ``ckpt_path`` restore (to recover optimizer/epoch state), so
``on_load_checkpoint`` runs the shared steps for that path while the loading
strategies run them for the non-Lightning paths.

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

    #: Whether this strategy needs the optimizer / scheduler / loop-progress state
    #: restored at fit time. The pipeline always applies weights + parity at
    #: model-build, but cannot restore optimizer/loop state (those objects exist
    #: only once ``trainer.fit()`` starts). Strategies that resume an interrupted
    #: run set this ``True`` so the trainer keeps Lightning's ``ckpt_path`` resume,
    #: which owns that runtime-state restore. Fresh-training strategies leave it
    #: ``False`` (the default) so ``ckpt_path`` is suppressed and no second load
    #: happens.
    restores_training_state: bool = False

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
        """Restore Anemoi metadata onto the model.

        Thin wrapper over :func:`preserve_anemoi_metadata` (the shared parity
        home, also used by ``AnemoiLightningModule.on_load_checkpoint``).
        """
        preserve_anemoi_metadata(model, checkpoint_data)

    def _apply_format_migrations(self, context: CheckpointContext) -> None:
        """Bring ``context.checkpoint_data`` up to date with the migration ledger.

        Thin wrapper over :func:`apply_checkpoint_format_migrations`; reassigns the
        (possibly migrated) checkpoint onto the context. ``context.checkpoint_path``
        is forwarded so an on-disk checkpoint gets the full ledger-driven migration
        rather than the in-memory ``chunking_fix`` fallback.
        """
        if context.checkpoint_data is None:
            return
        context.checkpoint_data = apply_checkpoint_format_migrations(
            context.checkpoint_data,
            context.checkpoint_path,
        )

    def _refresh_checkpoint_processors(self, context: CheckpointContext) -> None:
        """Replace pre/post processor weights in the checkpoint with the current model's.

        Thin wrapper over :func:`refresh_checkpoint_processors`; reads
        ``config.training.update_ds_stats_on_ckpt_load.{states,tendencies}``
        defensively (a missing config layer disables the refresh).
        """
        if context.checkpoint_data is None:
            return
        update_cfg = getattr(
            getattr(getattr(context, "config", None), "training", None),
            "update_ds_stats_on_ckpt_load",
            None,
        )
        if update_cfg is None:
            return
        # refresh_checkpoint_processors / _inject_model_weights operate on the inner
        # AnemoiModelInterface: its state_dict keys lack the LightningModule's leading
        # ``model.`` (which _inject_model_weights re-adds), so they re-inject at the
        # correct ``model.<processor>...`` level. This matches the legacy contract
        # (on_load_checkpoint passes ``self.model``). ``context.model`` is the
        # LightningModule, so pass its inner ``.model``; passing the LightningModule
        # itself double-prefixes to ``model.model.*`` and breaks the strict warm-start
        # load. ``getattr`` keeps the no-op path safe when no model is set.
        inner_model = getattr(context.model, "model", None)
        refresh_checkpoint_processors(
            context.checkpoint_data,
            inner_model,
            update_states=bool(getattr(update_cfg, "states", False)),
            update_tendencies=bool(getattr(update_cfg, "tendencies", False)),
        )

    def _mark_weights_loaded(self, model: nn.Module) -> None:
        """Mark the model as having successfully loaded weights.

        Sets ``model.weights_initialized = True``. Downstream checks read
        this attribute to detect a source stage that ran without any
        loading strategy applying weights. Two readers treat it differently:

        - ``CheckpointPipeline._verify_weights_loaded`` treats it as a
          **gate**: if a source stage was configured but the flag is False,
          it raises ``CheckpointLoadError`` rather than train on random
          weights.
        - ``validation.validate_pipeline_health`` treats it as a health
          **finding** (collected into the issues list).

        A strategy that forgets to call ``_mark_weights_loaded`` after a
        source has run will therefore trip the pipeline gate. Tests that
        construct a context without a real source stage are unaffected.

        Parameters
        ----------
        model : nn.Module
            Model to mark as weight-loaded
        """
        model.weights_initialized = True
        LOGGER.debug("Marked model weights as initialized")

    def _apply_trainable_edge_perm_migration(self, context: CheckpointContext) -> None:
        """Apply the runtime trainable-edge-permutation migration to the checkpoint.

        Thin wrapper over :func:`apply_trainable_edge_perm_migration`; reassigns the
        (possibly rewritten) checkpoint onto the context.
        """
        if context.checkpoint_data is None or context.model is None:
            return
        context.checkpoint_data = apply_trainable_edge_perm_migration(context.checkpoint_data, context.model)

    def _extract_variables_metadata(self, model: nn.Module, checkpoint_data: dict[str, Any]) -> None:
        """Populate ``model._ckpt_variables_metadata`` from the checkpoint.

        Thin wrapper over :func:`extract_checkpoint_variables_metadata`.
        """
        extract_checkpoint_variables_metadata(model, checkpoint_data)

    def _warn_on_hparams_divergence(self, context: CheckpointContext) -> None:
        """Warn when the checkpoint's stored hyper-parameters differ from the run config.

        Thin wrapper over :func:`warn_on_hparams_divergence`.
        """
        warn_on_hparams_divergence(context.checkpoint_data, context.config)


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


def _migrator() -> Any | None:
    """Return an anemoi-models ``Migrator``, or ``None`` when unavailable.

    Constructing a ``Migrator`` imports every migration script anemoi-models
    ships, so instantiation is inside the guard too: an anemoi-models without the
    migration framework, or with an unimportable scripts package, degrades to
    ``None`` instead of failing the checkpoint load.
    """
    try:
        from anemoi.models.migrations import Migrator

        return Migrator()
    except ImportError as exc:
        LOGGER.debug("anemoi-models migration ledger unavailable (%s); falling back to the in-memory path", exc)
        return None


def _ledger_records_chunking_fix(migrator: Any, checkpoint: dict[str, Any]) -> bool:
    """Whether the checkpoint's own migration ledger already records ``chunking_fix``.

    Every checkpoint anemoi writes carries a ledger of the migrations applied to it
    (stamped on save by ``RegisterMigrations``). ``Migrator.registered_migrations``
    reads it and returns an empty list for checkpoints that carry none, so this is
    safe on any dict shape. A checkpoint that already records ``chunking_fix`` needs
    no format migration, so we do no work and read no files — the common case for
    anything written by a current anemoi.
    """
    try:
        return any(m.name.endswith("chunking_fix") for m in migrator.registered_migrations(checkpoint))
    except (KeyError, AttributeError, TypeError) as exc:
        LOGGER.debug("Could not read the checkpoint migration ledger (%s); treating it as incomplete", exc)
        return False


def _chunking_fix_applicable(checkpoint: dict[str, Any]) -> bool:
    """Whether the checkpoint exposes the processor geometry ``chunking_fix`` reads.

    The migration indexes ``config.model.processor.num_layers`` / ``num_chunks``
    and divides by the latter. Processors that do not declare both — notably
    ``NoOpProcessor`` (autoencoders) and ``PointWiseMLPProcessor`` — are simply not
    candidates. Screening for the geometry up front keeps "not applicable" separate
    from "the migration is broken", so a genuine migration bug still surfaces
    instead of being absorbed by a broad ``except``.
    """
    try:
        processor = checkpoint["hyper_parameters"]["config"].model.processor
    except (KeyError, AttributeError, TypeError):
        return False
    num_layers = getattr(processor, "num_layers", None)
    num_chunks = getattr(processor, "num_chunks", None)
    return isinstance(num_layers, int) and isinstance(num_chunks, int) and num_chunks > 0


def _sync_checkpoint_migrations(migrator: Any, checkpoint_path: Any) -> dict[str, Any] | None:
    """Apply every migration the checkpoint is missing, ledger-driven, from its file.

    Delegates to ``Migrator.sync``, which diffs the checkpoint's ledger against the
    migrations the installed anemoi-models ships and runs only what is missing —
    all of them, not the two this module can name. Returns the migrated checkpoint,
    or ``None`` when the file is not a migratable Lightning training checkpoint so
    the caller can fall back.

    Raises
    ------
    CheckpointIncompatibleError
        If the checkpoint is too old for the installed anemoi-models, or records
        migrations this version does not know about.
    """
    from anemoi.models.migrations import IncompatibleCheckpointException
    from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

    try:
        _old_ckpt, migrated, ops = migrator.sync(checkpoint_path)
    except IncompatibleCheckpointException as exc:
        msg = f"Checkpoint at {checkpoint_path} cannot be migrated by the installed anemoi-models: {exc}"
        raise CheckpointIncompatibleError(msg) from exc
    except ValueError as exc:
        # sync() migrates Lightning training checkpoints only; inference checkpoints
        # and raw state_dict saves carry no 'pytorch-lightning_version' and are out
        # of scope for it.
        LOGGER.debug("Checkpoint at %s is not a migratable training checkpoint (%s)", checkpoint_path, exc)
        return None

    if ops:
        LOGGER.info(
            "Applied %d checkpoint migration(s) from the anemoi-models ledger: %s",
            len(ops),
            ", ".join(op.migration.name for op in ops),
        )
    return migrated


# The trainable-edge permutation migration is runtime and model-dependent
# (``migrate(ckpt, model)``); it ships alongside chunking_fix in anemoi-models.
# Resolve the friendly name first, then the timestamp-prefixed module
# (``1779202136_trainable_edge_perm_fix``, the name the legacy import in
# ``anemoi.training.utils.checkpoint`` uses). Returns ``None`` when neither is
# importable (older anemoi-models), treated as "no migration needed".
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
    """Copy model parameters into ``state_dict`` under ``model.<key>``; return count injected.

    Mirrors the legacy refresh (``train/methods/base.py``): the configured processor
    prefixes are extended with every live-model key containing ``model_output_idx``,
    so those index buffers always carry the live values, not stale checkpoint ones.
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
# Shared Lightning-parity functions.
#
# These context-free functions are the single home for the checkpoint-load
# parity steps. Both the pipeline loading strategies (via the thin
# ``LoadingStrategy._*`` wrappers above) and the trainer's Lightning hook
# (``AnemoiLightningModule.on_load_checkpoint``) call them, so the algorithm
# lives in exactly one place. Each caller keeps its own assignment idiom: the
# strategies reassign ``context.checkpoint_data = apply_*_migration(...)`` while
# ``on_load_checkpoint`` discards the return and relies on the migration
# mutating Lightning's checkpoint dict in place (its long-standing contract).
# ---------------------------------------------------------------------------


def apply_checkpoint_format_migrations(
    checkpoint: dict[str, Any],
    checkpoint_path: Any = None,
) -> dict[str, Any]:
    """Bring a checkpoint up to date with the anemoi-models migration ledger.

    Applicability is decided from the ledger every anemoi checkpoint carries, not
    by running a migration and seeing whether it throws. A checkpoint whose ledger
    already records ``chunking_fix`` is returned untouched and no file is read,
    which is the common case.

    When the ledger is incomplete the checkpoint is migrated properly:
    ``Migrator.sync`` runs every migration the checkpoint is missing — all of the
    ones anemoi-models ships, not only the two this module can name by hand. That
    needs the checkpoint's file, so it applies to on-disk sources (``LocalSource``,
    ``RunSource``). Sources that keep no file behind — ``HTTPSource`` and
    ``S3Source`` delete their download before returning — fall back to the
    in-memory ``chunking_fix`` call, screened for applicability first.

    Parameters
    ----------
    checkpoint : dict
        The loaded checkpoint. Returned unchanged when ``None``.
    checkpoint_path : Path or str, optional
        The checkpoint's file, when one exists. Enables the full ledger-driven
        migration; without it only the ``chunking_fix`` fallback is available.

    Returns
    -------
    dict
        The (possibly migrated) checkpoint.

    Raises
    ------
    CheckpointIncompatibleError
        If the checkpoint is too old for the installed anemoi-models, or records
        migrations this version does not know about.
    """
    if checkpoint is None:
        return checkpoint

    from pathlib import Path

    migrator = _migrator()
    if migrator is not None:
        if _ledger_records_chunking_fix(migrator, checkpoint):
            LOGGER.debug("Checkpoint migration ledger is up to date; no format migration applied")
            return checkpoint

        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            migrated = _sync_checkpoint_migrations(migrator, checkpoint_path)
            if migrated is not None:
                return migrated

    # No ledger to consult, or no migratable file behind the checkpoint.
    return _apply_chunking_fix_in_memory(checkpoint)


def _apply_chunking_fix_in_memory(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Apply ``chunking_fix`` to a checkpoint that has no migratable file behind it.

    The narrow fallback for sources that hand over a dict and nothing else. Only
    ``chunking_fix`` is reachable this way, so a checkpoint arriving over HTTP or
    S3 that needs some *other* migration cannot be fully migrated here — that is a
    real limitation of the file-based ``Migrator.sync`` API and is logged as such
    rather than hidden.
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
        # The geometry screen above catches the shapes we know are not candidates;
        # this stays as a narrow net for checkpoints missing some other key the
        # migration reads. Anything else is a real fault and propagates.
        LOGGER.info("chunking_fix skipped: checkpoint shape incomplete (%s)", exc)
        return checkpoint


def apply_trainable_edge_perm_migration(checkpoint: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    """Apply the runtime, model-dependent trainable-edge-permutation migration.

    Returns the (possibly rewritten) checkpoint. A no-op when the migration is
    unavailable or ``checkpoint``/``model`` is missing, and tolerant of
    incomplete checkpoint shapes (skipped rather than raising).
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
    """Replace stale pre/post-processor weights in ``checkpoint['state_dict']`` with the model's.

    Honours ``training.update_ds_stats_on_ckpt_load.{states,tendencies}``: drops the
    matching ``model.(pre|post)_processors[_tendencies].*`` keys and re-injects them
    from ``model`` (plus any ``model_output_idx`` buffers). Mutates the state dict in
    place. A no-op when neither flag is set, ``model`` is ``None``, or no state dict
    is present.
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

    removed = _drop_keys_with_prefix(state_dict, prefixes)
    injected = _inject_model_weights(state_dict, model, prefixes) if model is not None else 0
    LOGGER.debug(
        "Refreshed checkpoint processors: removed %d stale entries, injected %d from current model",
        removed,
        injected,
    )


def preserve_anemoi_metadata(model: nn.Module, checkpoint_data: dict[str, Any]) -> None:
    """Restore ``model._ckpt_model_name_to_index`` from a checkpoint's ``data_indices``.

    Multi-dataset (current): ``hyper_parameters["data_indices"]`` is a
    ``dict[str, IndexCollection]`` and the attribute becomes
    ``{name: ic.name_to_index}``. Single-dataset (pre-multi-dataset) checkpoints
    carry a flat ``IndexCollection`` and are rejected with a ``TypeError`` (the
    current loaders require the dataset-keyed dict). Any other shape is a
    debug-logged skip.
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
        # Single-dataset IndexCollection from a pre-multi-dataset anemoi-core. The
        # current loaders expect dict[str, IndexCollection] keyed by dataset name; a
        # flat mapping silently breaks the dataset-keyed lookups downstream, so reject
        # it loudly rather than load a subtly-wrong model.
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
    """Populate ``model._ckpt_variables_metadata`` from the checkpoint.

    A no-op when ``model._ckpt_model_name_to_index`` is unset (metadata was not
    restored) or when ``checkpoint_data`` is ``None``, so it is safe to call
    unconditionally after :func:`preserve_anemoi_metadata`.
    """
    from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

    name_to_index = getattr(model, "_ckpt_model_name_to_index", None)
    if name_to_index is None or checkpoint_data is None:
        return
    model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(checkpoint_data, name_to_index)


def warn_on_hparams_divergence(checkpoint_data: dict[str, Any], run_config: Any) -> None:
    """Warn when the checkpoint's stored model hyper-parameters differ from the run config.

    Fill-model loading keeps the current architecture, so a checkpoint trained with a
    different model config whose tensor shapes happen to coincide would otherwise pass
    unnoticed. Best-effort: any comparison failure is silently skipped.
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
