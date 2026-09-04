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
is suppressed), so each strategy must itself apply every correction that
``BaseTrainingModule.on_load_checkpoint`` would have applied — otherwise the
loaded state dict would silently differ from the legacy path.

Those corrections are one function, :func:`apply_checkpoint_corrections`, with
one fixed order: format migrations, the trainable-edge-permutation migration,
then the processor-statistics refresh. The pipeline strategies call it on the
dict a source produced (via :meth:`LoadingStrategy._apply_corrections`);
``BaseTrainingModule.on_load_checkpoint`` calls it on the dict Lightning is
about to load. The metadata steps (:func:`preserve_anemoi_metadata`,
:func:`extract_checkpoint_variables_metadata`) and the weights-only hparams
check (:func:`warn_on_hparams_divergence`) stay separate because they read the
checkpoint rather than change it.

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
    apply it to the model according to a specific strategy.

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

    #: Whether selecting this strategy means the run resumes through
    #: ``Trainer.fit(ckpt_path=)``. ``True`` (``WarmStartLoader``) makes Lightning
    #: perform the one and only load — weights, optimizer, scheduler and loop
    #: progress — and the builder emits no loading stage at all. ``False`` (the
    #: default) means the strategy applies the weights itself at model build and
    #: the trainer suppresses ``ckpt_path`` so training state starts fresh.
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

    def _apply_corrections(self, context: CheckpointContext) -> None:
        """Run :func:`apply_checkpoint_corrections` on ``context.checkpoint_data``.

        Reassigns the (possibly replaced) checkpoint onto the context.
        ``context.checkpoint_path`` is forwarded so a checkpoint with an incomplete
        migration ledger is reported with the file to migrate. A context without
        checkpoint data is left untouched.
        """
        if context.checkpoint_data is None:
            return
        context.checkpoint_data = apply_checkpoint_corrections(
            context.checkpoint_data,
            context.model,
            context.config,
            checkpoint_path=context.checkpoint_path,
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

        Both readers exempt a resume, where the weights arrive at
        ``Trainer.fit(ckpt_path=)`` and no strategy runs. The trainer also reads
        the flag to decide whether the transfer-learning validators have a loaded
        checkpoint to compare against.

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


def _has_migration_ledger(checkpoint: dict[str, Any]) -> bool:
    """Whether the checkpoint carries a non-empty migration ledger.

    Distinguishes "this checkpoint tracks migrations and is behind" from "this
    checkpoint predates migration tracking". Only the first can be reported as
    out of date; the second is what every pre-ledger checkpoint looks like and
    it loads exactly as before.
    """
    try:
        return bool(checkpoint.get("migrations"))
    except AttributeError:  # pragma: no cover - checkpoint is always a mapping here
        return False


def _outstanding_migrations(migrator: Any, checkpoint: dict[str, Any]) -> list[str] | None:
    """Names of the migrations this checkpoint is missing, in application order.

    ``Migrator._resolve_migrations`` diffs the checkpoint's ledger against the
    newest compatibility group. It reads the ledger only: no file is touched and
    the checkpoint is not mutated.

    This has to be asked of the *whole* ledger, not of one migration by name.
    ``register_migrations`` stamps every migration the writing version knew, not
    the ones it applied, so any checkpoint written after a given migration
    shipped records it. Keying on a single name (the previous gate keyed on
    ``chunking_fix``, the third of ten) reported "up to date" for every
    checkpoint written since that migration existed, and every newer migration
    was silently skipped.

    Parameters
    ----------
    migrator : Any
        An anemoi-models ``Migrator``.
    checkpoint : dict
        The in-memory checkpoint whose ledger is read.

    Returns
    -------
    list[str] or None
        The outstanding migration names, empty when the checkpoint is current, or
        ``None`` when the ledger cannot be read at all, which the caller treats as
        "cannot tell" and falls through rather than refusing the load.
    """
    try:
        from anemoi.models.migrations import IncompatibleCheckpointException
    except ImportError:  # pragma: no cover - _migrator() already proved the import works
        IncompatibleCheckpointException = ()  # noqa: N806

    try:
        _setups, ops, _extra = migrator._resolve_migrations(checkpoint, migrator._grouped_migrations[-1])
    except (IncompatibleCheckpointException, AttributeError, KeyError, TypeError, IndexError) as exc:
        # IncompatibleCheckpointException derives from BaseException, so it has to be
        # named explicitly: ``except Exception`` would not catch it.
        LOGGER.debug("Could not resolve the checkpoint migration ledger (%s); treating it as unknown", exc)
        return None
    return [op.migration.name for op in ops]


def _warn_unmigrated_checkpoint(outstanding: list[str], checkpoint_path: Any) -> None:
    """Warn that a checkpoint is behind the installed anemoi-models. Never blocks.

    Warning rather than migrating in-process, and warning rather than refusing:

    - Migrating in-process means ``Migrator.sync``, a file-oriented CLI API. It
      re-reads the checkpoint twice and deep-copies it (four full copies of a
      multi-gigabyte checkpoint in host RAM per rank, two extra reads of a file
      whose bytes are already held) and it runs the migrations' ``migrate_setup``
      hooks, which mutate ``sys.modules`` of the live training process with
      nothing to undo them.
    - Refusing blocks work that has always succeeded. Most of what a ledger can
      be missing does not affect the load: ``initial`` returns the checkpoint
      unchanged, ``deprecate_eda`` and ``hardware_schema_update`` have no-op
      ``migrate`` (their setup hooks matter only while unpickling, which has
      already happened once we hold a dict), ``glu_mlp_implementation`` renames
      keys no non-GraphTransformer model has, ``rename_swa_to_weight_averaging``
      touches only the checkpoint's archived config, and
      ``trainable_edge_perm_fix`` is applied by
      :func:`apply_trainable_edge_perm_migration` moments later. Refusing would
      also break read-only workflows (``anemoi-training evaluate`` would need
      write access to the checkpoint) and remote sources, since the remedy is a
      local file operation.

    So: say precisely what is missing, and let the load proceed. A migration that
    genuinely matters still fails downstream (``preserve_anemoi_metadata`` raises
    ``TypeError`` naming ``anemoi-models migration sync`` for a pre-multi-dataset
    checkpoint), now with this warning already in the log to explain it.

    Parameters
    ----------
    outstanding : list[str]
        Names of the migrations the checkpoint is missing.
    checkpoint_path : Any
        The checkpoint's file, when one exists, to name in the remedy.
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
# corrections. :func:`apply_checkpoint_corrections` composes the three that
# change the checkpoint; both the pipeline loading strategies (via
# ``LoadingStrategy._apply_corrections``) and the trainer's Lightning hook
# (``BaseTrainingModule.on_load_checkpoint``) call it, so the algorithm and its
# order live in exactly one place.
# ---------------------------------------------------------------------------


def apply_checkpoint_corrections(
    checkpoint: dict[str, Any] | None,
    model: nn.Module | None,
    config: Any,
    *,
    checkpoint_path: Any = None,
) -> dict[str, Any] | None:
    """Bring a checkpoint into line with the installed anemoi-models and the live model.

    The one place the load-time corrections live, in one fixed order:

    1. :func:`apply_checkpoint_format_migrations` — the ledger-driven path may
       replace the dict wholesale, so it runs first;
    2. :func:`apply_trainable_edge_perm_migration` — takes the live model;
    3. :func:`refresh_checkpoint_processors` — driven by
       ``config.training.update_ds_stats_on_ckpt_load``.

    The edge-permutation migration runs before the refresh because it takes the
    live model and may rebuild the state dict: run afterwards it would discard the
    processor buffers the refresh injected. It is also the order the
    ``on_load_checkpoint`` hook has always used for ``ckpt_path`` loads.

    Called by ``BaseTrainingModule.on_load_checkpoint`` on the dict Lightning is
    about to load (resume) and by the pipeline loading strategies on the dict a
    source produced.

    Parameters
    ----------
    checkpoint : dict or None
        The loaded checkpoint. Returned unchanged when ``None``.
    model : nn.Module or None
        The training module. The edge-permutation migration takes it as is; the
        processor refresh reads its inner ``.model`` (the ``AnemoiModelInterface``),
        whose state-dict keys lack the leading ``model.`` that the refresh re-adds.
    config : Any
        The run config; ``training.update_ds_stats_on_ckpt_load.{states,tendencies}``
        is read defensively (a missing layer disables the refresh).
    checkpoint_path : Path or str, optional
        The checkpoint's file, when one exists; named in the warning a checkpoint
        with an incomplete migration ledger gets.

    Returns
    -------
    dict or None
        The corrected checkpoint. The same object when every step worked in place; a
        new object only when the ledger-driven migration replaced it.
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
    """Bring a checkpoint up to date with the anemoi-models migration ledger.

    Applicability is decided from the ledger every anemoi checkpoint carries, not
    by running a migration and seeing whether it throws. A checkpoint that is
    missing no migration is returned untouched and no file is read, which is the
    common case for anything a current anemoi wrote.

    "Missing nothing" is asked of the whole ledger, never of one migration by
    name: the ledger records what the writing version *knew*, not what it
    *applied*, so any checkpoint written after a migration shipped records that
    migration whether or not it was ever needed.

    When the ledger is incomplete the outstanding migrations are named in a
    WARNING and the load proceeds (see :func:`_warn_unmigrated_checkpoint` for why
    neither migrating in-process nor refusing is the right trade), then the
    in-memory ``chunking_fix`` fallback runs, screened for applicability first.
    A checkpoint with no ledger at all predates migration tracking and gets the
    same in-memory fallback it always did.

    Parameters
    ----------
    checkpoint : dict
        The loaded checkpoint. Returned unchanged when ``None``.
    checkpoint_path : Path or str, optional
        The checkpoint's file, when one exists. Only used to name the file in the
        remedy suggested when the ledger is incomplete.

    Returns
    -------
    dict
        The (possibly migrated) checkpoint.
    """
    if checkpoint is None:
        return checkpoint

    migrator = _migrator()
    # Two guards narrow this to a checkpoint that *has* a ledger which is incomplete:
    #   - "pytorch-lightning_version" is the key ``Migrator._load_ckpt`` itself gates
    #     on. An inference checkpoint or a raw state_dict save is not a migratable
    #     training checkpoint and has no ledger to be behind on.
    #   - a checkpoint with no ledger at all predates migration tracking rather than
    #     lagging it, and loads exactly as before.
    if migrator is not None and "pytorch-lightning_version" in checkpoint and _has_migration_ledger(checkpoint):
        outstanding = _outstanding_migrations(migrator, checkpoint)
        if outstanding == []:
            LOGGER.debug("Checkpoint migration ledger is up to date; no format migration applied")
            return checkpoint
        if outstanding:
            _warn_unmigrated_checkpoint(outstanding, checkpoint_path)

    # No ledger to consult, an unreadable one, or a checkpoint that is behind.
    return _apply_chunking_fix_in_memory(checkpoint)


def _apply_chunking_fix_in_memory(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Apply ``chunking_fix`` in memory, screened for applicability first.

    The one migration this module applies itself, because it is the one whose
    absence silently corrupts a non-strict load (renamed processor keys just go
    missing). Every other outstanding migration is reported by
    :func:`_warn_unmigrated_checkpoint`; migrating them belongs to the
    ``anemoi-models migration sync`` CLI, offline, once, not to every rank of
    every run.
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
    place. A no-op when neither flag is set or no state dict is present.

    The drop and the re-injection are one operation: without a model to re-inject
    from, nothing is dropped either, so a checkpoint is never left missing the
    processor entries a strict load needs. Dropping entries the model cannot replace
    (a model without those processors) is legitimate and is logged as a warning.
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
