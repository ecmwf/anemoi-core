# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract base class for checkpoint loading strategies.

A loading strategy applies checkpoint data to a model: warm start (resume),
cold start (fresh optimiser), transfer learning (partial load), weights-only.

Wiring to the Lightning trainer
-------------------------------
On the weights-only, transfer-learning and cold-start paths Lightning never sees
the checkpoint (``ckpt_path`` is suppressed) so the strategy must itself apply
every correction ``BaseTrainingModule.on_load_checkpoint`` would have applied,
or the loaded state dict silently differs from what a resume produces. Those
corrections are one function with one fixed order,
:func:`apply_checkpoint_corrections`, called by both.

Example
-------
>>> class WeightsOnlyLoader(LoadingStrategy):
...     def process(self, context: CheckpointContext) -> CheckpointContext:
...         state_dict = self._extract_state_dict(context)
...         context.model.load_state_dict(state_dict, strict=False)
...         self._mark_weights_loaded(context.model)
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

    Subclasses implement ``process``, with some helpers.

    - ``_extract_state_dict``: Extract model state dict from checkpoint data
    - ``_mark_weights_loaded``: Flag the model as having loaded weights

    Examples
    --------
    >>> class TransferLearningLoader(LoadingStrategy):
    ...     def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         state_dict = self._extract_state_dict(context)
    ...         # ... filter and apply compatible weights ...
    ...         preserve_anemoi_metadata(context.model, context.checkpoint_data)
    ...         self._mark_weights_loaded(context.model)
    ...         return context
    """

    #: Whether selecting this strategy means the run resumes through
    #: ``Trainer.fit(ckpt_path=)``. ``True`` (``WarmStartLoader``) makes Lightning
    #: perform the one and only load — weights, optimizer, scheduler and loop
    #: progress — and the builder emits no loading stage at all. ``False`` (the
    #: default) means the strategy applies the weights itself at model build and
    #: the runner withholds ``ckpt_path`` so training state starts fresh.
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
            Pipeline context with ``checkpoint_data`` populated by a prior source
            stage and ``model`` set to the target model.

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

        Delegates to :func:`anemoi.training.checkpoint.formats.extract_state_dict`,
        which handles the Lightning, PyTorch and raw state dict shapes.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` populated.

        Returns
        -------
        dict[str, Any]
            Extracted model state dictionary.

        Raises
        ------
        CheckpointLoadError
            If the source stage deferred its error to rank 0 on this rank.
        CheckpointValidationError
            If no valid state dict can be found in checkpoint data.
        """
        from anemoi.training.checkpoint.formats import extract_state_dict

        self._abort_if_source_deferred(context)
        return extract_state_dict(context.checkpoint_data)

    @staticmethod
    def _abort_if_source_deferred(context: CheckpointContext) -> None:
        """Fail with the real reason when the source deferred its error to rank 0.

        A source that cannot reach its checkpoint raises on rank 0 and passes
        through on every other rank, so a distributed run fails with one
        actionable message instead of N. But the pass-through leaves
        ``checkpoint_data`` unset, and the loading stage then reported
        "checkpoint data is not a dictionary ... might indicate a corrupted or
        incompatible checkpoint file" — about a file it never opened. In an
        aggregated SLURM log that misleading message is what most ranks print,
        burying rank 0's correct one.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context, carrying ``source_deferred`` when a source stage
            deferred on this rank.

        Raises
        ------
        CheckpointLoadError
            Always, when the source deferred. The message names rank 0 as the
            place to look.
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
        """Run :func:`apply_checkpoint_corrections` on ``context.checkpoint_data``.

        Reassigns the (possibly replaced) checkpoint onto the context, so the state
        dict extracted afterwards is the corrected one. A context without checkpoint
        data is left untouched.
        """
        if context.checkpoint_data is None:
            return
        context.checkpoint_data = apply_checkpoint_corrections(context.checkpoint_data, context.model)

    def _mark_weights_loaded(self, model: nn.Module) -> None:
        """Set ``model.weights_initialized = True``.

        This is how a source stage that ran without any strategy applying weights
        is detected: ``CheckpointPipeline._verify_weights_loaded`` treats it as a
        gate and raises rather than train on random weights. It is only a gate:
        whether the run resumes through ``Trainer.fit(ckpt_path=)`` is decided
        from the config, never from this flag. The gate exempts a resume,
        where the weights arrive at ``Trainer.fit(ckpt_path=)``. A strategy that
        forgets to call this therefore trips the gate.

        Parameters
        ----------
        model : nn.Module
            Model to mark as weight-loaded.
        """
        model.weights_initialized = True
        LOGGER.debug("Marked model weights as initialized")


# The two anemoi-models migrations a load applies in memory, imported the way
# ``train/methods/base.py`` and ``utils/checkpoint.py`` on ``main`` import them.
# Both ``migrate`` functions return the checkpoint they produce, which callers
# must use: today they work in place, but nothing in their contract says so.
_chunking_fix = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate
_trainable_edge_perm_fix = importlib.import_module(
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
).migrate


def _chunking_fix_applicable(checkpoint: dict[str, Any]) -> bool:
    """Whether the checkpoint exposes the processor geometry ``chunking_fix`` reads.

    The migration indexes ``config.model.processor.num_layers`` / ``num_chunks``
    and divides by the latter, so a processor declaring neither (``NoOpProcessor``,
    ``PointWiseMLPProcessor``) is not a candidate. It also rewrites
    ``checkpoint["state_dict"]``, so a checkpoint without one is not either.
    Screening up front keeps "not applicable" distinct from "the migration is
    broken", which a broad ``except`` would absorb.
    """
    try:
        processor = checkpoint["hyper_parameters"]["config"].model.processor
    except (KeyError, AttributeError, TypeError):
        return False
    num_layers = getattr(processor, "num_layers", None)
    num_chunks = getattr(processor, "num_chunks", None)
    return "state_dict" in checkpoint and isinstance(num_layers, int) and isinstance(num_chunks, int) and num_chunks > 0


# ---------------------------------------------------------------------------
# Shared Lightning-parity functions: the single home for the checkpoint-load
# corrections. Both the pipeline loading strategies and the trainer's
# ``on_load_checkpoint`` hook call apply_checkpoint_corrections, so the algorithm
# and its order live in exactly one place.
# ---------------------------------------------------------------------------


def apply_checkpoint_corrections(
    checkpoint: dict[str, Any] | None,
    model: nn.Module | None,
) -> dict[str, Any] | None:
    """Bring a checkpoint into line with the installed anemoi-models and the live model.

    The one place the load-time corrections live, in one fixed order:

    1. ``chunking_fix`` — only when the processor declares the geometry it reads
       (see :func:`_chunking_fix_applicable`);
    2. ``trainable_edge_perm_fix`` — takes the live model.

    Each migration's return value is the checkpoint the next step works on.
    Both migrations change only old-format entries, so running them on a
    checkpoint that already has them applied leaves it as it is.

    Called by ``BaseTrainingModule.on_load_checkpoint`` on the dict Lightning is
    about to load (resume) and by the pipeline loading strategies on the dict a
    source produced.

    Parameters
    ----------
    checkpoint : dict or None
        The loaded checkpoint. Returned unchanged when ``None``.
    model : nn.Module or None
        The training module, which the edge-permutation migration takes as is.

    Returns
    -------
    dict or None
        The corrected checkpoint: whatever the last migration returned. Callers
        must use this object, not the one they passed in.
    """
    if checkpoint is None:
        return checkpoint

    if _chunking_fix_applicable(checkpoint):
        checkpoint = _chunking_fix(checkpoint)
    else:
        LOGGER.debug(
            "chunking_fix not applied: the checkpoint declares no usable processor num_layers/num_chunks "
            "(e.g. a NoOpProcessor autoencoder, or a raw state_dict save)",
        )
    return _trainable_edge_perm_fix(checkpoint, model)


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
