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
from typing import NoReturn

import torch

from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.base import LoadingStrategy

if TYPE_CHECKING:
    from typing import Any

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class WeightsOnlyLoader(LoadingStrategy):
    """Load only model weights; training state starts fresh.

    This is the simplest loading strategy: extract the state dict from
    checkpoint data and load it into the model. The optimizer, scheduler and
    loop progress in the checkpoint are not used (the trainer withholds
    ``ckpt_path`` from Lightning after a pipeline load).

    Behavior
    --------
    - Loads weights with ``strict=self.strict`` (default ``True``)
    - **Leaves training-progress metadata untouched** (``epoch``,
      ``global_step``). A prior pipeline stage that set these values keeps
      them. If you want explicit zero-reset semantics, use
      :class:`ColdStartLoader`.

    Composes naturally inside larger pipelines where another stage owns
    training-progress state. For top-level "fresh training from pretrained
    weights" use :class:`ColdStartLoader` instead.

    ``strict`` governs the *key set*. It cannot express tolerance of a shape
    change: PyTorch records a size mismatch in ``error_msgs`` outside its
    ``if strict:`` block, so ``strict=False`` still raises on one. That is what
    ``skip_mismatched`` is for, and the two are independent.

    Parameters
    ----------
    strict : bool, optional
        Whether to require an exact match between checkpoint keys and
        model keys (default: True). Missing keys raise ``CheckpointLoadError``.
    skip_mismatched : bool, optional
        Whether to load a checkpoint whose variable-dependent layers have a
        different shape from the model's, skipping those parameters and leaving
        them at their initialised values (default: ``False`` — a shape mismatch
        is an error). Use it to fine-tune onto a dataset with fewer variables,
        together with ``training.allow_variable_subset``.

        Only shape mismatches are skipped, never keys the model does not have —
        that is ``strict``'s job, and silently dropping them would make
        ``strict=True`` unfalsifiable. Skipped parameters are logged at WARNING
        and recorded in ``context.metadata["skipped_params"]``, because with them
        gone nothing else would notice that part of the model is still random.
    """

    def __init__(self, strict: bool = True, skip_mismatched: bool = False) -> None:
        self.strict = strict
        self.skip_mismatched = skip_mismatched

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load weights into the model.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded.
        """
        self._apply_corrections(context)

        state_dict = self._extract_state_dict(context)

        skipped = self._shape_mismatched_keys(context.model, state_dict) if self.skip_mismatched else {}
        if skipped:
            state_dict = {key: value for key, value in state_dict.items() if key not in skipped}
            self._load_skipping(context, state_dict, skipped)
        else:
            try:
                context.model.load_state_dict(state_dict, strict=self.strict)
            except RuntimeError as e:
                raise CheckpointLoadError(context.checkpoint_path or "<in-memory checkpoint>", e) from e

        self._warn_on_hparams_divergence(context)
        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._extract_variables_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        context.metadata["loading_strategy"] = "weights_only"

        LOGGER.info("Loaded weights only (strict=%s); training state starts fresh", self.strict)

        return context

    @staticmethod
    def _shape_mismatched_keys(model: Any, state_dict: dict[str, Any]) -> dict[str, str]:
        """Keys present in both model and checkpoint whose tensor shapes differ.

        Deliberately narrower than
        :func:`~anemoi.training.checkpoint.loading.utils.filter_state_dict`, which
        also drops keys the target lacks. Those are exactly what ``strict``
        governs, so dropping them here would silently override the user's choice.

        Parameters
        ----------
        model : torch.nn.Module
            The model being loaded into.
        state_dict : dict
            The checkpoint's state dict.

        Returns
        -------
        dict[str, str]
            Key to a human-readable reason, in the same format
            ``filter_state_dict`` uses so the two report identically.
        """
        target = model.state_dict()
        return {
            key: f"Shape mismatch: {value.shape} vs {target[key].shape}"
            for key, value in state_dict.items()
            if key in target
            and isinstance(value, torch.Tensor)
            and isinstance(target[key], torch.Tensor)
            and value.shape != target[key].shape
        }

    def _load_skipping(
        self,
        context: CheckpointContext,
        state_dict: dict[str, Any],
        skipped: dict[str, str],
    ) -> None:
        """Load a state dict with the shape-mismatched parameters already removed.

        The removed keys read as *missing* to PyTorch, so the load itself has to be
        non-strict. When ``strict`` was requested its guarantee is re-applied here
        instead, over the keys that were not deliberately skipped — otherwise
        ``skip_mismatched`` would quietly disable ``strict`` altogether.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context, used for the checkpoint path in error messages and
            to record the skipped parameters.
        state_dict : dict
            The checkpoint state dict, already filtered.
        skipped : dict[str, str]
            Key to reason for every parameter removed.

        Raises
        ------
        CheckpointIncompatibleError
            If nothing at all could be loaded, or if ``strict`` and there are
            missing/unexpected keys beyond the skipped ones.
        CheckpointLoadError
            If the load fails for any other reason.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

        if not state_dict:
            msg = (
                "Every parameter in the checkpoint was skipped for a shape mismatch, so nothing "
                f"would be loaded and the model would train from random weights. Skipped: {skipped}. "
                "This is almost always the wrong checkpoint rather than a deliberate reduction."
            )
            raise CheckpointIncompatibleError(msg)

        try:
            incompatible = context.model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            raise CheckpointLoadError(context.checkpoint_path or "<in-memory checkpoint>", e) from e

        if self.strict:
            unexpected = list(incompatible.unexpected_keys)
            missing = [key for key in incompatible.missing_keys if key not in skipped]
            if missing or unexpected:
                msg = (
                    "strict=True and the checkpoint key set does not match the model's: "
                    f"missing={missing}, unexpected={unexpected}. (Shape-mismatched parameters were "
                    "skipped as configured and are not counted here.)"
                )
                raise CheckpointIncompatibleError(msg)

        context.metadata["skipped_params"] = skipped
        LOGGER.warning(
            "Loaded %d parameter(s); SKIPPED %d for a shape mismatch, which stay at their initialised "
            "values: %s. Verify this is the reduction you intended — nothing downstream will notice "
            "that these are untrained.",
            len(state_dict),
            len(skipped),
            ", ".join(sorted(skipped)),
        )


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

        self._apply_corrections(context)

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
    """Marker strategy: resume an interrupted run through Lightning's ``ckpt_path`` load.

    Selecting it tells the trainer that ``Trainer.fit(ckpt_path=)`` performs the
    one and only load of the checkpoint — weights, optimizer, scheduler and loop
    progress together — with ``BaseTrainingModule.on_load_checkpoint`` applying
    the corrections to the dict Lightning loads. The pipeline's part of a resume
    is to make the checkpoint reachable as a local file (the source's ``resolve``
    step) and to run the modifier stages; the builder emits no loading stage, so
    this class's ``process`` never runs in a training run.

    The class exists so ``training/checkpoint/loading=warm_start`` composes like
    every other strategy; :attr:`restores_training_state` is what the trainer and
    builder read.
    """

    restores_training_state = True

    async def process(self, context: CheckpointContext) -> NoReturn:
        """Refuse to run: a resume is loaded by ``Trainer.fit(ckpt_path=)``, not here.

        Parameters
        ----------
        context : CheckpointContext
            Unused; the pipeline never routes a resume through this stage.

        Returns
        -------
        NoReturn
            Never returns.

        Raises
        ------
        RuntimeError
            Always. Reaching this means the pipeline was hand-built with a
            ``WarmStartLoader`` stage; use the builder, or a loading strategy
            that applies weights.
        """
        del context
        msg = "WarmStartLoader is a marker: resume loads happen in Trainer.fit(ckpt_path=)"
        raise RuntimeError(msg)


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
