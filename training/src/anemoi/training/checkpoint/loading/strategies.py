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
from anemoi.training.checkpoint.loading.base import extract_checkpoint_variables_metadata
from anemoi.training.checkpoint.loading.base import preserve_anemoi_metadata
from anemoi.training.checkpoint.loading.base import warn_on_hparams_divergence

if TYPE_CHECKING:
    from typing import Any

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class WeightsOnlyLoader(LoadingStrategy):
    """Load only model weights; training state starts fresh.

    This is the simplest loading strategy: extract the state dict from
    checkpoint data and load it into the model. The optimizer, scheduler and
    loop progress are not used.

    Behavior
    --------
    - Loads weights with ``strict=self.strict`` (default ``True``)
    - **Leaves training-progress metadata untouched** (``epoch``,
      ``global_step``). To reset them to zero, use :class:`ColdStartLoader`.

    Composes naturally inside larger pipelines where another stage owns
    training-progress state. For top-level "fresh training from pretrained
    weights" use :class:`ColdStartLoader` instead.

    ``strict`` controls the key set only. ``strict=False`` still raises on a shape
    mismatch; use ``skip_mismatched`` for that.

    Parameters
    ----------
    strict : bool, optional
        Whether to require an exact match between checkpoint keys and
        model keys (default: True).
    skip_mismatched : bool, optional
        Whether to skip parameters with a different shape and keep their
        initial values (default: False). Use it with
        ``training.allow_variable_subset`` to fine-tune on fewer variables.
        Skipped parameters are logged and recorded in
        ``context.metadata["skipped_params"]``.
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

        warn_on_hparams_divergence(context.checkpoint_data, context.config)
        preserve_anemoi_metadata(context.model, context.checkpoint_data)
        extract_checkpoint_variables_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        context.metadata["loading_strategy"] = "weights_only"

        LOGGER.info("Loaded weights only (strict=%s); training state starts fresh", self.strict)

        return context

    @staticmethod
    def _shape_mismatched_keys(model: Any, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return the keys in both model and checkpoint whose tensor shapes differ.

        Parameters
        ----------
        model : torch.nn.Module
            The target model.
        state_dict : dict
            The checkpoint state dict.

        Returns
        -------
        dict[str, str]
            Key to reason, in the ``filter_state_dict`` format.
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
        """Load a state dict with the shape-mismatched parameters removed.

        The load is non-strict, so this method does the ``strict`` check itself,
        excluding the skipped keys.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context.
        state_dict : dict
            The filtered checkpoint state dict.
        skipped : dict[str, str]
            Key to reason for each removed parameter.

        Raises
        ------
        CheckpointIncompatibleError
            If nothing loads, or if ``strict`` and other keys do not match.
        CheckpointLoadError
            If the load fails for a different reason.
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
    mismatches are skipped rather than raising an error. Each skip is logged
    as a warning.

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

        preserve_anemoi_metadata(context.model, context.checkpoint_data)
        extract_checkpoint_variables_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        context.metadata["loading_strategy"] = "transfer_learning"
        context.metadata["transferred_params"] = list(filtered.keys())
        context.metadata["skipped_params"] = skipped

        LOGGER.info(
            "Transfer learning: loaded %d params, skipped %d",
            len(filtered),
            len(skipped),
        )
        if skipped:
            self._warn_about_skipped(len(filtered), skipped)

        return context

    @staticmethod
    def _warn_about_skipped(transferred: int, skipped: dict[str, str]) -> None:
        """Log each checkpoint tensor that did not transfer, and the reason.

        A model width change can make almost all tensors skip.

        Parameters
        ----------
        transferred : int
            Number of loaded parameters.
        skipped : dict[str, str]
            Skipped key to reason.
        """
        shape_mismatched = sum("Shape mismatch" in reason for reason in skipped.values())
        headline = (
            f"Transfer learning: {len(skipped)} checkpoint parameters did not transfer "
            f"({shape_mismatched} shape-mismatched, left at random initialisation in the model; "
            f"{len(skipped) - shape_mismatched} not present in the model)"
        )
        if len(skipped) > transferred:
            headline += f", more than the {transferred} that did"
        listing = "\n  ".join(f"{key}: {reason}" for key, reason in sorted(skipped.items()))
        LOGGER.warning("%s:\n  %s", headline, listing)


class WarmStartLoader(LoadingStrategy):
    """Marker strategy: resume a run through ``Trainer.fit(ckpt_path=)``.

    Lightning loads the full training state. The builder adds no loading stage,
    so ``process`` does not run. The model must match the checkpoint exactly.
    For a changed model, use ``WeightsOnlyLoader`` or ``TransferLearningLoader``.
    """

    restores_training_state = True

    async def process(self, context: CheckpointContext) -> NoReturn:
        """Raise: ``Trainer.fit(ckpt_path=)`` loads a resume, not this stage.

        Parameters
        ----------
        context : CheckpointContext
            Not used.

        Returns
        -------
        NoReturn
            Never returns.

        Raises
        ------
        RuntimeError
            Always.
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
