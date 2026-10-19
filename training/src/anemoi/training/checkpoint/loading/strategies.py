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

from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.base import LoadingStrategy
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections
from anemoi.training.checkpoint.loading.base import preserve_anemoi_metadata
from anemoi.training.checkpoint.loading.base import warn_on_hparams_divergence
from anemoi.training.checkpoint.loading.utils import shape_mismatches

if TYPE_CHECKING:
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

    ``strict`` controls the key set only. A shape mismatch always raises,
    unless ``skip_mismatched`` is set.

    Parameters
    ----------
    strict : bool, optional
        Whether to require an exact match between checkpoint keys and
        model keys (default: True). If False, checkpoint keys that are not
        in the model are skipped.
    skip_mismatched : bool, optional
        Whether to skip parameters with a different shape and keep their
        initial values (default: False). Use it with
        ``training.allow_variable_subset`` to fine-tune on fewer variables.
        Skipped parameters are logged and recorded in
        ``context.metadata["skipped_params"]``.
    """

    strategy_name = "weights_only"

    def __init__(self, strict: bool = True, skip_mismatched: bool = False) -> None:
        self.strict = strict
        self.skip_mismatched = skip_mismatched

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load weights into the model.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded.

        Raises
        ------
        CheckpointIncompatibleError
            If shapes differ and not ``skip_mismatched``, if all parameters are
            skipped, or if ``strict`` and the key sets do not match.
        CheckpointLoadError
            If ``load_state_dict`` fails for a different reason.
        """
        state_dict = self._extract_state_dict(context)
        # The corrections change the state dict in place.
        apply_checkpoint_corrections(context.checkpoint_data, context.model, context.config)

        mismatched = shape_mismatches(context.model.state_dict(), state_dict)
        if mismatched and not self.skip_mismatched:
            msg = "Parameter shapes differ between checkpoint and model. Set skip_mismatched: true to skip them."
            raise CheckpointIncompatibleError(msg, shape_mismatches=mismatched)
        to_load = {key: value for key, value in state_dict.items() if key not in mismatched}
        if mismatched and not to_load:
            msg = (
                "Every parameter in the checkpoint was skipped for a shape mismatch, so nothing "
                f"would be loaded and the model would train from random weights. Skipped: {mismatched}. "
                "This is almost always the wrong checkpoint rather than a deliberate reduction."
            )
            raise CheckpointIncompatibleError(msg)

        try:
            result = context.model.load_state_dict(to_load, strict=self.strict and not mismatched)
        except RuntimeError as e:
            raise CheckpointLoadError(context.checkpoint_path or "<in-memory checkpoint>", e) from e

        # A strict load without skipped parameters raised above. Check the other case here.
        missing = [key for key in result.missing_keys if key not in mismatched]
        if self.strict and (missing or result.unexpected_keys):
            msg = (
                "strict=True and the checkpoint key set does not match the model's. "
                "(Shape-mismatched parameters were skipped as configured and are not counted here.)"
            )
            raise CheckpointIncompatibleError(msg, missing_keys=missing, unexpected_keys=result.unexpected_keys)

        skipped = mismatched | dict.fromkeys(result.unexpected_keys, "Key not in target")
        transferred = [key for key in to_load if key not in skipped]

        warn_on_hparams_divergence(context.checkpoint_data, context.config)
        preserve_anemoi_metadata(context.model, context.checkpoint_data)
        context.model.weights_initialized = True

        context.metadata["loading_strategy"] = self.strategy_name
        context.metadata["transferred_params"] = transferred
        context.metadata["skipped_params"] = skipped

        LOGGER.info("Loaded %d parameter(s) (strict=%s); training state starts fresh", len(transferred), self.strict)
        if skipped:
            LOGGER.warning(
                "SKIPPED %d checkpoint parameter(s)%s: %d shape-mismatched (these keep their initial "
                "values), %d not present in the model:\n  %s",
                len(skipped),
                f", more than the {len(transferred)} that did load" if len(skipped) > len(transferred) else "",
                len(mismatched),
                len(skipped) - len(mismatched),
                "\n  ".join(f"{key}: {reason}" for key, reason in sorted(skipped.items())),
            )

        return context


class TransferLearningLoader(WeightsOnlyLoader):
    """Load the parameters that match the model. Skip and log the others.

    A :class:`WeightsOnlyLoader` with ``strict=False`` and ``skip_mismatched=True``.

    Parameters
    ----------
    skip_mismatched : bool, optional
        Whether to skip keys with mismatched shapes (default: True).
        If False, shape mismatches raise ``CheckpointIncompatibleError``.
    strict : bool, optional
        Whether to require an exact match of the key sets (default: False).
    """

    strategy_name = "transfer_learning"

    def __init__(self, skip_mismatched: bool = True, strict: bool = False) -> None:
        super().__init__(strict=strict, skip_mismatched=skip_mismatched)


class WarmStartLoader(LoadingStrategy):
    """Marker strategy: resume a run through ``Trainer.fit(ckpt_path=)``.

    Lightning loads the full training state. The builder adds no loading stage,
    so ``process`` does not run. The model must match the checkpoint exactly.
    For a changed model, use ``WeightsOnlyLoader`` or ``TransferLearningLoader``.
    """

    restores_training_state = True

    def process(self, context: CheckpointContext) -> NoReturn:
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

    def process(self, context: CheckpointContext) -> CheckpointContext:
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
        context = super().process(context)

        context.metadata["epoch"] = 0
        context.metadata["global_step"] = 0
        context.metadata["loading_strategy"] = "cold_start"
        context.metadata["pretrained_from"] = str(context.checkpoint_path) if context.checkpoint_path else None

        LOGGER.info("Cold start: training state reset, pretrained from %s", context.checkpoint_path)

        return context
