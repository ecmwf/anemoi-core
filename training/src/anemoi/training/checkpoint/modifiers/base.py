# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract base class for model modifiers.

Model modifiers form the post-loading transformation layer of the
checkpoint pipeline. They mutate ``context.model`` after a loading
strategy has applied checkpoint weights — typical examples are
parameter freezing, LoRA adapter injection, and quantisation.

``ModelModifier`` is a thin intermediate ABC over
:class:`~anemoi.training.checkpoint.base.PipelineStage`. Its name is
the contract the frozen ``ComponentCatalog`` uses to find concrete
modifier classes: catalog discovery looks for any class in
``anemoi.training.checkpoint.modifiers`` whose MRO contains a base
named ``ModelModifier``.

Concrete subclasses must implement ``process``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext


class ModelModifier(PipelineStage):
    """Abstract base class for all model modifiers.

    Subclasses apply a post-loading transformation to ``context.model``
    and are expected to record what they did under
    ``context.metadata["modifiers_applied"]``.

    Examples
    --------
    >>> class FreezingModifierStage(ModelModifier):
    ...     async def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         for param in context.model.parameters():
    ...             param.requires_grad = False
    ...         return context
    """

    @abstractmethod
    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Apply the modification to ``context.model``.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context. ``context.model`` must already be
            populated (typically by a preceding loading strategy).

        Returns
        -------
        CheckpointContext
            The same context, with ``context.model`` mutated in place
            and ``context.metadata["modifiers_applied"]`` updated.
        """
