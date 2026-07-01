# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""FreezingModifierStage — native PipelineStage for freezing model submodules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.modifiers.base import ModelModifier

if TYPE_CHECKING:
    import torch

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class FreezingModifierStage(ModelModifier):
    """Freezes specified submodules. Native PipelineStage — full feature port.

    Submodules are addressed by their full path within the model: an exact
    child name or a dot-separated path (e.g., "processor.0",
    "encoder.layers.2"). A bare name does not match nested submodules —
    the same path semantics as the legacy
    ``anemoi.training.utils.checkpoint.freeze_submodule_by_name`` (#1159).

    Parameters
    ----------
    submodules_to_freeze : list[str]
        Full paths of the submodules to freeze, relative to the model.
    strict : bool, default False
        If True, raise an error when a specified module is not found.
        If False, log a warning and continue.
    validate_gradients : bool, default True
        If True, validate that frozen parameters do not accumulate gradients
        after a forward/backward pass.
    """

    def __init__(
        self,
        submodules_to_freeze: list[str],
        strict: bool = False,
        validate_gradients: bool = True,
    ) -> None:
        self.submodules_to_freeze = list(submodules_to_freeze)

        self.strict = strict
        self.validate_gradients = validate_gradients

        LOGGER.debug(
            "Initialized FreezingModifierStage with modules: %s (strict=%s, validate=%s)",
            self.submodules_to_freeze,
            self.strict,
            self.validate_gradients,
        )

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Freeze specified submodules on context.model.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context carrying the model to modify.

        Returns
        -------
        CheckpointContext
            Updated context with frozen parameters and metadata.
        """
        model = context.model

        if not self.submodules_to_freeze:
            LOGGER.info("No submodules specified for freezing")
            return context

        LOGGER.info("Freezing the following submodules: %s", self.submodules_to_freeze)

        frozen_modules: list[dict] = []
        total_frozen = 0

        for module_name in self.submodules_to_freeze:
            frozen_count = self._freeze_submodule_by_name(model, module_name)
            if frozen_count is None:
                msg = f"Module '{module_name}' not found"
                if self.strict:
                    raise ValueError(msg)
                LOGGER.warning("%s. SKIPPING freezing.", msg)
                continue
            LOGGER.info("Froze %d parameters in '%s'", frozen_count, module_name)
            frozen_modules.append({"name": module_name, "frozen_params": frozen_count})
            total_frozen += frozen_count

        if self.validate_gradients:
            self._validate_gradient_flow(model)

        context.metadata.setdefault("modifiers_applied", []).append(
            {
                "type": "freezing",
                "submodules": self.submodules_to_freeze,
                "frozen_modules": frozen_modules,
                "total_frozen_params": total_frozen,
            },
        )

        return context

    def _freeze_submodule_by_name(self, module: torch.nn.Module, target_name: str) -> int | None:
        """Freeze the parameters of the submodule at ``target_name``.

        ``target_name`` is resolved with :meth:`torch.nn.Module.get_submodule`,
        i.e. as a full path relative to ``module``. There is no name-match
        search at arbitrary depth — a bare name only resolves a direct child,
        matching the legacy ``freeze_submodule_by_name`` semantics (#1159).

        Parameters
        ----------
        module : torch.nn.Module
            The parent module to resolve the path within.
        target_name : str
            Full path of the submodule to freeze (e.g., "processor.0",
            "encoder.attention").

        Returns
        -------
        int | None
            Number of parameters newly frozen, or ``None`` when no submodule
            exists at ``target_name``. A found submodule whose parameters are
            already frozen yields ``0``, not ``None``.
        """
        try:
            target_module = module.get_submodule(target_name)
        except AttributeError:
            return None

        frozen_count = sum(p.requires_grad for p in target_module.parameters())
        target_module.requires_grad_(False)
        return frozen_count

    def _validate_gradient_flow(self, model: torch.nn.Module) -> None:
        """Validate that the frozen submodules' parameters are non-trainable.

        A parameter with ``requires_grad=False`` cannot accumulate gradients, so
        checking the flag is sufficient and needs no forward/backward pass (which
        would also require model-specific input). A submodule name that does not
        resolve was never frozen and is skipped.
        """
        for module_name in self.submodules_to_freeze:
            try:
                target_module = model.get_submodule(module_name)
            except AttributeError:
                continue

            trainable = [name for name, p in target_module.named_parameters() if p.requires_grad]
            if trainable:
                LOGGER.warning(
                    "Frozen submodule '%s' still has trainable parameters: %s",
                    module_name,
                    trainable,
                )
