# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pipeline stage that freezes named submodules of the model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.modifiers.base import ModelModifier

if TYPE_CHECKING:
    import torch

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class FreezingModifierStage(ModelModifier):
    """Freeze the parameters of the named submodules.

    Each name is a dot path from ``context.model`` (the LightningModule),
    resolved with :meth:`torch.nn.Module.get_submodule`. In a training run, the
    graph model is at ``model.model``, e.g. ``model.model.processor`` or
    ``model.model.encoder.0``. A bare name does not match nested submodules.

    Parameters
    ----------
    submodules_to_freeze : list[str]
        Paths of the submodules to freeze.
    strict : bool, default False
        If True, raise an error when a specified module is not found.
        If False, log a warning and continue. If no path resolves, always raise.
    """

    def __init__(self, submodules_to_freeze: list[str], strict: bool = False) -> None:
        self.submodules_to_freeze = list(submodules_to_freeze)
        self.strict = strict

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Freeze specified submodules on context.model.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context carrying the model to modify.

        Returns
        -------
        CheckpointContext
            Updated context with frozen parameters and metadata.

        Raises
        ------
        CheckpointConfigError
            If no listed submodule resolves. This does not depend on ``strict``.
        ValueError
            If ``strict`` and a listed submodule is not found.
        """
        if not self.submodules_to_freeze:
            LOGGER.info("No submodules specified for freezing")
            return context

        LOGGER.info("Freezing the following submodules: %s", self.submodules_to_freeze)

        frozen_modules: list[dict] = []
        for module_name in self.submodules_to_freeze:
            frozen_count = self._freeze_submodule_by_name(context.model, module_name)
            if frozen_count is None:
                msg = f"Module '{module_name}' not found"
                if self.strict:
                    raise ValueError(msg)
                LOGGER.warning("%s. SKIPPING freezing.", msg)
                continue
            LOGGER.info("Froze %d parameters in '%s'", frozen_count, module_name)
            frozen_modules.append({"name": module_name, "frozen_params": frozen_count})

        if not frozen_modules:
            # Otherwise the fine-tune silently trains all parameters.
            msg = (
                f"FreezingModifierStage was configured with {self.submodules_to_freeze} but not one path "
                "resolved, so nothing was frozen. Paths are resolved with torch.nn.Module.get_submodule "
                "from the LightningModule; for a standard training run the graph model is at model.model "
                "(e.g. model.model.encoder.<dataset>, model.model.processor)."
            )
            LOGGER.error(msg)
            raise CheckpointConfigError(msg, config_path="training.checkpoint.modifiers")

        context.metadata.setdefault("modifiers_applied", []).append(
            {
                "type": "freezing",
                "submodules": self.submodules_to_freeze,
                "frozen_modules": frozen_modules,
                "total_frozen_params": sum(module["frozen_params"] for module in frozen_modules),
            },
        )

        return context

    @staticmethod
    def _freeze_submodule_by_name(module: torch.nn.Module, target_name: str) -> int | None:
        """Freeze the parameters of the submodule at ``target_name``.

        Parameters
        ----------
        module : torch.nn.Module
            The module to resolve the path in.
        target_name : str
            Full path of the submodule to freeze (e.g., "model.model.processor").

        Returns
        -------
        int | None
            Number of parameters newly frozen, or ``None`` if there is no
            submodule at ``target_name``. An already frozen submodule gives ``0``.
        """
        try:
            target_module = module.get_submodule(target_name)
        except AttributeError:
            return None

        frozen_count = sum(p.requires_grad for p in target_module.parameters())
        target_module.requires_grad_(False)
        return frozen_count
