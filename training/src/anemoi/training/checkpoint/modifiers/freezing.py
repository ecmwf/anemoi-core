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

from anemoi.training.checkpoint.modifiers.base import ModelModifier

if TYPE_CHECKING:
    import torch

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)

_CONFIG_PATH = "training.checkpoint.modifiers"


class FreezingModifierStage(ModelModifier):
    """Freeze the parameters of the named submodules.

    Submodules are addressed by their full path from ``context.model``: an
    exact child name or a dot-separated path (e.g., "model.model.processor",
    "model.model.encoder.global"). A bare name does not match nested
    submodules — the full dot-path resolution introduced in #1159.

    ``context.model`` is the LightningModule, and the graph model that owns
    ``encoder`` / ``processor`` / ``decoder`` is two attribute hops below it
    (LightningModule -> AnemoiModelInterface -> the graph model). Prefix those
    names with ``model.model.`` to address them, which is where the legacy
    ``training.submodules_to_freeze`` path resolved names.

    Parameters
    ----------
    submodules_to_freeze : list[str]
        Paths of the submodules to freeze, from ``context.model``.
    strict : bool, default False
        If True, raise an error when a specified module is not found.
        If False, log a warning and continue. Either way, a config where *no*
        listed path resolves is an error — see :meth:`process`.
    """

    def __init__(
        self,
        submodules_to_freeze: list[str],
        strict: bool = False,
    ) -> None:
        self.submodules_to_freeze = list(submodules_to_freeze)

        self.strict = strict

        LOGGER.debug(
            "Initialized FreezingModifierStage with modules: %s (strict=%s)",
            self.submodules_to_freeze,
            self.strict,
        )

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
            If there is no model on the context, or if not one of the listed
            submodules resolves on it. A freezing config that freezes
            nothing is never what the user meant, so it is an error regardless
            of ``strict`` — ``strict`` governs tolerance of *one* missing name,
            not of an entirely inert config.
        ValueError
            If ``strict`` and a listed submodule is not found.
        """
        if not self.submodules_to_freeze:
            LOGGER.info("No submodules specified for freezing")
            return context

        model = self._require_model(context.model)

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

        if not frozen_modules:
            self._raise_nothing_resolved()

        context.metadata.setdefault("modifiers_applied", []).append(
            {
                "type": "freezing",
                "submodules": self.submodules_to_freeze,
                "frozen_modules": frozen_modules,
                "total_frozen_params": total_frozen,
            },
        )

        return context

    @staticmethod
    def _require_model(model: torch.nn.Module | None) -> torch.nn.Module:
        """Return the context's model, refusing a context without one.

        Parameters
        ----------
        model : torch.nn.Module or None
            The context's model, i.e. the LightningModule in a training run.

        Returns
        -------
        torch.nn.Module
            The module that ``submodules_to_freeze`` is resolved against.

        Raises
        ------
        CheckpointConfigError
            If there is no model.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        if model is None:
            msg = "FreezingModifierStage needs a model on the context, but context.model is None."
            LOGGER.error(msg)
            raise CheckpointConfigError(msg, config_path=_CONFIG_PATH)
        return model

    def _raise_nothing_resolved(self) -> None:
        """Refuse a freezing config where not one listed submodule resolved.

        Raises
        ------
        CheckpointConfigError
            Always. Mirrors ``CheckpointPipeline._verify_weights_loaded``:
            configured-but-did-nothing is a hard error, because the alternative
            is a fine-tune that silently trains every parameter.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        msg = (
            f"FreezingModifierStage was configured with {self.submodules_to_freeze} but not one path "
            "resolved. Nothing was frozen. Paths are resolved with torch.nn.Module.get_submodule from "
            "the LightningModule; for a standard training run the graph model is at model.model, so name "
            "its children with that prefix (e.g. model.model.encoder.<dataset>, model.model.processor, "
            "model.model.decoder.<dataset>). Refusing to proceed with a freezing config that freezes nothing."
        )
        LOGGER.error(msg)
        raise CheckpointConfigError(msg, config_path=_CONFIG_PATH)

    @staticmethod
    def _freeze_submodule_by_name(module: torch.nn.Module, target_name: str) -> int | None:
        """Freeze the parameters of the submodule at ``target_name``.

        ``target_name`` is resolved with :meth:`torch.nn.Module.get_submodule`,
        i.e. as a full path relative to ``module``. There is no name-match
        search at arbitrary depth — a bare name only resolves a direct child,
        the dot-path semantics introduced in #1159.

        Parameters
        ----------
        module : torch.nn.Module
            The parent module to resolve the path within.
        target_name : str
            Full path of the submodule to freeze (e.g., "model.model.processor",
            "model.model.encoder.0").

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
