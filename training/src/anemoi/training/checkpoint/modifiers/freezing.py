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

_CONFIG_PATH = "training.checkpoint.modifiers"


class FreezingModifierStage(ModelModifier):
    """Freezes specified submodules. Native PipelineStage — full feature port.

    Submodules are addressed by their full path *within the root*: an exact
    child name or a dot-separated path (e.g., "processor", "encoder.data").
    A bare name does not match nested submodules — the full dot-path
    resolution introduced in #1159.

    The root matters. ``context.model`` is the LightningModule, and the graph
    model that owns ``encoder`` / ``processor`` / ``decoder`` is two attribute
    hops below it (LightningModule -> AnemoiModelInterface -> the graph model).
    Set ``submodule_root: model.model`` to address those, which is where the
    legacy ``training.submodules_to_freeze`` path resolved names. The default
    ``""`` resolves against ``context.model`` itself, for callers that hand the
    stage a bare module.

    Parameters
    ----------
    submodules_to_freeze : list[str]
        Paths of the submodules to freeze, relative to ``submodule_root``.
    submodule_root : str, default ""
        Path within ``context.model`` that ``submodules_to_freeze`` is resolved
        against, itself in dot notation. ``""`` means ``context.model`` itself.
        For a standard anemoi training run this is ``"model.model"``.
    strict : bool, default False
        If True, raise an error when a specified module is not found.
        If False, log a warning and continue. Either way, a config where *no*
        listed path resolves is an error — see :meth:`process`.
    validate_gradients : bool, default True
        If True, verify after freezing that the named submodules hold no
        trainable parameters.
    """

    def __init__(
        self,
        submodules_to_freeze: list[str],
        submodule_root: str = "",
        strict: bool = False,
        validate_gradients: bool = True,
    ) -> None:
        self.submodules_to_freeze = list(submodules_to_freeze)

        self.submodule_root = submodule_root
        self.strict = strict
        self.validate_gradients = validate_gradients

        LOGGER.debug(
            "Initialized FreezingModifierStage with modules: %s (root=%r, strict=%s, validate=%s)",
            self.submodules_to_freeze,
            self.submodule_root,
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

        Raises
        ------
        CheckpointConfigError
            If ``submodule_root`` does not resolve, or if not one of the listed
            submodules resolves against it. A freezing config that freezes
            nothing is never what the user meant, so it is an error regardless
            of ``strict`` — ``strict`` governs tolerance of *one* missing name,
            not of an entirely inert config.
        ValueError
            If ``strict`` and a listed submodule is not found.
        """
        if not self.submodules_to_freeze:
            LOGGER.info("No submodules specified for freezing")
            return context

        model = self._resolve_root(context.model)

        LOGGER.info(
            "Freezing the following submodules (root=%r): %s",
            self.submodule_root,
            self.submodules_to_freeze,
        )

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

    def _resolve_root(self, model: torch.nn.Module | None) -> torch.nn.Module:
        """Resolve ``submodule_root`` within ``model``.

        Parameters
        ----------
        model : torch.nn.Module or None
            The context's model, i.e. the LightningModule in a training run.

        Returns
        -------
        torch.nn.Module
            The module that ``submodules_to_freeze`` is resolved against.
            ``get_submodule("")`` returns the module itself, so the default
            root costs nothing.

        Raises
        ------
        CheckpointConfigError
            If there is no model, or ``submodule_root`` does not resolve. A
            root that silently fell back to the model would reintroduce the
            defect this parameter exists to fix.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        if model is None:
            msg = "FreezingModifierStage needs a model on the context, but context.model is None."
            LOGGER.error(msg)
            raise CheckpointConfigError(msg, config_path=_CONFIG_PATH)

        try:
            return model.get_submodule(self.submodule_root)
        except AttributeError as exc:
            msg = (
                f"FreezingModifierStage submodule_root {self.submodule_root!r} does not resolve on "
                f"{type(model).__name__}. For a standard training run the graph model that owns "
                "encoder/processor/decoder is at 'model.model'."
            )
            LOGGER.exception(msg)
            raise CheckpointConfigError(msg, config_path=_CONFIG_PATH) from exc

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
            f"resolved against submodule_root {self.submodule_root!r}. Nothing was frozen. Paths are "
            "resolved with torch.nn.Module.get_submodule relative to that root; for a standard "
            "training run set submodule_root: model.model and name the graph model's children "
            "(e.g. encoder.<dataset>, processor, decoder.<dataset>). Refusing to proceed with a "
            "freezing config that freezes nothing."
        )
        LOGGER.error(msg)
        raise CheckpointConfigError(msg, config_path=_CONFIG_PATH)

    def _freeze_submodule_by_name(self, module: torch.nn.Module, target_name: str) -> int | None:
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

        Parameters
        ----------
        model : torch.nn.Module
            The **resolved root** — the same module the freezing loop used. Passing
            the unresolved ``context.model`` here would make every lookup miss and
            the validation vacuous, which is the defect this stage was fixed for.
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
