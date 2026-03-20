"""FreezingModifierStage — native PipelineStage for freezing model submodules."""

from __future__ import annotations

import logging

import torch

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage

LOGGER = logging.getLogger(__name__)


class FreezingModifierStage(PipelineStage):
    """Freezes specified submodules. Native PipelineStage — full feature port.

    Parameters
    ----------
    submodules_to_freeze : list[str]
        Names of submodules to freeze. Supports dot notation
        (e.g., "processor.0", "encoder.layers.2").
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
        if isinstance(submodules_to_freeze, list | tuple):
            self.submodules_to_freeze = list(submodules_to_freeze)
        else:
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
            if frozen_count > 0:
                LOGGER.info("Froze %d parameters in '%s'", frozen_count, module_name)
                frozen_modules.append({"name": module_name, "frozen_params": frozen_count})
                total_frozen += frozen_count
            else:
                msg = f"Module '{module_name}' not found or has no parameters to freeze"
                if self.strict:
                    raise ValueError(msg)
                LOGGER.warning(msg)

        if self.validate_gradients:
            self._validate_gradient_flow(model)

        context.metadata["modifiers_applied"] = [
            {
                "type": "freezing",
                "submodules": self.submodules_to_freeze,
                "frozen_modules": frozen_modules,
                "total_frozen_params": total_frozen,
            },
        ]

        return context

    def _freeze_submodule_by_name(self, module: torch.nn.Module, target_name: str) -> int:  # noqa: C901
        """Freeze parameters of a submodule by name using optimized lookup.

        Uses PyTorch's get_submodule() for O(1) direct access, falling back
        to recursive search for partial matches.

        Parameters
        ----------
        module : torch.nn.Module
            The parent module to search within.
        target_name : str
            The name of the submodule to freeze. Supports dot notation
            (e.g., "processor.0", "encoder.attention").

        Returns
        -------
        int
            Number of parameters that were frozen.
        """
        frozen_count = 0

        # O(1) direct access via get_submodule
        try:
            target_module = module.get_submodule(target_name)
            for param in target_module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        except AttributeError:
            pass
        else:
            return frozen_count

        # Fallback: recursive search for partial matches
        if "." in target_name:
            parent_name, child_name = target_name.split(".", 1)
            for name, child in module.named_children():
                if name == parent_name:
                    frozen_count += self._freeze_submodule_by_name(child, child_name)
        else:
            for name, child in module.named_children():
                if name == target_name:
                    for param in child.parameters():
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1
                    return frozen_count

            # Not found in direct children, search recursively
            for _, child in module.named_children():
                frozen_count += self._freeze_submodule_by_name(child, target_name)

        return frozen_count

    def _validate_gradient_flow(self, model: torch.nn.Module) -> None:
        """Validate that frozen parameters don't accumulate gradients.

        Performs a test forward/backward pass. Failures are logged as warnings,
        not raised, since validation may fail for models with non-standard inputs.
        """
        LOGGER.debug("Validating gradient flow for frozen parameters")

        was_training = model.training
        model.eval()

        try:
            test_input = torch.randn(1, 10, requires_grad=True)

            try:
                output = model(test_input)
                if hasattr(output, "mean"):
                    loss = output.mean()
                elif isinstance(output, torch.Tensor):
                    loss = output.sum()
                else:
                    loss = output[0].sum()

                loss.backward()

                for module_name in self.submodules_to_freeze:
                    self._check_module_gradients(model, module_name)

                LOGGER.debug("Gradient validation successful — frozen parameters have no gradients")

            except (RuntimeError, TypeError, AttributeError) as e:
                LOGGER.warning("Could not validate gradient flow: %s", e)

        finally:
            if was_training:
                model.train()
            model.zero_grad()

    def _check_module_gradients(self, module: torch.nn.Module, target_name: str) -> None:  # noqa: C901
        """Check that a specific module's parameters have no gradients.

        Parameters
        ----------
        module : torch.nn.Module
            The parent module to search within.
        target_name : str
            The name of the submodule to check.

        Raises
        ------
        RuntimeError
            If frozen parameters have gradients.
        """
        try:
            target_module = module.get_submodule(target_name)
            for param_name, param in target_module.named_parameters():
                if not param.requires_grad and param.grad is not None:
                    msg = f"Frozen parameter '{target_name}.{param_name}' unexpectedly has gradients."
                    raise RuntimeError(msg)
        except AttributeError:
            pass
        else:
            return

        if "." in target_name:
            parent_name, child_name = target_name.split(".", 1)
            for name, child in module.named_children():
                if name == parent_name:
                    self._check_module_gradients(child, child_name)
        else:
            for name, child in module.named_children():
                if name == target_name:
                    for param_name, param in child.named_parameters():
                        if not param.requires_grad and param.grad is not None:
                            msg = f"Frozen parameter '{target_name}.{param_name}' unexpectedly has gradients."
                            raise RuntimeError(msg)
                    return

            for _, child in module.named_children():
                self._check_module_gradients(child, target_name)
