# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pipeline stage that adds PEFT adapters (LoRA) to the graph model."""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from torch import nn

from anemoi.training.checkpoint.modifiers.base import ModelModifier

if TYPE_CHECKING:
    from peft import LoraConfig

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)

_CONFIG_PATH = "training.checkpoint.modifiers"
_GRAPH_MODEL = "model.model"
_ALL_LINEAR = "all-linear"
_INSTALL_HINT = "pip install anemoi-training[peft]"

# LoraConfig options the stage refuses. Each one either breaks a guarantee the stage
# gives (only the adapted layers' keys change, the base weights stay as loaded, the
# adapters plus ``trainable_modules`` are all that trains) or is untested with anemoi
# models. The predicate says which values are allowed.
_REJECTED_OPTIONS: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    ("modules_to_save", lambda v: not v, "it wraps whole modules and changes their keys; use trainable_modules"),
    (
        "init_lora_weights",
        lambda v: v is True or v == "gaussian",
        "only True or 'gaussian' leave the base weights as loaded",
    ),
    ("bias", lambda v: v == "none", "it trains base-layer biases outside the adapters"),
    ("lora_bias", lambda v: not v, "it adds trainable biases outside the adapters"),
    ("use_dora", lambda v: not v, "DoRA is not supported yet"),
    ("layers_to_transform", lambda v: v is None, "target_modules selects the layers"),
    ("layers_pattern", lambda v: not v, "target_modules selects the layers"),
    ("rank_pattern", lambda v: not v, "per-module ranks are not supported yet"),
    ("alpha_pattern", lambda v: not v, "per-module alphas are not supported yet"),
    ("exclude_modules", lambda v: not v, "target_modules selects the layers"),
    ("target_parameters", lambda v: not v, "only Linear modules can be adapted"),
)


class PeftModifierStage(ModelModifier):
    """Add LoRA adapters to the graph model.

    Adapters are injected with PEFT's low-level ``inject_adapter_in_model`` into the
    graph model at ``model.model`` (not wrapped in a ``PeftModel``), so the module
    tree stays anemoi's and only the adapted layers' state-dict keys change. After
    injection only the adapters and ``trainable_modules`` train; every other
    parameter of ``context.model`` is frozen.

    Paths are full paths from ``context.model``, as in
    :class:`~anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage`.
    ``target_modules`` in ``peft_config`` is one of:

    - ``"all-linear"``: every ``nn.Linear`` in the graph model;
    - a list of exact full paths (``model.model.processor.0.lin_query``); a bare
      name such as ``lin_query`` is not a path and does not match;
    - a regular expression, matched in full against full paths; it selects the
      ``nn.Linear`` modules among its matches.

    Parameters
    ----------
    peft_config : Mapping or peft.LoraConfig
        LoRA options as a plain mapping (``r``, ``lora_alpha``, ``target_modules``,
        ...), or a ``LoraConfig``. Only LoRA is supported, and options that would
        change other modules' keys or the base weights are refused (see
        ``_REJECTED_OPTIONS``).
    trainable_modules : list[str], optional
        Full paths of modules to keep trainable next to the adapters.

    Raises
    ------
    CheckpointConfigError
        If ``peft`` is not installed, the config is not LoRA, or it uses a refused
        option.
    """

    def __init__(
        self,
        peft_config: Mapping[str, Any] | LoraConfig,
        trainable_modules: list[str] | None = None,
    ) -> None:
        self.lora_config = self._build_lora_config(peft_config)
        self.trainable_modules = list(trainable_modules or [])

        LOGGER.debug(
            "Initialized PeftModifierStage (r=%s, target_modules=%s, trainable_modules=%s)",
            self.lora_config.r,
            self.lora_config.target_modules,
            self.trainable_modules,
        )

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Inject LoRA adapters into ``context.model``'s graph model.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context carrying the model to modify.

        Returns
        -------
        CheckpointContext
            The context, with adapters injected, the trainable set applied and a
            record appended to ``metadata["modifiers_applied"]``.

        Raises
        ------
        CheckpointConfigError
            If the model has no graph model at ``model.model``, already carries
            adapters, a target or trainable path does not resolve, or nothing matches.
        """
        from peft import inject_adapter_in_model

        model = context.model
        graph_model = self._graph_model(model)
        if self._adapted_layers(graph_model):
            self._raise(
                "The model already carries PEFT adapters; adding a second set on top is not supported yet.",
            )

        targets = self._resolve_targets(graph_model, self.lora_config.target_modules)

        # Hand PEFT the resolved modules as one exact regex. PEFT matches a list of names
        # by suffix, which could pick up a deeper module whose path ends the same way.
        explicit = copy.deepcopy(self.lora_config)
        explicit.target_modules = "|".join(re.escape(name) for name in targets)
        inject_adapter_in_model(explicit, graph_model)

        adapted = self._adapted_layers(graph_model)
        if sorted(adapted) != sorted(targets):
            self._raise(f"PEFT adapted {sorted(adapted)} but the stage resolved {sorted(targets)}.")

        self._apply_trainable_set(model, adapted.values())

        adapted_paths = [f"{_GRAPH_MODEL}.{name}" for name in targets]
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        LOGGER.info(
            "LoRA (r=%d) added to %d layers: %s. Trainable parameters: %d of %d.",
            self.lora_config.r,
            len(adapted_paths),
            adapted_paths,
            n_trainable,
            n_total,
        )

        context.metadata.setdefault("modifiers_applied", []).append(
            {
                "type": "peft",
                "adapter": "lora",
                "adapted_modules": adapted_paths,
                "trainable_modules": self.trainable_modules,
                "trainable_params": n_trainable,
                "total_params": n_total,
            },
        )
        return context

    @classmethod
    def _build_lora_config(cls, peft_config: Mapping[str, Any] | LoraConfig) -> LoraConfig:
        """Build and vet the ``LoraConfig``.

        Parameters
        ----------
        peft_config : Mapping or peft.LoraConfig
            The configured LoRA options.

        Returns
        -------
        peft.LoraConfig
            The vetted config.

        Raises
        ------
        CheckpointConfigError
            If ``peft`` is missing, an option is unknown or refused, or the config is
            not a ``LoraConfig``.
        """
        try:
            from peft import LoraConfig
        except ImportError as e:
            cls._raise(f"PeftModifierStage needs the peft package. Install it with: {_INSTALL_HINT}", cause=e)

        if isinstance(peft_config, Mapping):
            from omegaconf import DictConfig
            from omegaconf import OmegaConf

            options = (
                OmegaConf.to_container(peft_config, resolve=True)
                if isinstance(peft_config, DictConfig)
                else dict(peft_config)
            )
            try:
                lora_config = LoraConfig(**options)
            except (TypeError, ValueError) as e:
                cls._raise(f"Invalid LoRA option in peft_config: {e}", cause=e)
        elif type(peft_config) is LoraConfig:
            lora_config = peft_config
        else:
            cls._raise(f"PeftModifierStage supports LoraConfig only, got {type(peft_config).__name__}.")

        refused = [
            f"{option}={getattr(lora_config, option)!r} ({reason})"
            for option, allowed, reason in _REJECTED_OPTIONS
            if not allowed(getattr(lora_config, option, None))
        ]
        if refused:
            cls._raise(f"PeftModifierStage refuses these LoRA options: {'; '.join(refused)}.")
        return lora_config

    @classmethod
    def _graph_model(cls, model: nn.Module | None) -> nn.Module:
        """Return the graph model at ``model.model`` that adapters attach to."""
        if model is None:
            cls._raise("PeftModifierStage needs a model on the context, but context.model is None.")
        try:
            return model.get_submodule(_GRAPH_MODEL)
        except AttributeError as e:
            cls._raise(
                f"PeftModifierStage attaches adapters to the graph model at '{_GRAPH_MODEL}', which the model lacks.",
                cause=e,
            )

    @staticmethod
    def _adapted_layers(module: nn.Module) -> dict[str, nn.Module]:
        """Return the PEFT tuner layers in ``module`` by name."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        return {name: m for name, m in module.named_modules() if isinstance(m, BaseTunerLayer)}

    @classmethod
    def _resolve_targets(cls, graph_model: nn.Module, spec: Any) -> list[str]:
        """Resolve ``target_modules`` to Linear module names relative to the graph model.

        Parameters
        ----------
        graph_model : nn.Module
            The module at ``model.model``.
        spec : str, set or list
            ``"all-linear"``, a regex over full paths, or a collection of exact full paths.

        Returns
        -------
        list[str]
            Names relative to ``graph_model``, sorted.

        Raises
        ------
        CheckpointConfigError
            If ``spec`` is missing, matches nothing, names a path outside the graph
            model or one that does not exist, or selects a module that is not Linear.
        """
        prefix = f"{_GRAPH_MODEL}."
        modules = {name: m for name, m in graph_model.named_modules() if name}

        if not spec:
            cls._raise("peft_config.target_modules is required: 'all-linear', a regex, or a list of full paths.")

        if spec == _ALL_LINEAR:
            selected = [name for name, m in modules.items() if isinstance(m, nn.Linear)]
        elif isinstance(spec, str):
            # A regex selects the Linear modules among its matches, so a pattern over a
            # whole subtree (``model\.model\.processor\..*``) adapts its Linear layers.
            selected = [
                name for name, m in modules.items() if isinstance(m, nn.Linear) and re.fullmatch(spec, prefix + name)
            ]
            if not selected:
                cls._raise(f"target_modules regex {spec!r} matched no nn.Linear module under '{_GRAPH_MODEL}'.")
        else:
            outside = sorted(path for path in spec if not path.startswith(prefix))
            if outside:
                cls._raise(f"target_modules must be full paths under '{_GRAPH_MODEL}': {outside}.")
            missing = sorted(path for path in spec if path[len(prefix) :] not in modules)
            if missing:
                cls._raise(f"target_modules paths not found (full paths from the training module): {missing}.")
            selected = [path[len(prefix) :] for path in spec]

        not_linear = sorted(prefix + name for name in selected if not isinstance(modules[name], nn.Linear))
        if not_linear:
            cls._raise(f"target_modules selects modules that are not nn.Linear: {not_linear}.")
        if not selected:
            cls._raise(f"target_modules {spec!r} selected no Linear module under '{_GRAPH_MODEL}'.")
        return sorted(selected)

    def _apply_trainable_set(self, model: nn.Module, adapted_layers: Any) -> None:
        """Freeze everything, then unfreeze the adapters and ``trainable_modules``."""
        missing = []
        keep = []
        for path in self.trainable_modules:
            try:
                keep.append(model.get_submodule(path))
            except AttributeError:
                missing.append(path)
        if missing:
            self._raise(f"trainable_modules paths not found (full paths from the training module): {missing}.")

        model.requires_grad_(False)
        for layer in adapted_layers:
            for adapter_name in layer.adapter_layer_names:
                getattr(layer, adapter_name).requires_grad_(True)
        for module in keep:
            module.requires_grad_(True)

    @staticmethod
    def _raise(msg: str, cause: Exception | None = None) -> NoReturn:
        """Log and raise a ``CheckpointConfigError`` for the modifiers config."""
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        LOGGER.error(msg)
        raise CheckpointConfigError(msg, config_path=_CONFIG_PATH) from cause
