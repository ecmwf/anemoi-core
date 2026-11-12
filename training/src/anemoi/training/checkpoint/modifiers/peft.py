# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pipeline stage that adds PEFT adapters (LoRA, MISS, LoHa, ...) to the graph model."""

from __future__ import annotations

import copy
import enum
import itertools
import logging
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from torch import nn

from anemoi.training.checkpoint.modifiers.base import ModelModifier

if TYPE_CHECKING:
    from peft import PeftConfig

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)

_CONFIG_PATH = "training.checkpoint.modifiers"
_GRAPH_MODEL = "model.model"
_INSTALL_HINT = "pip install anemoi-training[peft]"
# Raised when the stage is built without peft, here or by the pipeline from a config.
MISSING_PEFT_MESSAGE = f"PeftModifierStage needs the peft package. Install it with: {_INSTALL_HINT}"

# Where a checkpoint records the adapters it holds: hyper_parameters["metadata"][_RECORD_KEY].
_RECORD_KEY = "peft"
_RECORD_VERSION = 1
# The configs of adapters merged into the base weights by earlier PEFT stages, oldest first.
_LINEAGE_ATTR = "_anemoi_peft_lineage"
# Config fields that describe where a config came from, not the adapters it builds.
_PROVENANCE_FIELDS = frozenset({"peft_version", "revision", "base_model_name_or_path", "inference_mode"})
# State-dict key parts that only exist while adapters are injected and unmerged.
_ADAPTER_KEY_PARTS = (".base_layer.", ".original_module.", ".modules_to_save.")


class PeftModifierStage(ModelModifier):
    """Add PEFT adapters to the graph model.

    ``peft_config`` is any non-prompt-learning PEFT config (``peft.LoraConfig``,
    ``peft.MissConfig``, ``peft.LoHaConfig``, ...), built by Hydra from its
    ``_target_``. The stage hands it to ``peft.inject_adapter_in_model`` with the
    graph model at ``model.model``, so every option means what the PEFT
    documentation says: ``target_modules`` and ``modules_to_save`` name modules of
    the graph model, matched by PEFT (suffix names, a regex, or ``"all-linear"``),
    and PEFT sets which parameters of the graph model train. Parameters outside the
    graph model keep their ``requires_grad``.

    The model is not wrapped in a ``PeftModel``, whose wrapper would prefix every
    state-dict key with ``base_model.model.`` and replace anemoi's model interface.

    A model that already holds adapters (a loaded PEFT checkpoint) has them merged
    into its base weights first; their configs are kept as lineage in the adapter
    record.

    Parameters
    ----------
    peft_config : peft.PeftConfig
        The adapter method and its options.

    Raises
    ------
    CheckpointConfigError
        If ``peft`` is not installed, ``peft_config`` is not a PEFT config, or it is
        a prompt-learning method.
    """

    def __init__(self, peft_config: PeftConfig) -> None:
        try:
            from peft import PeftConfig
        except ImportError as e:
            self._raise(MISSING_PEFT_MESSAGE, cause=e)

        if not isinstance(peft_config, PeftConfig):
            self._raise(
                "peft_config must be a PEFT config built from its _target_ (e.g. peft.LoraConfig), "
                f"got {type(peft_config).__name__}.",
            )
        if peft_config.is_prompt_learning:
            self._raise(
                f"{type(peft_config).__name__} is a prompt-learning method: it adds virtual tokens to a language "
                "model's input and has no meaning for the graph model.",
            )
        # Hydra hands nested lists over as ListConfig. Rebuilding from plain data gives
        # PEFT plain containers, the same way a restore from a checkpoint record does.
        self.peft_config = PeftConfig.from_peft_type(**_plain(peft_config.to_dict()))
        self._warn_about_options(self.peft_config)

    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Inject the adapters into ``context.model``'s graph model.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context carrying the model to modify.

        Returns
        -------
        CheckpointContext
            The context, with adapters injected and an entry appended to
            ``metadata["modifiers_applied"]``.

        Raises
        ------
        CheckpointConfigError
            If the model has no graph model at ``model.model``, or PEFT refuses the
            config for it (no target matches, a target type it cannot adapt, ...).
        """
        model = context.model
        graph_model = self._graph_model(model)

        lineage = list(getattr(graph_model, _LINEAGE_ATTR, []))
        previous = adapter_record(model)
        if previous is not None:
            merge_adapters(graph_model)
            lineage.append(previous["adapters"])
            LOGGER.info("Merged the model's existing PEFT adapters into the base weights before adding new ones")

        _inject_adapter(graph_model, self.peft_config)
        setattr(graph_model, _LINEAGE_ATTR, lineage)

        adapted = _adapted_module_names(graph_model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        peft_type = _plain(self.peft_config.peft_type)
        LOGGER.info(
            "%s adapters added to %d modules of the graph model at '%s': %s. Trainable parameters: %d of %d.",
            peft_type,
            len(adapted),
            _GRAPH_MODEL,
            adapted,
            n_trainable,
            n_total,
        )

        context.metadata.setdefault("modifiers_applied", []).append(
            {
                "type": "peft",
                "peft_type": peft_type,
                "merged_previous_adapters": previous is not None,
                "adapted_modules": adapted,
                "trainable_params": n_trainable,
                "total_params": n_total,
            },
        )
        return context

    @staticmethod
    def _warn_about_options(peft_config: PeftConfig) -> None:
        """Warn about options that train or change more than the adapters."""
        bias = getattr(peft_config, "bias", "none")
        if bias != "none":
            LOGGER.warning("bias=%r: base-layer biases train beside the adapters.", bias)
        if getattr(peft_config, "lora_bias", False):
            LOGGER.warning("lora_bias=True: the adapters add trainable biases to the base layers.")
        if peft_config.modules_to_save:
            LOGGER.warning(
                "modules_to_save=%s: these modules train in full; their state-dict keys change until merged.",
                sorted(peft_config.modules_to_save),
            )
        init = getattr(peft_config, "init_lora_weights", True)
        if init is not True and init != "gaussian":
            LOGGER.warning(
                "init_lora_weights=%r: either the base weights change at injection, or the adapters do not "
                "start at zero, so the adapted model's output differs from the loaded model's.",
                init,
            )

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
    def _raise(msg: str, cause: Exception | None = None) -> NoReturn:
        """Log and raise a ``CheckpointConfigError`` for the modifiers config."""
        from anemoi.training.checkpoint.exceptions import CheckpointConfigError

        LOGGER.error(msg)
        raise CheckpointConfigError(msg, config_path=_CONFIG_PATH) from cause


def _inject_adapter(graph_model: nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
    """Call ``peft.inject_adapter_in_model``, raising its refusals as config errors."""
    from peft import inject_adapter_in_model

    try:
        inject_adapter_in_model(peft_config, graph_model, adapter_name=adapter_name)
    except ValueError as e:
        PeftModifierStage._raise(f"PEFT could not add {type(peft_config).__name__} adapters: {e}", cause=e)


def _adapted_module_names(graph_model: nn.Module) -> list[str]:
    """Names, relative to the graph model, of the modules PEFT adapted or wrapped."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    return sorted(
        name for name, m in graph_model.named_modules() if isinstance(m, BaseTunerLayer | ModulesToSaveWrapper)
    )


def _plain(value: Any) -> Any:
    """Turn a PEFT config value into plain JSON-safe data.

    Enums become their value, sets sorted lists, and any other sequence or mapping
    (including OmegaConf's ``ListConfig`` and ``DictConfig``) a list or dict.
    """
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(_plain(v) for v in value)
    if isinstance(value, Sequence):
        return [_plain(v) for v in value]
    return value


def _graph_model_or_none(model: nn.Module | None) -> nn.Module | None:
    """Return the module at ``model.model``, or ``None`` when there is none."""
    if model is None:
        return None
    try:
        graph_model = model.get_submodule(_GRAPH_MODEL)
    except AttributeError:
        return None
    return graph_model if isinstance(graph_model, nn.Module) else None


def adapter_record(model: nn.Module | None) -> dict[str, Any] | None:
    """Describe the PEFT adapters ``model`` holds, as plain data for checkpoint metadata.

    The record is derived from the live model, never from the run config, so it is
    right however the adapters got there: a PEFT stage in this run, or adapters
    restored from a checkpoint and trained further without declaring a stage.

    Parameters
    ----------
    model : nn.Module or None
        The training module.

    Returns
    -------
    dict or None
        ``None`` when the graph model carries no adapters. Otherwise ``version``,
        ``adapters`` (adapter name -> PEFT config as a plain dict), ``lineage``
        (the ``adapters`` of earlier, merged levels, oldest first) and ``merged``.
    """
    graph_model = _graph_model_or_none(model)
    if graph_model is None or not getattr(graph_model, "peft_config", None):
        return None
    return {
        "version": _RECORD_VERSION,
        "adapters": {name: _plain(config.to_dict()) for name, config in graph_model.peft_config.items()},
        "lineage": list(getattr(graph_model, _LINEAGE_ATTR, [])),
        "merged": False,
    }


def _without_provenance(adapters: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {key: value for key, value in config.items() if key not in _PROVENANCE_FIELDS}
        for name, config in adapters.items()
    }


def restore_checkpoint_adapters(checkpoint: Mapping[str, Any], model: nn.Module | None) -> None:
    """Make the model hold the adapters the checkpoint records, before a strict load.

    Called from :func:`~anemoi.training.checkpoint.loading.base.apply_checkpoint_corrections`,
    so the pipeline loaders and the Lightning resume hook rebuild adapters the same way.
    Each recorded config is rebuilt with ``PeftConfig.from_peft_type`` and injected,
    so PEFT re-matches the targets and re-marks the trainable parameters. If that
    yields other modules than the checkpoint holds, the strict load that follows
    fails on the missing and unexpected keys.

    Parameters
    ----------
    checkpoint : Mapping
        The loaded checkpoint dict.
    model : nn.Module or None
        The training module the checkpoint loads into.

    Raises
    ------
    CheckpointIncompatibleError
        If the checkpoint has adapter keys but no record, or the model already holds
        adapters whose configs differ from the recorded ones.
    CheckpointConfigError
        If the checkpoint holds adapters and ``peft`` is not installed.
    """
    from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

    metadata = (checkpoint.get("hyper_parameters") or {}).get("metadata") or {}
    record = metadata.get(_RECORD_KEY)
    if record is None:
        state_dict = checkpoint.get("state_dict") or {}
        adapter_keys = sorted(key for key in state_dict if any(part in key for part in _ADAPTER_KEY_PARTS))
        if adapter_keys:
            msg = (
                "The checkpoint holds PEFT adapter weights but no adapter record in its metadata, so the "
                f"adapters cannot be rebuilt (e.g. {adapter_keys[0]}). Refusing to load it into a plain model, "
                "where a non-strict load would silently skip the adapted modules."
            )
            raise CheckpointIncompatibleError(msg, unexpected_keys=adapter_keys)
        return
    if record.get("merged") or model is None:
        return

    live = adapter_record(model)
    if live is None:
        _restore_adapters(model, record)
        return
    expected, found = _without_provenance(record["adapters"]), _without_provenance(live["adapters"])
    if found != expected:
        differing = sorted(
            f"{name}.{key}"
            for name in expected.keys() | found.keys()
            for key in expected.get(name, {}).keys() | found.get(name, {}).keys()
            if expected.get(name, {}).get(key) != found.get(name, {}).get(key)
        )
        msg = f"The model already holds PEFT adapters that differ from the ones the checkpoint records: {differing}."
        raise CheckpointIncompatibleError(msg)
    LOGGER.debug("The model already holds the adapters the checkpoint records; nothing to restore")


def _restore_adapters(model: nn.Module, record: Mapping[str, Any]) -> None:
    """Inject the adapters ``record`` describes into the model's graph model."""
    try:
        from peft import PeftConfig
    except ImportError as e:
        msg = f"The checkpoint holds PEFT adapters; loading it needs the peft package. Install it with: {_INSTALL_HINT}"
        PeftModifierStage._raise(msg, cause=e)

    graph_model = PeftModifierStage._graph_model(model)
    for name, config in record["adapters"].items():
        # from_peft_type picks the config class from peft_type, and drops (with a
        # warning) fields this peft version does not know.
        _inject_adapter(graph_model, PeftConfig.from_peft_type(**config), adapter_name=name)
    setattr(graph_model, _LINEAGE_ATTR, list(record.get("lineage", [])))
    LOGGER.info(
        "Restored the PEFT adapters recorded in the checkpoint on: %s",
        _adapted_module_names(graph_model),
    )


def merge_adapters(module: nn.Module) -> None:
    """Merge every PEFT adapter in ``module`` into its base layer, in place.

    Each adapted layer is replaced by its base layer carrying the merged weights, and
    each ``modules_to_save`` wrapper by its trained copy, so the state-dict keys are
    the plain model's again. The ``peft_config`` that ``inject_adapter_in_model``
    leaves behind is dropped: nothing in ``module`` refers to ``peft`` afterwards.

    Parameters
    ----------
    module : nn.Module
        The module holding the adapters, e.g. the graph model or the model interface.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    for name, layer in list(module.named_modules()):
        if not name:
            continue
        if isinstance(layer, BaseTunerLayer):
            layer.merge()
            replacement = layer.get_base_layer()
        elif isinstance(layer, ModulesToSaveWrapper):
            replacement = layer.modules_to_save[layer.active_adapters[0]]
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent) if parent else module, child, replacement)
    for submodule in module.modules():
        submodule.__dict__.pop("peft_config", None)


def merged_copy_on_cpu(model: nn.Module) -> nn.Module:
    """Return a CPU copy of ``model`` with its adapters merged, leaving ``model`` as it is.

    The copy is built on the CPU, so saving an inference checkpoint costs no extra
    accelerator memory. Merging in place and unmerging afterwards is not an option:
    the round trip is not exact in floating point, so the frozen base weights on the
    saving rank would drift from the other ranks, and DDP never syncs frozen weights.

    Parameters
    ----------
    model : nn.Module
        The module to copy, e.g. the model interface the inference checkpoint pickles.

    Returns
    -------
    nn.Module
        The merged copy, free of any reference to ``peft``.
    """
    memo: dict[int, Any] = {}
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if id(tensor) not in memo:
            cpu = tensor.detach().to("cpu", copy=True)
            memo[id(tensor)] = (
                nn.Parameter(cpu, requires_grad=tensor.requires_grad) if isinstance(tensor, nn.Parameter) else cpu
            )
    merged = copy.deepcopy(model, memo)
    merge_adapters(merged)
    for submodule in merged.modules():
        submodule.__dict__.pop(_LINEAGE_ATTR, None)
    return merged


def has_adapters(module: nn.Module) -> bool:
    """Whether any part of ``module`` holds PEFT adapters, checked without importing ``peft``.

    ``inject_adapter_in_model`` leaves ``peft_config`` on the module it adapted and
    :func:`merge_adapters` removes it, so its presence marks unmerged adapters.

    Parameters
    ----------
    module : nn.Module
        Any module, e.g. the training module or the model interface.

    Returns
    -------
    bool
        ``True`` when unmerged adapters are present.
    """
    return any("peft_config" in submodule.__dict__ for submodule in module.modules())


def write_adapter_record(checkpoint: dict[str, Any], model: nn.Module) -> None:
    """Record the adapters ``model`` holds in ``checkpoint["hyper_parameters"]["metadata"]``.

    Called by ``BaseTrainingModule.on_save_checkpoint``, so every Lightning save
    carries the record, whichever callback or call triggered it. The record is
    derived from the live model (see :func:`adapter_record`); a model without
    adapters drops any stale record. Lightning hands the hook the live
    hyper-parameters, so the containers are copied rather than changed in place.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dict Lightning is about to write.
    model : nn.Module
        The training module.
    """
    hyper_parameters = checkpoint.get("hyper_parameters")
    if hyper_parameters is None:
        return
    record = adapter_record(model)
    metadata = hyper_parameters.get("metadata") or {}
    if record is None and _RECORD_KEY not in metadata:
        return

    metadata = dict(metadata)
    if record is None:
        metadata.pop(_RECORD_KEY, None)
    else:
        metadata[_RECORD_KEY] = record
    hyper_parameters = copy.copy(hyper_parameters)
    hyper_parameters["metadata"] = metadata
    checkpoint["hyper_parameters"] = hyper_parameters
