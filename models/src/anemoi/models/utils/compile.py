# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import reduce
from importlib.util import find_spec

import torch
import torch_geometric
from hydra.utils import get_class
from numpy import unique
from omegaconf import DictConfig
from torch.nn import Module

LOGGER = logging.getLogger(__name__)


def _get_compile_entry(module: str, compile_config: DictConfig) -> DictConfig | None:
    """Search the compile config for an entry c module name.

    module: str -> full module name e.g. 'anemoi.models.layers.conv.GraphTransformerConv'
    compile_config : DictConfig -> The 'compile' entry within the models config

    returns: None, if 'module' is not listed within 'compile_config'. Otherwise returns the modules entry.

    """
    for entry in compile_config:
        if get_class(entry["module"]) is type(module):
            return entry

    return None


def _meets_library_versions_for_compile() -> bool:
    """Returns True if minimum library versions for compilation in Anemoi is met."""
    has_triton = True
    if find_spec("triton") is None:
        msg = "Triton not installed! Consider installing Triton to "
        msg += "enable compilation and improve speed and memory usage."
        LOGGER.warning(msg)
        has_triton = False

    version_req = torch.__version__ >= "2.6" and torch_geometric.__version__ >= "2.6"

    if not version_req:
        msg = "Minimum library versions for compilation not met. "
        msg += f"torch: v{torch.__version__}<2.6 or torch_geometric: v{torch_geometric.__version__}<2.6. "
        msg += "Please upgrade these libraries to enable compilation."
        LOGGER.warning(msg)

    return version_req and has_triton


def _drop_global_state_guards(guard_entries: list) -> list[bool]:
    """Guard filter that drops dynamo's process-global-state guard.

    The global-state guard fails on autograd worker threads, whose thread-local intra-op
    setting (num_threads) differs from the main thread's. Activation checkpointing
    recomputes forwards on those threads, so a compiled module inside a checkpointed
    region recompiles mid-recomputation and the saved-tensor bookkeeping breaks with a
    non-deterministic ``CheckpointError`` (pytorch/pytorch#166926). Remove this workaround
    once fixed upstream.

    The guard is monolithic: dropping it also stops recompiles on changes to grad mode,
    autocast, deterministic algorithms and default dtype (per-field filtering is not
    possible). Two protections remain: tensor guards on ``requires_grad`` keep training
    and inference graphs apart, and autocast changes are still caught through the inputs'
    dispatch key sets. What is lost is only the corner case of a global-state change with
    tensor-identical inputs, which does not occur for modules fed upstream activations.
    Set ``filter_global_state_guards: false`` on a compile entry to opt out.
    """
    return [entry.guard_type != "GLOBAL_STATE" for entry in guard_entries]


def mark_for_compilation(model: Module, compile_config: DictConfig | None) -> Module:
    """Marks modules within 'model' for compilation, according to 'compile_config'.

    Modules are not compiled here. The compilation will occur
    automatically before the first forward iteration.

    returns an updated model, with modules marked for compilation
    """
    if compile_config is None:
        return model

    if not _meets_library_versions_for_compile():
        return model

    default_compile_options = {}
    compiled_modules = []
    # Loop through all modules
    for name, module in model.named_modules():
        entry = _get_compile_entry(module, compile_config)
        # entry is 'None' if compilation was not requested for this module
        if entry is not None:
            options = dict(entry.get("options", default_compile_options))
            # torch.__version__ is a TorchVersion, so this is a version-aware comparison
            if entry.get("filter_global_state_guards", True) and torch.__version__ >= "2.8":
                from torch._inductor import list_mode_options

                inner_options = {}
                mode = options.pop("mode", None)
                if mode is not None:
                    # mode and options are mutually exclusive in torch.compile, so a
                    # requested mode is translated into the equivalent options
                    inner_options.update(list_mode_options(mode, options.get("dynamic")))
                inner_options.update(dict(options.get("options", {})))
                inner_options.setdefault("guard_filter_fn", _drop_global_state_guards)
                options["options"] = inner_options
            elif torch.__version__ < "2.8":
                LOGGER.warning(
                    "torch<2.8 cannot filter dynamo's global-state guard; compiled modules "
                    "inside gradient-checkpointed regions may fail non-deterministically "
                    "with a CheckpointError (pytorch/pytorch#166926)."
                )

            LOGGER.debug("%s will be compiled with the following options: %s", str(module), str(options))
            # It is just marked for JIT-compilation later
            # It will be compiled before its first forward pass

            module.compile(**options)

            parts = name.split(".")
            parent = reduce(getattr, parts[:-1], model)
            LOGGER.debug("Replacing %s with a compiled version", str(parts[-1]))
            setattr(parent, parts[-1], module)

            compiled_modules.append(entry.module)

    LOGGER.info("The following modules will be compiled: %s", str(unique(compiled_modules)))

    return model
