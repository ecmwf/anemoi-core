# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Mapping
from collections.abc import Sequence
from functools import reduce
from importlib.util import find_spec

import torch
import torch_geometric
from numpy import unique
from torch.nn import Module

from anemoi.models.builders import resolve_target

LOGGER = logging.getLogger(__name__)


def _get_compile_entry(
    module: Module,
    compile_config: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Search the compile config for an entry c module name.

    module: Module -> module instance to match against configured targets
    compile_config : Sequence[Mapping[str, object]] -> The 'compile' entry within the models config

    returns: None, if 'module' is not listed within 'compile_config'. Otherwise returns the modules entry.

    """
    for entry in compile_config:
        if resolve_target(entry["module"]) is type(module):
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


def mark_for_compilation(model: Module, compile_config: Sequence[Mapping[str, object]] | None) -> Module:
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
            options = entry.get("options", default_compile_options)

            LOGGER.debug("%s will be compiled with the following options: %s", str(module), str(options))
            # It is just marked for JIT-compilation later
            # It will be compiled before its first forward pass

            module.compile(**options)

            parts = name.split(".")
            parent = reduce(getattr, parts[:-1], model)
            LOGGER.debug("Replacing %s with a compiled version", str(parts[-1]))
            setattr(parent, parts[-1], module)

            compiled_modules.append(entry.get("module"))

    LOGGER.info("The following modules will be compiled: %s", str(unique(compiled_modules)))

    return model
