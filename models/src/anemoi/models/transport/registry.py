# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import inspect
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def _settings_taken_by(cls: type) -> set[str]:
    """Names of the settings the constructor of ``cls`` takes."""
    parameters = inspect.signature(cls.__init__).parameters.values()
    return {
        parameter.name
        for parameter in parameters
        if parameter.name != "self"
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def build_registered(registry: dict[str, type], name: str, settings: dict[str, Any], context: str, **fixed: Any) -> Any:
    """Build the entry ``name`` of ``registry`` from ``settings``.

    Settings that only other entries of the registry take, such as the Karras ``rho`` when a linear schedule
    is chosen, are left out, so a config can switch between entries without removing them. Settings that no
    entry takes are passed on, and the constructor rejects them.

    Parameters
    ----------
    registry : dict[str, type]
        Classes by name, for example the sigma schedules.
    name : str
        Name of the entry to build.
    settings : dict[str, Any]
        Settings from the config, without the name.
    context : str
        What is built, for error and log messages.
    **fixed : Any
        Further constructor arguments that do not come from the config.
    """
    if name not in registry:
        msg = f"Unknown {context}: {name!r}. Available: {', '.join(sorted(registry))}."
        raise ValueError(msg)
    cls = registry[name]
    taken = _settings_taken_by(cls)
    taken_by_others = set().union(*(_settings_taken_by(other) for other in registry.values())) - taken
    left_out = sorted(key for key in settings if key in taken_by_others)
    if left_out:
        LOGGER.info("The %s %r does not use %s; leaving them out.", context, name, ", ".join(left_out))
    return cls(**{key: value for key, value in settings.items() if key not in taken_by_others}, **fixed)
