# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any


def build_registered(registry: dict[str, type], name: str, settings: dict[str, Any], context: str, **fixed: Any) -> Any:
    """Build the entry ``name`` of ``registry`` from ``settings``.

    Every setting goes to the constructor, which rejects any it does not take.

    Parameters
    ----------
    registry : dict[str, type]
        Classes by name, for example the sigma schedules.
    name : str
        Name of the entry to build.
    settings : dict[str, Any]
        Settings from the config, without the name.
    context : str
        What is built, for the error message of an unknown name.
    **fixed : Any
        Further constructor arguments that do not come from the config.
    """
    if name not in registry:
        msg = f"Unknown {context}: {name!r}. Available: {', '.join(sorted(registry))}."
        raise ValueError(msg)
    return registry[name](**settings, **fixed)
