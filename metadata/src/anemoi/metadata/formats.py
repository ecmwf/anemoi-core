# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Detection of the metadata format and of the version of each component.

Metadata is versioned on the top-level ``version`` key (absent in the oldest
checkpoints).  The structured components (``data``, ``task`` and ``lineage``)
additionally carry their own version, so that a change to one does not affect
the others.  The versions are stamped in the top-level ``component_versions``
map (format ``3.1`` and later)::

    {"version": "3.1", "component_versions": {"data": "1.0", "task": "1.0", "lineage": "1.0"}, ...}

When a component has no stamp, its version is inferred.  The lineage version
follows what the file records, whatever its ``version``: ``checkpoint_lineage``
present gives ``1.0``, else ``uuid`` or ``parent_uuid`` present ``0.2``, else
``0.1`` (nothing recorded).  The ``data`` and ``task`` versions are inferred
from ``version`` and the content:

* no ``metadata_inference`` block (legacy F0 and F1; ``version`` absent or
  ``"1.0"``): component version ``0.1``;
* ``version`` absent, ``"1.x"`` or ``"2.x"`` with ``metadata_inference``:
  component version ``0.2``;
* ``version`` ``"3.x"`` (richer-batch layout with ``data_type`` and
  ``grid_size``): component version ``1.0``;
* ``version`` ``"4"`` or later: unknown future format, read with the newest
  known adapter and a warning; without a ``metadata_inference`` block there is
  nothing newer to read it with, so it is read as legacy, also with a warning.

An absent ``version`` with ``data_type`` or ``grid_size`` in a dataset block
is read as ``3.0`` content.  The adapters registered for these component
versions live in :mod:`anemoi.metadata.components`.
"""

import warnings
from collections.abc import Mapping
from typing import Any

from packaging.version import InvalidVersion
from packaging.version import Version

from .registry import ComponentRegistry

DATA = "data"
TASK = "task"
LINEAGE = "lineage"
COMPONENTS = (DATA, TASK, LINEAGE)

LEGACY_COMPONENT_VERSION = "0.1"
LAYOUT_2_COMPONENT_VERSION = "0.2"
LAYOUT_3_COMPONENT_VERSION = "1.0"

#: Lineage versions, by what the file records: nothing, ``uuid``, ``checkpoint_lineage``.
LINEAGE_NOT_RECORDED = "0.1"
LINEAGE_UUID = "0.2"
LINEAGE_ENTRIES = "1.0"

#: Highest top-level major version whose layout is known.
_NEWEST_KNOWN_MAJOR = 3


def parse_version(value: Any) -> Version | None:
    """Parse a version string, returning ``None`` if absent or invalid."""
    if not isinstance(value, str):
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def is_legacy(raw: Mapping[str, Any]) -> bool:
    """Whether *raw* is a legacy checkpoint: it has no ``metadata_inference`` block."""
    return not isinstance(raw.get("metadata_inference"), dict)


def stamped_versions(raw: Mapping[str, Any]) -> dict[str, str]:
    """Return the component versions stamped in ``component_versions``.

    Entries for components that are not structured components of this
    package, and values that are not strings, are ignored.
    """
    stamps = raw.get("component_versions")
    if not isinstance(stamps, dict):
        return {}
    return {name: value for name, value in stamps.items() if name in COMPONENTS and isinstance(value, str)}


def _has_layout_3_content(raw: Mapping[str, Any]) -> bool:
    inference = raw.get("metadata_inference")
    if not isinstance(inference, dict):
        return False
    names = inference.get("dataset_names")
    for name in names if isinstance(names, list) else []:
        block = inference.get(name)
        if isinstance(block, dict) and ("data_type" in block or "grid_size" in block):
            return True
    return False


def _inferred_lineage_version(raw: Mapping[str, Any]) -> str:
    """Infer the lineage version from what *raw* records, whatever its ``version``."""
    if isinstance(raw.get("checkpoint_lineage"), list):
        return LINEAGE_ENTRIES
    if raw.get("uuid") is not None or raw.get("parent_uuid") is not None:
        return LINEAGE_UUID
    return LINEAGE_NOT_RECORDED


def inferred_version(raw: Mapping[str, Any], component: str) -> str:
    """Infer the version of *component* from ``version`` and the content of *raw*.

    The lineage version follows what the file records, whatever its
    ``version``.  For the other components, a ``version`` newer than any known
    format warns and returns the newest known adapter version.
    """
    if component == LINEAGE:
        return _inferred_lineage_version(raw)
    version = parse_version(raw.get("version"))

    if is_legacy(raw):
        if version is not None and version.major > _NEWEST_KNOWN_MAJOR:
            # Without metadata_inference the newer layout cannot be read: legacy adapters are all there is.
            warnings.warn(
                f"Unknown metadata version {raw.get('version')!r} without metadata_inference; "
                f"reading the {component} component with the legacy adapter ({LEGACY_COMPONENT_VERSION}).",
                UserWarning,
                stacklevel=3,
            )
        return LEGACY_COMPONENT_VERSION

    if version is None:
        if _has_layout_3_content(raw):
            return LAYOUT_3_COMPONENT_VERSION
        return LAYOUT_2_COMPONENT_VERSION
    if version.major <= 2:
        return LAYOUT_2_COMPONENT_VERSION
    if version.major == _NEWEST_KNOWN_MAJOR:
        return LAYOUT_3_COMPONENT_VERSION

    # Importing the components package registers every adapter.
    from .components import ComponentAdapter  # noqa: F401

    newest = ComponentRegistry.newest_version(component, any_name=True)
    warnings.warn(
        f"Unknown metadata version {raw.get('version')!r}; reading the {component} component "
        f"with the newest known adapter ({newest}).",
        UserWarning,
        stacklevel=3,
    )
    return newest


def component_versions(raw: Mapping[str, Any]) -> dict[str, str]:
    """Return the version of each structured component of *raw*.

    ``component_versions`` stamps are used when present; every other
    component is inferred from ``version`` and the content.
    """
    stamps = stamped_versions(raw)
    return {component: stamps.get(component) or inferred_version(raw, component) for component in COMPONENTS}
