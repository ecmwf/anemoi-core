# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Lineage component adapters.

The lineage component says where a checkpoint comes from: its own ``uuid``,
the ``uuid`` of the checkpoint it was loaded from (``parent_uuid``) and the
``checkpoint_lineage`` entries, one per checkpoint load (the loading strategy
and the ``uuid`` of the file loaded).  The checkpoint callback writes ``uuid``
when it saves; the loading code writes the other two.  It is a checkpoint
level component with one adapter per version.

Component versions, inferred from the content of an unstamped file
(:func:`anemoi.metadata.formats.inferred_version`):

* ``0.1``: nothing recorded (formats F0 and some F1).  Every value raises
  :class:`~anemoi.metadata.exceptions.VersionError`.
* ``0.2``: ``uuid`` recorded, and ``parent_uuid`` when the run was warm started
  from a checkpoint (formats F1 to F5, and the later ones until the lineage
  list).  There is no ``checkpoint_lineage``: :attr:`LineageAdapter.entries` raises.
* ``1.0``: ``checkpoint_lineage`` recorded (the list exists even when it is
  empty), as written by training since the lineage list was introduced (format
  F6, also the ``3.x`` formats).

``parent_uuid`` and "no parent"
-------------------------------

From ``0.2`` an absent ``parent_uuid`` means *no parent* and reads as ``None``,
not as *unknown*: ``0.2`` writers store the key only for a warm start, and a
fresh run of a ``1.0`` writer records ``checkpoint_lineage == []`` and no
``parent_uuid``.  (A stored ``None`` is a run loaded from a file that had no
``uuid``.)  *Not recorded* is the ``0.1`` case, which raises.

In ``1.0`` the entries are authoritative when the scalar is missing: if
``parent_uuid`` is absent (or ``None``) but the last ``checkpoint_lineage``
entry records a parent, :attr:`LineageAdapter.parent_uuid` returns that
parent.  The strict write schema rejects such a file, but reading stays
lenient.

A new version subclasses the nearest one, registers itself with
:meth:`~anemoi.metadata.registry.ComponentRegistry.register` and gates what it
adds with :func:`~anemoi.metadata.components.base.requires`.
"""

import copy
from collections.abc import Mapping
from typing import Any
from typing import cast

from anemoi.metadata.exceptions import VersionError
from anemoi.metadata.registry import ComponentRegistry

from .base import ComponentAdapter
from .base import requires


class LineageAdapter(ComponentAdapter):
    """Reads ``uuid``, ``parent_uuid`` and ``checkpoint_lineage``, each gated on the version that records it."""

    @property
    @requires("0.2")
    def uuid(self) -> str | None:
        """The ``uuid`` of this checkpoint (``None`` when a stamped file lacks it)."""
        return self._raw.get("uuid")

    @property
    @requires("0.2")
    def parent_uuid(self) -> str | None:
        """The ``uuid`` of the checkpoint this one was loaded from; ``None`` when it has no parent.

        From ``1.0``, when the scalar ``parent_uuid`` is absent the parent of
        the last ``checkpoint_lineage`` entry is returned if it records one.
        """
        parent = self._raw.get("parent_uuid")
        if parent is None:
            try:
                entries = self.entries
            except VersionError:  # before 1.0 there are no entries to fall back on
                return None
            last = entries[-1] if entries else None
            if isinstance(last, dict):
                parent = last.get("parent_uuid")
        return parent

    @property
    @requires("1.0")
    def entries(self) -> list[dict[str, Any]]:
        """The ``checkpoint_lineage`` entries, oldest first: ``{"loading_strategy": ..., "parent_uuid": ...}``.

        Empty for a fresh run.
        """
        value = self._raw.get("checkpoint_lineage")
        return copy.deepcopy(value) if isinstance(value, list) else []


def build_lineage_adapter(raw: Mapping[str, object], version: str) -> LineageAdapter:
    """Build the lineage adapter of a checkpoint for the lineage component *version* (stamped or inferred)."""
    return cast("type[LineageAdapter]", ComponentRegistry.resolve("lineage", version))(raw)


@ComponentRegistry.register("lineage", "0.1")
class LineageV0_1(LineageAdapter):
    """Lineage ``0.1``: nothing is recorded, every value raises ``VersionError``."""


@ComponentRegistry.register("lineage", "0.2")
class LineageV0_2(LineageV0_1):
    """Lineage ``0.2``: ``uuid`` and, for a warm start, ``parent_uuid``."""


@ComponentRegistry.register("lineage", "1.0")
class LineageV1_0(LineageV0_2):
    """Lineage ``1.0``: adds the ``checkpoint_lineage`` entries."""
