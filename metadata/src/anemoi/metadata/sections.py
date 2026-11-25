# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unversioned reader for the raw sections of checkpoint metadata.

Some top-level keys are raw dumps written by training and the checkpoint
callback (``config``, ``dataset``, ``provenance_training``, ``timestamp``,
``training``, ``tracker``, ``model``, ``seed``, ``base_seed``, ``run_id``).
They carry no version of their own: :class:`SectionsReader` returns them as
they are, applying only the
fallbacks needed to read checkpoints from different writers
(``provenance_training`` then ``provenance``; ``timestamp`` then
``created_at``).  A missing section never fails; it reads as ``None`` (or an
empty container where the accessor says so).

The reader is not a component adapter: it is not versioned and has no
registry.  The structured, versioned parts of the metadata (data, task and
lineage, which holds ``uuid``, ``parent_uuid`` and ``checkpoint_lineage``) are
read by the adapters in :mod:`anemoi.metadata.components`.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any

_SEED_KEYS = ("seed", "base_seed")


class SectionsReader:
    """Read raw top-level sections of a metadata dictionary as they are.

    Parameters
    ----------
    raw : Mapping[str, Any]
        The raw metadata dictionary.  It is read, never modified.
    """

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw = raw

    def section(self, name: str, default: Any = None) -> Any:
        """Return the top-level value *name* as stored, or *default*."""
        return self._raw.get(name, default)

    def _dict(self, name: str) -> dict[str, Any]:
        value = self._raw.get(name)
        return value if isinstance(value, dict) else {}

    @property
    def config(self) -> dict[str, Any]:
        """The ``config`` dump, or an empty dict."""
        return self._dict("config")

    @property
    def dataset(self) -> dict[str, Any]:
        """The ``dataset`` dump (keyed by dataset name from format 2.0), or an empty dict."""
        return self._dict("dataset")

    @property
    def provenance(self) -> dict[str, Any]:
        """Provenance: ``provenance_training`` if present, else ``provenance``, else an empty dict."""
        for key in ("provenance_training", "provenance"):
            value = self._raw.get(key)
            if isinstance(value, dict) and value:
                return value
        return {}

    @property
    def precision(self) -> str | None:
        """``config.training.precision``, or ``None`` when not recorded."""
        training = self.config.get("training")
        if isinstance(training, dict):
            return training.get("precision")
        return None

    @property
    def timestamp(self) -> datetime | None:
        """When the metadata was written: ``timestamp``, else ``created_at``, parsed.

        Returns ``None`` when neither is present or the value is not an ISO
        8601 string or a datetime.
        """
        for key in ("timestamp", "created_at"):
            value = self._raw.get(key)
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    continue
        return None

    @property
    def run_id(self) -> str | None:
        """The ``run_id``: top-level, else ``metadata_inference.run_id``."""
        value = self._raw.get("run_id")
        if value is None:
            value = self._dict("metadata_inference").get("run_id")
        return value

    @property
    def seeds(self) -> dict[str, Any]:
        """The recorded ``seed`` and ``base_seed``, as stored.

        Each is read from the top level, else from ``metadata_inference``;
        keys that are recorded nowhere are left out.
        """
        inference = self._dict("metadata_inference")
        found: dict[str, Any] = {}
        for key in _SEED_KEYS:
            value = self._raw.get(key)
            if value is None:
                value = inference.get(key)
            if value is not None:
                found[key] = value
        return found
