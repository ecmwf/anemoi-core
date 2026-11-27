# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint adapter: the layered, component-based reader of checkpoint metadata.

A :class:`CheckpointAdapter` holds the raw metadata dictionary and composes

* one adapter per structured, versioned component (``data``, ``task`` and
  ``lineage``, see :mod:`anemoi.metadata.components`), chosen from the
  ``component_versions`` stamps or inferred from the layout
  (:mod:`anemoi.metadata.formats`); the data component has one adapter per
  dataset, chosen by the dataset's ``data_type``, and
* the unversioned :class:`~anemoi.metadata.sections.SectionsReader` for the
  raw dumps (``config``, ``dataset``, provenance, timestamp, seeds, ...).

Each component module owns the selection and construction of its adapters
(``build_task_adapter``, ``build_data_adapters``, ``build_lineage_adapter``);
this class only calls them with the versions from :mod:`anemoi.metadata.formats`.
A new format or component version only needs a version reader that overrides what
differs.  The parts are exposed as :meth:`CheckpointAdapter.data` (per dataset),
and the read-only properties ``task``, ``lineage`` and ``sections``.  Reading is lenient:
unknown keys are kept and missing optional keys never fail;
:meth:`CheckpointAdapter.to_dict` returns the raw content unchanged.
"""

import copy
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from packaging.version import Version

from .components.data import DataAdapter
from .components.data import build_data_adapters
from .components.lineage import LineageAdapter
from .components.lineage import build_lineage_adapter
from .components.task import TaskAdapter
from .components.task import build_task_adapter
from .formats import DATA
from .formats import LINEAGE
from .formats import TASK
from .formats import component_versions
from .formats import is_legacy
from .formats import parse_version
from .sections import SectionsReader

#: Raw dumps that are dictionaries; :meth:`CheckpointAdapter.get_section` reads them as ``{}`` when absent.
_DICT_SECTIONS = ("config", "training", "dataset", "environment", "provenance")


class CheckpointAdapter:
    """Reads checkpoint metadata through layered component adapters.

    Parameters
    ----------
    raw : Mapping[str, Any]
        The raw metadata dictionary.  A deep copy is held, so later changes
        to *raw* do not affect the adapter.

    Every format, including the legacy ones without ``metadata_inference``
    (component version ``0.1``), is read through component adapters.

    Attributes
    ----------
    sections : SectionsReader
        Read-only property: the unversioned reader of the raw dumps.
    dataset_names : list[str]
        Read-only property: the ordered dataset names.
    task : TaskAdapter
        Read-only property: the checkpoint-level adapter of the task component
        (the registered version and task name that match); per dataset timing
        is ``checkpoint.task.for_dataset(name)``.
    lineage : LineageAdapter
        Read-only property: the adapter of the lineage component (``uuid``,
        ``parent_uuid`` and the ``checkpoint_lineage`` entries); a value the
        checkpoint's lineage version does not record raises
        :class:`~anemoi.metadata.exceptions.VersionError`.

    Notes
    -----
    The data component has one adapter per dataset, because a checkpoint can
    mix gridded and tabular datasets: ``checkpoint.data("era5")`` is the
    adapter of that dataset, whose methods take no dataset name, for example
    ``checkpoint.data("era5").grid_size()``.  It combines the version reader of
    the data component version with the behaviour of the dataset's ``data_type``
    (see :mod:`anemoi.metadata.components.data`).

    A format or component version that differs from a known one gets its own
    version reader (subclassing the nearest one); this class does not
    change.  The task adapter is selected for
    :func:`~anemoi.metadata.components.task.effective_task_name` (the
    forecaster when none is recorded), so legacy checkpoints read as forecasters.
    """

    def __init__(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            msg = f"metadata must be a mapping, got {type(raw).__name__}"
            raise TypeError(msg)
        self._raw: dict[str, Any] = copy.deepcopy(dict(raw))
        self._sections = SectionsReader(self._raw)
        self._component_versions = component_versions(self._raw)
        self._task: TaskAdapter = build_task_adapter(self._raw, self._component_versions[TASK])
        self._lineage: LineageAdapter = build_lineage_adapter(self._raw, self._component_versions[LINEAGE])
        self._dataset_names: list[str] = self._task.dataset_names()
        self._data: dict[str, DataAdapter] = build_data_adapters(
            self._raw,
            self._component_versions[DATA],
            self._dataset_names,
        )

    # ------------------------------------------------------------------
    # Components and sections
    # ------------------------------------------------------------------

    @property
    def sections(self) -> SectionsReader:
        """The unversioned reader of the raw dumps (config, provenance, seeds, ...)."""
        return self._sections

    @property
    def dataset_names(self) -> list[str]:
        """The ordered dataset names of the checkpoint (a copy); ``["data"]`` for a legacy checkpoint."""
        return list(self._dataset_names)

    def data(self, dataset_name: str) -> DataAdapter:
        """Return the data component adapter of one dataset.

        Raises
        ------
        KeyError
            If *dataset_name* is not one of :attr:`dataset_names`.
        """
        try:
            return self._data[dataset_name]
        except KeyError:
            msg = f"Dataset {dataset_name!r} not found. Available datasets: {self._dataset_names!r}"
            raise KeyError(msg) from None

    @property
    def task(self) -> TaskAdapter:
        """The checkpoint-level task component adapter."""
        return self._task

    @property
    def lineage(self) -> LineageAdapter:
        """The lineage component adapter (``uuid``, ``parent_uuid``, ``entries``)."""
        return self._lineage

    # ------------------------------------------------------------------
    # Envelope
    # ------------------------------------------------------------------

    @property
    def raw(self) -> dict[str, Any]:
        """The raw metadata dictionary held by the adapter (do not modify)."""
        return self._raw

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the raw metadata, unchanged.

        Returns
        -------
        dict[str, Any]
            Deep copy of the raw metadata dictionary, including unknown keys.
        """
        return copy.deepcopy(self._raw)

    @property
    def version(self) -> str | None:
        """The top-level ``version`` string, or ``None`` if absent."""
        value = self._raw.get("version")
        return value if isinstance(value, str) else None

    @property
    def parsed_version(self) -> Version | None:
        """The ``version`` parsed for comparison, or ``None`` if absent or invalid."""
        return parse_version(self._raw.get("version"))

    @property
    def component_versions(self) -> dict[str, str]:
        """Version of each structured component: the stamp when present, otherwise inferred."""
        return dict(self._component_versions)

    @property
    def is_legacy(self) -> bool:
        """Whether this is a legacy checkpoint (no ``metadata_inference`` block)."""
        return is_legacy(self._raw)

    @property
    def created_at(self) -> datetime | None:
        """When the metadata was written (``timestamp``, else ``created_at``), if recorded."""
        return self.sections.timestamp

    def get_section(self, name: str) -> Any:
        """Return a top-level key of the raw metadata as stored.

        An absent key reads as ``None``, except for the raw dumps that are
        dictionaries (``config``, ``training``, ``dataset``, ``environment``,
        ``provenance``), which read as an empty dict.

        Parameters
        ----------
        name : str
            Top-level key (e.g. ``"config"``, ``"training"``).
        """
        if name not in self._raw and name in _DICT_SECTIONS:
            return {}
        return self._raw.get(name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(version={self.version!r}, component_versions={self._component_versions!r})"
