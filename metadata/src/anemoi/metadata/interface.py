# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""User-facing metadata interface.

:class:`Metadata` wraps a :class:`~anemoi.metadata.base.CheckpointAdapter` and
is organised by component:

* checkpoint level: :attr:`Metadata.version`, :attr:`Metadata.component_versions`,
  :attr:`Metadata.dataset_names`, :attr:`Metadata.task` (one task per checkpoint, see :class:`TaskView`),
  :attr:`Metadata.lineage` (``uuid``, ``parent_uuid``, ``entries``, see :class:`LineageView`) and
  :attr:`Metadata.sections` (the raw dumps, see :class:`SectionsView`);
* per dataset: :meth:`Metadata.dataset` and :attr:`Metadata.datasets` give a
  :class:`DatasetView` with ``name``, ``data`` (the data component bound to
  the dataset, see :class:`DatasetData`), ``arrays`` (the supporting arrays,
  see :class:`~anemoi.metadata.arrays.DatasetArrays`).

There are four single-dataset shortcuts (``variables``, ``grid_size``,
``latitudes``, ``longitudes``); they raise on a checkpoint with several
datasets instead of picking the first.

This module holds no knowledge of any metadata format: every value comes from
a component adapter, the sections reader or the arrays view.  Everything
returned (lists, dicts, arrays) is a copy, so changing it never changes the
instance.  :meth:`Metadata.with_overrides` returns a new instance.
"""

import copy
import sys
from collections.abc import Mapping
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import TypedDict

import numpy as np

from .arrays import DatasetArrays
from .arrays import merge_supporting_arrays
from .arrays import split_by_dataset
from .exceptions import MetadataError

if TYPE_CHECKING:
    from anemoi.transform.variables import Variable

    from .base import CheckpointAdapter
    from .components.data import DataAdapter
    from .components.lineage import LineageAdapter
    from .components.task import TaskAdapter
    from .components.task import TaskTiming
    from .sections import SectionsReader

_MISSING = object()


class DatasetSelectionError(MetadataError, ValueError):
    """Raised when a dataset must be named but was not, or the checkpoint has none."""


class VariableTypes(TypedDict):
    """Variable names by role in the model."""

    forcing: list[str]
    prognostic: list[str]
    diagnostic: list[str]
    target: list[str]


# ---------------------------------------------------------------------------
# Data component bound to one dataset
# ---------------------------------------------------------------------------


class DatasetData:
    """The data component of one dataset: a read-only view of its data adapter.

    Every method is that of the per-dataset data component adapter
    (:class:`~anemoi.metadata.components.data.DataAdapter`).  A tabular
    dataset has no grid: its grid values are ``None`` (*not applicable*),
    distinct from an empty value (*not recorded*).

    Parameters
    ----------
    data : DataAdapter
        The data component adapter of the dataset.
    """

    def __init__(self, data: "DataAdapter") -> None:
        self._data = data

    # -- kind ---------------------------------------------------------------

    def data_type(self) -> str:
        """Return the dataset kind: ``"gridded"`` or ``"tabular"``."""
        return self._data.data_type()

    def is_gridded(self) -> bool:
        """Return whether the dataset is gridded."""
        return self._data.is_gridded()

    def is_tabular(self) -> bool:
        """Return whether the dataset is tabular (no grid)."""
        return self._data.is_tabular()

    def is_input_only(self) -> bool:
        """Return whether the dataset has input variables but no output variables."""
        return self._data.is_input_only()

    # -- variables and indices ---------------------------------------------

    def variables(self) -> list[str]:
        """Return every variable name, inputs (in tensor order) then output-only variables."""
        return self._data.variables()

    def variable_indices(self) -> dict[str, int]:
        """Return the input variable name to tensor index mapping."""
        return self._data.variable_indices()

    def output_variable_indices(self) -> dict[str, int]:
        """Return the output variable name to tensor index mapping (empty if input-only)."""
        return self._data.output_variable_indices()

    def input_index_to_variable(self) -> dict[int, str]:
        """Return the input tensor index to variable name mapping."""
        return self._data.input_index_to_variable()

    def output_index_to_variable(self) -> dict[int, str]:
        """Return the output tensor index to variable name mapping."""
        return self._data.output_index_to_variable()

    def prognostic_input_mask(self) -> list[int]:
        """Return the sorted input tensor indices of the prognostic variables."""
        return self._data.prognostic_input_mask()

    def prognostic_output_mask(self) -> list[int]:
        """Return the sorted output tensor indices of the prognostic variables."""
        return self._data.prognostic_output_mask()

    def variable_types(self) -> VariableTypes:
        """Return the variable names by role (forcing, prognostic, diagnostic, target)."""
        types = self._data.variable_types()
        return VariableTypes(
            forcing=types["forcing"],
            prognostic=types["prognostic"],
            diagnostic=types["diagnostic"],
            target=types["target"],
        )

    def variable_categories(self, *, per_variable: bool = False) -> dict[str, list[str]]:
        """Categorise the variables by role, adding ``computed``, ``accumulation`` and ``constant``.

        With *per_variable* the result is ``{variable: [category, ...]}``,
        otherwise ``{category: [variable, ...]}``.
        """
        return self._data.variable_categories(per_variable=per_variable)

    def select_variables(self, *, include: list[str] | None = None, exclude: list[str] | None = None) -> list[str]:
        """Select variables by category expression (``"prognostic"``, ``"prognostic+constant"``).

        Unknown categories raise ``ValueError``; there is no substring matching.
        """
        return self._data.select_variables(include=include, exclude=exclude)

    def typed_variables(self) -> "dict[str, Variable]":
        """Return an ``anemoi.transform`` ``Variable`` per variable (needs ``anemoi-transform``)."""
        return self._data.typed_variables()

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata (MARS keys, flags, units); empty if not recorded."""
        return self._data.variables_metadata()

    def accumulations(self) -> list[str]:
        """Return the variables that are accumulations."""
        return self._data.accumulations()

    def computed_forcings(self) -> list[str]:
        """Return the variables computed at inference time."""
        return self._data.computed_forcings()

    # -- grid and data source ----------------------------------------------

    def grid_size(self) -> int | None:
        """Return the number of grid points; ``None`` if tabular or not recorded."""
        return self._data.grid_size()

    def data_request(self) -> dict[str, Any] | None:
        """Return the data request; ``None`` if tabular, an empty dict if not recorded."""
        return self._data.data_request()

    def grid(self) -> Any:
        """Return the grid of the data request (e.g. ``"O96"``); ``None`` if tabular or not recorded."""
        return self._data.grid()

    def area(self) -> Any:
        """Return the area of the data request; ``None`` if tabular or not recorded."""
        return self._data.area()

    def sources(self) -> list[dict[str, Any]]:
        """Return the source dataset configurations; empty if not recorded."""
        return self._data.sources()

    def open_dataset_args(self) -> dict[str, Any]:
        """Return the arguments for opening the training dataset; empty if not recorded."""
        return self._data.open_dataset_args()

    def data_frequency(self) -> str | None:
        """Return the data frequency (e.g. ``"6h"``), or ``None`` if not recorded."""
        return self._data.data_frequency()

    def dataloader_config(self, partition: str = "training") -> dict[str, Any]:
        """Return the dataloader dataset configuration of a partition; empty if absent."""
        return self._data.dataloader_config(partition)

    def __repr__(self) -> str:
        return f"DatasetData(name={self._data.dataset_name!r})"


# ---------------------------------------------------------------------------
# Task component (checkpoint level)
# ---------------------------------------------------------------------------


class TaskView:
    """Read-only view of the task component of a checkpoint.

    A checkpoint has one task.  Timing values are read from every dataset and
    exposed as one value when the datasets agree; when they disagree, access
    without a dataset raises :class:`~anemoi.metadata.components.task.TaskTimingError`
    and :meth:`for_dataset` gives the timing of one dataset.

    Parameters
    ----------
    task : TaskAdapter
        The task component adapter of the checkpoint.
    """

    def __init__(self, task: "TaskAdapter") -> None:
        self._task = task

    @property
    def task_name(self) -> str:
        """The task name (``"forecaster"`` when none is recorded)."""
        return self._task.task_name

    @property
    def timing(self) -> "TaskTiming":
        """The timing shared by all datasets (raises ``TaskTimingError`` if they differ)."""
        return self._task.timing

    def for_dataset(self, dataset_name: str) -> "TaskTiming":
        """Return the timing of one dataset (``KeyError`` for an unknown name)."""
        return self._task.for_dataset(dataset_name)

    def timings(self) -> "dict[str, TaskTiming]":
        """Return the timing of every dataset, by dataset name."""
        return self._task.timings()

    def timestep(self, dataset_name: str | None = None) -> str | None:
        """Return the model timestep (e.g. ``"6h"``); the shared value when no dataset is given."""
        return self._task.timestep(dataset_name)

    def input_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return the relative date indices of the model input."""
        return self._task.input_relative_date_indices(dataset_name)

    def output_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return the relative date indices of the model output."""
        return self._task.output_relative_date_indices(dataset_name)

    @property
    def n_step_input(self) -> int | None:
        """Number of input steps shared by all datasets."""
        return self._task.n_step_input

    @property
    def n_step_output(self) -> int | None:
        """Number of output steps shared by all datasets."""
        return self._task.n_step_output

    @property
    def lagged(self) -> list[timedelta] | None:
        """Input times relative to the last input, as timedeltas (ascending, not positive)."""
        return self._task.lagged

    @property
    def input_offsets(self) -> list[timedelta] | None:
        """Input times relative to the last input, as timedeltas."""
        return self._task.input_offsets

    @property
    def output_offsets(self) -> list[timedelta] | None:
        """Output times relative to the last input, as timedeltas."""
        return self._task.output_offsets

    @property
    def rollout_shift(self) -> timedelta | None:
        """Time by which each rollout step advances, as a timedelta."""
        return self._task.rollout_shift

    @property
    def advance_map(self) -> dict[str, list[tuple[int, int]]] | None:
        """Index mapping that advances input and output into the next input."""
        return self._task.advance_map

    @property
    def input_explicit_times(self) -> list[int] | None:
        """Times of the input steps in timesteps of the data: the config's explicit times, else the indices."""
        return self._task.input_explicit_times

    @property
    def target_explicit_times(self) -> list[int] | None:
        """Times of the target steps in timesteps of the data: the config's explicit times, else the indices."""
        return self._task.target_explicit_times

    def __repr__(self) -> str:
        return f"TaskView(task_name={self.task_name!r})"


# ---------------------------------------------------------------------------
# Lineage component (checkpoint level)
# ---------------------------------------------------------------------------


class LineageView:
    """Read-only view of the lineage component of a checkpoint.

    Where the checkpoint comes from.  A value that the checkpoint's lineage
    version does not record raises
    :class:`~anemoi.metadata.exceptions.VersionError` rather than reading as
    ``None``; see :mod:`anemoi.metadata.components.lineage` for the versions.

    Parameters
    ----------
    lineage : LineageAdapter
        The lineage component adapter of the checkpoint.
    """

    def __init__(self, lineage: "LineageAdapter") -> None:
        self._lineage = lineage

    @property
    def uuid(self) -> str | None:
        """The ``uuid`` of the checkpoint."""
        return self._lineage.uuid

    @property
    def parent_uuid(self) -> str | None:
        """The ``uuid`` of the checkpoint this one was loaded from; ``None`` when it has no parent."""
        return self._lineage.parent_uuid

    @property
    def entries(self) -> list[dict[str, Any]]:
        """The ``checkpoint_lineage`` entries, oldest first (a copy); empty for a fresh run."""
        return self._lineage.entries

    def __repr__(self) -> str:
        return "LineageView()"


# ---------------------------------------------------------------------------
# Raw sections
# ---------------------------------------------------------------------------


class SectionsView:
    """Read-only view of the raw, unversioned sections of the metadata.

    Values are returned as stored (copies, with the fallbacks of the sections
    reader); an absent dump reads as an empty dict, an absent scalar as ``None``.

    Parameters
    ----------
    reader : SectionsReader
        The sections reader of the checkpoint adapter.
    """

    def __init__(self, reader: "SectionsReader") -> None:
        self._reader = reader

    def _dict(self, name: str) -> dict[str, Any]:
        value = self._reader.section(name)
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def get(self, name: str, default: Any = None) -> Any:
        """Return the top-level key *name* as stored (a copy), or *default*."""
        return copy.deepcopy(self._reader.section(name, default))

    @property
    def config(self) -> dict[str, Any]:
        """The training ``config`` dump."""
        return copy.deepcopy(self._reader.config)

    @property
    def training(self) -> dict[str, Any]:
        """The ``training`` dump written by the checkpoint callback."""
        return self._dict("training")

    @property
    def tracker(self) -> dict[str, Any]:
        """The ``tracker`` dump written by the checkpoint callback."""
        return self._dict("tracker")

    @property
    def environment(self) -> dict[str, Any]:
        """The ``environment`` dump."""
        return self._dict("environment")

    @property
    def provenance(self) -> dict[str, Any]:
        """The provenance (``provenance_training``, else ``provenance``)."""
        return copy.deepcopy(self._reader.provenance)

    @property
    def timestamp(self) -> datetime | None:
        """When the metadata was written (``timestamp``, else ``created_at``)."""
        return self._reader.timestamp

    @property
    def run_id(self) -> str | None:
        """The ``run_id``."""
        return self._reader.run_id

    @property
    def seeds(self) -> dict[str, Any]:
        """The recorded ``seed`` and ``base_seed``."""
        return copy.deepcopy(self._reader.seeds)

    @property
    def precision(self) -> str | None:
        """The training precision (e.g. ``"16-mixed"``)."""
        return self._reader.precision

    def __repr__(self) -> str:
        return "SectionsView()"


# ---------------------------------------------------------------------------
# Per-dataset view
# ---------------------------------------------------------------------------


class DatasetView:
    """Read-only view of one dataset of a :class:`Metadata` instance.

    There is no task here: the task is per checkpoint (:attr:`Metadata.task`).

    Parameters
    ----------
    raw : CheckpointAdapter
        The underlying checkpoint adapter.
    name : str
        The dataset name this view is bound to.
    arrays : DatasetArrays | None, optional
        The supporting arrays of the dataset; none when omitted.

    Examples
    --------
    >>> view = Metadata.from_checkpoint("model.ckpt").dataset("era5")
    >>> view.data.grid_size()
    40320
    >>> view.data.variable_indices()
    {'2t': 0, 'msl': 1, ...}
    >>> view.arrays.latitudes.shape
    (40320,)
    """

    def __init__(self, raw: "CheckpointAdapter", name: str, arrays: DatasetArrays | None = None) -> None:
        self._raw = raw
        self._name = name
        self._arrays = arrays if arrays is not None else DatasetArrays()
        self._data = DatasetData(raw.data(name))

    @property
    def name(self) -> str:
        """The dataset name this view is bound to."""
        return self._name

    @property
    def data(self) -> DatasetData:
        """The data component bound to this dataset."""
        return self._data

    @property
    def arrays(self) -> DatasetArrays:
        """The supporting arrays of this dataset (each ``None`` when absent, e.g. tabular)."""
        return self._arrays

    @property
    def number_of_grid_points(self) -> int | None:
        """Number of grid points of the dataset.

        The length of ``arrays.grid_indices`` when the dataset has it (a
        limited-area model keeps a subset of the source grid), else the length
        of ``arrays.latitudes``, else the grid size recorded in the metadata
        (``data.grid_size()``); ``None`` when none of them is known (a tabular
        dataset, for example).
        """
        grid_indices = self._arrays.grid_indices
        if grid_indices is not None:
            return len(grid_indices)
        latitudes = self._arrays.latitudes
        if latitudes is not None:
            return len(latitudes)
        return self._data.grid_size()

    def __repr__(self) -> str:
        return f"DatasetView(name={self._name!r})"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Return *base* with *patch* merged in: dicts merge recursively, any other value replaces."""
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class Metadata:
    """User-facing metadata interface.

    Parameters
    ----------
    raw : CheckpointAdapter
        The underlying checkpoint adapter.
    supporting_arrays : Mapping[str, Any] | None, optional
        The supporting arrays of the checkpoint, as returned by
        ``load_metadata(path, supporting_arrays=True)`` (keyed by dataset name,
        or flat for single-dataset checkpoints).  They are copied.

    Examples
    --------
    >>> metadata = Metadata.from_checkpoint("model.ckpt")
    >>> metadata.version
    '2.0'
    >>> metadata.dataset_names
    ['era5']
    >>> metadata.task.timestep()
    '6h'
    >>> metadata.dataset("era5").data.grid_size()
    40320
    """

    def __init__(self, raw: "CheckpointAdapter", supporting_arrays: Mapping[str, Any] | None = None) -> None:
        self._raw = raw
        self._arrays = split_by_dataset(supporting_arrays, raw.dataset_names)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "Metadata":
        """Load metadata and supporting arrays from a checkpoint file.

        Parameters
        ----------
        path : str | Path
            Path to the checkpoint file.

        Raises
        ------
        CheckpointError
            If the file is invalid or metadata is missing.
        """
        from .checkpoint import load_metadata

        raw, arrays = load_metadata(path, supporting_arrays=True)
        return cls(raw, arrays)

    @classmethod
    def from_dict(cls, data: dict[str, Any], supporting_arrays: Mapping[str, Any] | None = None) -> "Metadata":
        """Build metadata from a raw dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Raw metadata dictionary (copied).
        supporting_arrays : Mapping[str, Any] | None, optional
            Supporting arrays, keyed by dataset name or flat for a
            single-dataset checkpoint (copied).
        """
        from .base import CheckpointAdapter

        return cls(CheckpointAdapter(data), supporting_arrays)

    def with_overrides(
        self,
        patch: Mapping[str, Any],
        supporting_arrays: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> "Metadata":
        """Return a new instance whose raw metadata is this one's with *patch* deep-merged in.

        Dicts merge recursively; any other value (lists included) replaces
        the stored one.  The supporting arrays are kept, except those named in
        *supporting_arrays*, which are replaced or added.  This instance is not
        changed.

        Parameters
        ----------
        patch : Mapping[str, Any]
            Partial raw metadata, e.g. ``{"config": {"data": {"frequency": "3h"}}}``.
        supporting_arrays : Mapping[str, Mapping[str, numpy.ndarray]] | None, optional
            Arrays to replace or add, ``{dataset_name: {array_name: array}}``
            (copied), for example the grid an external graph builder produced.
            The arrays of other datasets, and the other arrays of these
            datasets, are kept.

        Raises
        ------
        KeyError
            If *supporting_arrays* names a dataset the new metadata does not have.
        """
        merged = _deep_merge(self._raw.to_dict(), patch)
        arrays = self._arrays
        if supporting_arrays:
            arrays = merge_supporting_arrays(self._arrays, supporting_arrays)
        result = type(self).from_dict(merged, arrays)
        unknown = [name for name in (supporting_arrays or {}) if name not in result.dataset_names]
        if unknown:
            msg = f"Unknown dataset(s) {unknown!r} in supporting_arrays. Available: {result.dataset_names!r}"
            raise KeyError(msg)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the raw metadata dictionary, unchanged (unknown keys included)."""
        return self._raw.to_dict()

    # ------------------------------------------------------------------
    # Checkpoint level
    # ------------------------------------------------------------------

    @property
    def version(self) -> str | None:
        """The top-level metadata ``version`` string, or ``None`` if the checkpoint has none."""
        return self._raw.version

    @property
    def component_versions(self) -> dict[str, str]:
        """Version of each structured component (``data``, ``task``, ``lineage``): stamped, else inferred."""
        return self._raw.component_versions

    @property
    def dataset_names(self) -> list[str]:
        """Ordered dataset names of the checkpoint."""
        return self._raw.dataset_names

    @property
    def task(self) -> TaskView:
        """The task of the checkpoint (name, timing, offsets, ...)."""
        return TaskView(self._raw.task)

    @property
    def lineage(self) -> LineageView:
        """The lineage of the checkpoint (``uuid``, ``parent_uuid``, ``entries``)."""
        return LineageView(self._raw.lineage)

    @property
    def sections(self) -> SectionsView:
        """The raw sections: config, provenance, timestamp, seeds, ..."""
        return SectionsView(self._raw.sections)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def dataset(self, name: str | None = None) -> DatasetView:
        """Return the view of one dataset.

        Parameters
        ----------
        name : str | None, optional
            Dataset name.  It may be omitted only when the checkpoint has
            exactly one dataset.

        Raises
        ------
        DatasetSelectionError
            If *name* is omitted and the checkpoint has no dataset or several.
        KeyError
            If *name* is not a dataset of the checkpoint.
        """
        names = self.dataset_names
        if name is None:
            if len(names) != 1:
                raise DatasetSelectionError(
                    f"A dataset name is required: the checkpoint has {len(names)} datasets ({names!r})"
                    if names
                    else "The checkpoint has no datasets",
                )
            name = names[0]
        elif name not in names:
            msg = f"Dataset {name!r} not found. Available datasets: {names!r}"
            raise KeyError(msg)
        return DatasetView(self._raw, name, DatasetArrays(self._arrays.get(name)))

    @property
    def datasets(self) -> dict[str, DatasetView]:
        """All datasets by name, in :attr:`dataset_names` order."""
        return {name: self.dataset(name) for name in self.dataset_names}

    # ------------------------------------------------------------------
    # Single-dataset shortcuts
    # ------------------------------------------------------------------

    @property
    def variables(self) -> list[str]:
        """Variable names of the only dataset (``DatasetSelectionError`` if there are several)."""
        return self.dataset().data.variables()

    @property
    def grid_size(self) -> int | None:
        """Grid size of the only dataset (``DatasetSelectionError`` if there are several)."""
        return self.dataset().data.grid_size()

    @property
    def latitudes(self) -> np.ndarray | None:
        """Latitudes of the only dataset (``DatasetSelectionError`` if there are several)."""
        return self.dataset().arrays.latitudes

    @property
    def longitudes(self) -> np.ndarray | None:
        """Longitudes of the only dataset (``DatasetSelectionError`` if there are several)."""
        return self.dataset().arrays.longitudes

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def validate_environment(self) -> list[str]:
        """Compare the recorded Python version with the running one.

        Returns
        -------
        list[str]
            Warning messages; empty when they match or no environment is recorded.
        """
        warnings: list[str] = []
        checkpoint_python = self.sections.environment.get("python_version")
        if checkpoint_python is not None:
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if checkpoint_python != current_python:
                warnings.append(f"Python version mismatch: checkpoint={checkpoint_python}, current={current_python}")
        return warnings

    # ------------------------------------------------------------------
    # Raw access and dunders
    # ------------------------------------------------------------------

    def __getitem__(self, section: str) -> Any:
        """Return a copy of a top-level key of the raw metadata (e.g. ``"config"``).

        Only stored keys are reachable, never methods or attributes.

        Raises
        ------
        KeyError
            If *section* is not a top-level key of the raw metadata.
        """
        value = self._raw.sections.section(section, _MISSING)
        if value is _MISSING:
            raise KeyError(section)
        return copy.deepcopy(value)

    def __repr__(self) -> str:
        return f"Metadata(version={self.version!r}, datasets={self.dataset_names!r})"
