# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Data component adapters.

The data component covers what the datamodule and the dataset readers write:
``metadata_inference.dataset_names`` and, per dataset, ``data_indices`` and
``variable_types`` (plus, from format 3.0, ``data_type`` and ``grid_size``).
Adapters may also read the raw ``dataset`` and ``config`` dumps for values the
component does not own (data request, sources, frequency, variables metadata).

One checkpoint can mix gridded and tabular datasets, so a data adapter is built
for **one dataset**: its methods take no dataset name.  It varies on two axes,
the layout version and the dataset's ``data_type``, so each adapter is built
from a *version reader* (where and how the values are stored) and a *data
behaviour* (what the kind of dataset has) on top of :class:`DataAdapter`;
:func:`register_data_version` combines a reader with every behaviour and
registers the result under ``("data", version, data_type)``.  A change for a
new component version is one reader that reaches every data type.

Component versions:

* ``0.1``: legacy checkpoints without ``metadata_inference`` (reader in
  :mod:`anemoi.metadata.components.legacy`).  Always gridded.
* ``0.2``: layout of metadata format ``2.0`` (multi-dataset ``metadata_inference``).
  Every dataset is gridded.
* ``1.0``: layout of metadata format ``3.0``; adds ``data_type`` (``"gridded"`` or
  ``"tabular"``) and ``grid_size`` (``null`` for tabular datasets).

Each reader subclasses the nearest older one and overrides only what differs.
The behaviour of a dataset is chosen from the ``data_type`` its reader reports.

"Not applicable" and "not recorded"
-----------------------------------

A tabular dataset has no grid.  Values that only make sense on a grid
(:meth:`~DataAdapter.grid_size`, :meth:`~DataAdapter.data_request`,
:meth:`~DataAdapter.grid`, :meth:`~DataAdapter.area`) are ``None`` for a tabular
dataset: *not applicable*.  For a gridded dataset the same values are never
"not applicable":

* ``data_request`` is an empty dict when the checkpoint does not record one;
* ``grid_size``, ``grid`` and ``area`` are ``None`` when they are not recorded
  and cannot be derived.

So an empty dict or list always means *not recorded*, and ``None`` from a
grid-related value means *not applicable* if :meth:`~DataAdapter.is_tabular` is true
and *not recorded* otherwise.

Every list, dict and nested structure returned by an adapter is a copy: changing
it never changes the adapter or the raw metadata.
"""

import copy
import logging
import warnings
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any
from typing import cast

from anemoi.metadata.registry import ComponentRegistry

from .base import ComponentAdapter

LOG = logging.getLogger(__name__)

_VARIABLE_TYPE_KEYS = {"forcing", "prognostic", "diagnostic", "target"}

#: Categories added by :meth:`DataAdapter.variable_categories` on top of the variable types.
_DERIVED_CATEGORIES = {"computed", "accumulation", "constant"}

#: Every category name :meth:`DataAdapter.select_variables` accepts.
VARIABLE_CATEGORIES = _VARIABLE_TYPE_KEYS | _DERIVED_CATEGORIES

GRIDDED = "gridded"
TABULAR = "tabular"


def _as_dict(value: object) -> dict[str, Any]:
    """Return *value* if it is a dict, otherwise an empty dict."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _as_int(value: object) -> int | None:
    """Return *value* if it is an int (not a bool), otherwise ``None``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _find_all(node: object, key: str) -> Iterator[object]:
    """Yield the value of every *key* found in nested dicts and lists, depth first."""
    if isinstance(node, dict):
        for name, value in cast("dict[str, object]", node).items():
            if name == key:
                yield value
            yield from _find_all(value, key)
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            yield from _find_all(item, key)


class DataAdapter(ComponentAdapter):
    """Machinery shared by every data adapter, whatever its version or data type.

    The adapter reads **one dataset** of the checkpoint.  It varies on two
    axes, each a mixin placed in front of this class:

    * a *version reader* (:class:`DataReaderV0_2` and its subclasses) knows
      where and how the metadata is stored: :meth:`recorded_data_type`,
      :meth:`variable_indices`, :meth:`output_variable_indices`,
      :meth:`variable_types`, :meth:`variables_metadata`,
      :meth:`_dataset_section` and :meth:`_recorded_grid_size`;
    * a *data behaviour* (:class:`GriddedBehaviour`, :class:`TabularBehaviour`)
      knows what the kind of dataset has: :meth:`data_type`,
      :meth:`grid_size` and :meth:`data_request`.

    :func:`register_data_version` combines one version reader with every
    behaviour, so a change made in a version reader reaches every data type.
    This class is not usable on its own: the methods named above raise
    ``NotImplementedError`` until a reader and a behaviour provide them.

    Parameters
    ----------
    raw : Mapping[str, Any]
        The raw metadata dictionary of the checkpoint.
    dataset_name : str
        The dataset to read.

    Raises
    ------
    KeyError
        If *dataset_name* is not one of the dataset names.
    """

    def __init__(self, raw: Mapping[str, Any], dataset_name: str) -> None:
        super().__init__(raw)
        self._name: str = self.resolve_dataset(dataset_name)

    @property
    def dataset_name(self) -> str:
        """The dataset this adapter reads."""
        return self._name

    def _block(self) -> dict[str, Any]:
        """Return the ``metadata_inference.<dataset>`` block of this dataset, or an empty dict."""
        return self._dataset_block(self._name)

    # ------------------------------------------------------------------
    # Hooks implemented by the version reader
    # ------------------------------------------------------------------

    def recorded_data_type(self) -> str:
        """Return the dataset kind as the layout records it (``"gridded"`` when it records none).

        The behaviour of the adapter is chosen from this value.
        """
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def variable_indices(self) -> dict[str, int]:
        """Return the input variable name to tensor index mapping."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def output_variable_indices(self) -> dict[str, int]:
        """Return the output variable name to tensor index mapping.

        Empty for an input-only dataset.
        """
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def variable_types(self) -> dict[str, list[str]]:
        """Return variable names by role (forcing, prognostic, diagnostic, target)."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata, with ``constant_fields`` marked ``constant_in_time``.

        Empty when the checkpoint records none (always for tabular datasets today).
        """
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _dataset_section(self) -> dict[str, Any]:
        """Return the raw ``dataset`` dump of this dataset (not a copy: for internal reading)."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _recorded_grid_size(self) -> int | None:
        """Return the grid size the layout records or lets one derive; ``None`` if neither."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Hooks implemented by the behaviour
    # ------------------------------------------------------------------

    def data_type(self) -> str:
        """Return the dataset kind: ``"gridded"`` or ``"tabular"``."""
        msg = f"{type(self).__name__} has no data behaviour"
        raise NotImplementedError(msg)

    def grid_size(self) -> int | None:
        """Return the number of grid points; ``None`` if tabular (not applicable) or not recorded."""
        msg = f"{type(self).__name__} has no data behaviour"
        raise NotImplementedError(msg)

    def data_request(self) -> dict[str, Any] | None:
        """Return the data request (grid, area, parameters, ...).

        ``None`` for a tabular dataset (not applicable); an empty dict when a
        gridded dataset records none.
        """
        msg = f"{type(self).__name__} has no data behaviour"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Dataset kind
    # ------------------------------------------------------------------

    def is_gridded(self) -> bool:
        """Return whether the dataset is gridded."""
        return self.data_type() == GRIDDED

    def is_tabular(self) -> bool:
        """Return whether the dataset is tabular (no grid, e.g. observations)."""
        return self.data_type() == TABULAR

    def is_input_only(self) -> bool:
        """Return whether the dataset has input variables but no output variables."""
        return bool(self.variable_indices()) and not self.output_variable_indices()

    # ------------------------------------------------------------------
    # Variables and indices
    # ------------------------------------------------------------------

    def input_index_to_variable(self) -> dict[int, str]:
        """Return the input tensor index to variable name mapping (inverse of :meth:`variable_indices`)."""
        return {index: name for name, index in self.variable_indices().items()}

    def output_index_to_variable(self) -> dict[int, str]:
        """Return the output tensor index to variable name mapping (inverse of :meth:`output_variable_indices`)."""
        return {index: name for name, index in self.output_variable_indices().items()}

    def variables(self) -> list[str]:
        """Return every variable name of the dataset, inputs then output-only variables.

        Input variables come first in input tensor index order; variables that
        are only outputs follow in output tensor index order.
        """
        inputs = self.variable_indices()
        outputs = self.output_variable_indices()
        result = [name for name, _ in sorted(inputs.items(), key=lambda item: item[1])]
        result.extend(name for name, _ in sorted(outputs.items(), key=lambda item: item[1]) if name not in inputs)
        return result

    def prognostic_input_mask(self) -> list[int]:
        """Return the sorted input tensor indices of the prognostic variables.

        Empty when the dataset has no output variables (input-only): nothing is
        fed back, whatever ``variable_types`` says.
        """
        if not self.output_variable_indices():
            return []
        prognostic = set(self.variable_types()["prognostic"])
        return sorted(index for name, index in self.variable_indices().items() if name in prognostic)

    def prognostic_output_mask(self) -> list[int]:
        """Return the sorted output tensor indices of the prognostic variables (empty if input-only)."""
        prognostic = set(self.variable_types()["prognostic"])
        return sorted(index for name, index in self.output_variable_indices().items() if name in prognostic)

    # ------------------------------------------------------------------
    # Grid (from the data request)
    # ------------------------------------------------------------------

    def grid(self) -> Any:
        """Return the grid of the data request (e.g. ``"O96"``); ``None`` if tabular or not recorded."""
        request = self.data_request()
        return None if request is None else copy.deepcopy(request.get("grid"))

    def area(self) -> Any:
        """Return the area of the data request; ``None`` if tabular or not recorded."""
        request = self.data_request()
        return None if request is None else copy.deepcopy(request.get("area"))

    # ------------------------------------------------------------------
    # dataset (raw dump)
    # ------------------------------------------------------------------

    def accumulations(self) -> list[str]:
        """Return the variables whose ``process`` is ``accumulated`` or ``accumulation``."""
        metadata = self.variables_metadata()
        return [name for name, meta in metadata.items() if meta.get("process") in ("accumulated", "accumulation")]

    def computed_forcings(self) -> list[str]:
        """Return the variables flagged ``computed_forcing``."""
        metadata = self.variables_metadata()
        return [name for name, meta in metadata.items() if meta.get("computed_forcing", False)]

    def sources(self) -> list[dict[str, Any]]:
        """Return the source dataset configurations, or an empty list."""
        return copy.deepcopy(self._dataset_section().get("sources") or [])

    def open_dataset_args(self) -> dict[str, Any]:
        """Return the arguments used to open the training dataset, or an empty dict."""
        return copy.deepcopy(self._dataset_section().get("arguments") or {})

    def data_frequency(self) -> str | None:
        """Return the data frequency.

        ``config.task.output_timestep`` takes precedence, then
        ``dataset.frequency``, then ``config.data.frequency``; a ``null`` value
        is skipped.  ``None`` if none is recorded.
        """
        config = self._section("config")
        output_timestep = _as_dict(config.get("task")).get("output_timestep")
        if output_timestep is not None:
            return output_timestep
        frequency = self._dataset_section().get("frequency")
        if frequency is not None:
            return frequency
        return _as_dict(config.get("data")).get("frequency")

    # ------------------------------------------------------------------
    # Categories, selection and typed variables
    # ------------------------------------------------------------------

    def variable_categories(self, *, per_variable: bool = False) -> dict[str, list[str]]:
        """Categorise the variables by their role in the model.

        On top of the variable types (``forcing``, ``prognostic``, ``diagnostic``,
        ``target``) three derived categories are merged in:

        * ``computed``: variables flagged as computed forcings;
        * ``accumulation``: variables whose ``process`` is accumulated;
        * ``constant``: variables whose ``constant_in_time`` flag is true.

        Parameters
        ----------
        per_variable : bool
            If false (default) return ``{category: [variable, ...]}``; the four
            variable-type keys are always present (possibly empty), the derived
            ones only when they hold a variable.  If true return
            ``{variable: [category, ...]}`` with sorted categories.

        Returns
        -------
        dict[str, list[str]]
            The category mapping.
        """
        base = self.variable_types()
        metadata = self.variables_metadata()
        derived = {
            "computed": [name for name, meta in metadata.items() if meta.get("computed_forcing", False)],
            "accumulation": [
                name for name, meta in metadata.items() if meta.get("process") in ("accumulated", "accumulation")
            ],
            "constant": [name for name, meta in metadata.items() if meta.get("constant_in_time")],
        }

        if not per_variable:
            result = dict(base)
            for category, names in derived.items():
                names = list(dict.fromkeys(names))
                if names:
                    result[category] = names
            return result

        inverted: defaultdict[str, set[str]] = defaultdict(set)
        for category, names in {**base, **derived}.items():
            for name in names:
                inverted[name].add(category)
        return {name: sorted(categories) for name, categories in inverted.items()}

    def select_variables(self, *, include: list[str] | None = None, exclude: list[str] | None = None) -> list[str]:
        """Select variables by category expression.

        Each entry of *include* and *exclude* is a category name (one of
        :data:`VARIABLE_CATEGORIES`) or several joined with ``+`` (for example
        ``"prognostic+constant"``), which selects the variables in all of them.
        A variable is kept if it matches any *include* entry (all variables when
        *include* is ``None``) and no *exclude* entry.  There is no substring
        matching: an entry that is not a category raises.

        Parameters
        ----------
        include : list[str] | None
            Expressions to keep.
        exclude : list[str] | None
            Expressions to remove.

        Returns
        -------
        list[str]
            The selected variables, in :meth:`variables` order.

        Raises
        ------
        ValueError
            If an entry names an unknown category.
        """
        include_sets = self._parse_expressions(include)
        exclude_sets = self._parse_expressions(exclude)

        per_variable = self.variable_categories(per_variable=True)

        def matches(variable: str, expressions: list[frozenset[str]]) -> bool:
            categories = set(per_variable.get(variable, ()))
            return any(expression <= categories for expression in expressions)

        result = self.variables()
        if include_sets is not None:
            result = [name for name in result if matches(name, include_sets)]
        if exclude_sets is not None:
            result = [name for name in result if not matches(name, exclude_sets)]
        return result

    @staticmethod
    def _parse_expressions(expressions: list[str] | None) -> list[frozenset[str]] | None:
        """Parse category expressions, raising ``ValueError`` on an unknown category."""
        if expressions is None:
            return None
        parsed = []
        for expression in expressions:
            parts = expression.split("+")
            unknown = [part for part in parts if part not in VARIABLE_CATEGORIES]
            if unknown:
                msg = (
                    f"Unknown variable category {unknown!r} in {expression!r}. "
                    f"Known categories: {list(VARIABLE_CATEGORIES)}"
                )
                raise ValueError(msg)
            parsed.append(frozenset(parts))
        return parsed

    def typed_variables(self) -> dict[str, Any]:
        """Return an :class:`anemoi.transform.variables.Variable` for each variable.

        Built from :meth:`variables_metadata`; a variable without metadata gets
        an empty one.

        Raises
        ------
        ImportError
            If ``anemoi-transform`` is not installed.
        """
        try:
            from anemoi.transform.variables import Variable
        except ImportError as exc:
            msg = "typed_variables() requires anemoi-transform. Install it with: pip install anemoi-transform"
            raise ImportError(msg) from exc

        metadata = self.variables_metadata()
        return {name: Variable.from_dict(name, metadata.get(name, {})) for name in self.variables()}

    # ------------------------------------------------------------------
    # config (raw dump)
    # ------------------------------------------------------------------

    def dataloader_config(self, partition: str = "training") -> dict[str, Any]:
        """Return the dataloader dataset configuration of a partition.

        Reads ``config.dataloader.<partition>``, picks the per-dataset entry
        under ``datasets`` for multi-dataset checkpoints and unwraps the
        ``dataset_config`` key used by newer checkpoints.
        """
        dataloader = _as_dict(_as_dict(self._section("config").get("dataloader")).get(partition))

        entry = _as_dict(dataloader.get("datasets")).get(self._name)
        if isinstance(entry, dict):
            dataloader = _as_dict(entry)

        config_val: object = dataloader.get("dataset_config")
        if isinstance(config_val, str):
            config_val = {"dataset": config_val}
        if isinstance(config_val, dict):
            settings = dict(_as_dict(config_val))
            for key in ("start", "end"):
                if key in dataloader:
                    settings.setdefault(key, dataloader[key])
            return copy.deepcopy({k: v for k, v in settings.items() if v is not None})

        return copy.deepcopy(dataloader)


# --------------------------------------------------------------------------
# Version axis: readers
# --------------------------------------------------------------------------


class DataReaderV0_2(DataAdapter):
    """Version reader for the component version ``0.2`` (metadata format ``2.0``).

    Knows where the data values are stored: the per dataset
    ``metadata_inference.<dataset>`` block and the raw ``dataset`` dump (keyed
    by dataset name).  Every dataset of this layout is
    gridded.  A reader for another layout subclasses the nearest one and
    overrides what differs; it is then combined with every data type by
    :func:`register_data_version`.
    """

    def recorded_data_type(self) -> str:
        """Return ``"gridded"``: this layout records no data type."""
        return GRIDDED

    def variable_indices(self) -> dict[str, int]:
        """Return the input variable name to tensor index mapping."""
        indices = _as_dict(self._block().get("data_indices"))
        return dict(indices.get("input") or {})

    def output_variable_indices(self) -> dict[str, int]:
        """Return the output variable name to tensor index mapping (empty if input-only)."""
        indices = _as_dict(self._block().get("data_indices"))
        return dict(indices.get("output") or {})

    def variable_types(self) -> dict[str, list[str]]:
        """Return variable names by role (forcing, prognostic, diagnostic, target)."""
        stored = _as_dict(self._block().get("variable_types"))
        return {key: list(stored.get(key) or []) for key in _VARIABLE_TYPE_KEYS}

    def _dataset_section(self) -> dict[str, Any]:
        """Return the raw ``dataset`` dump of this dataset (not a copy: for internal reading).

        The dump is keyed by dataset name from format 2.0.  A tabular dataset
        has an empty block under its name, which is returned as it is.
        """
        section = self._section("dataset")
        nested = section.get(self._name)
        return nested if isinstance(nested, dict) else section

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        """Return per-variable metadata, with ``constant_fields`` marked ``constant_in_time``."""
        section = self._dataset_section()
        result = copy.deepcopy(_as_dict(section.get("variables_metadata")))
        for name in section.get("constant_fields") or []:
            if name in result:
                result[name]["constant_in_time"] = True
        return result

    def _recorded_grid_size(self) -> int | None:
        """Return the first available of ``shapes.grid`` and the last axis of ``dataset.shape``."""
        shapes = _as_dict(self._block().get("shapes"))
        dataset_shape = self._dataset_section().get("shape")
        candidates = (
            shapes.get("grid"),
            dataset_shape[-1] if isinstance(dataset_shape, list) and dataset_shape else None,
        )
        for candidate in candidates:
            size = _as_int(candidate)
            if size is not None:
                return size
        return None


class DataReaderV1_0(DataReaderV0_2):
    """Version reader for the component version ``1.0`` (metadata format ``3.0``): ``data_type`` and ``grid_size``."""

    def recorded_data_type(self) -> str:
        """Return the stored ``data_type``; ``"gridded"`` when the key is absent."""
        return self._block().get("data_type") or GRIDDED

    def _recorded_grid_size(self) -> int | None:
        """Return the stored ``grid_size``, else as :meth:`DataReaderV0_2._recorded_grid_size`."""
        stored = _as_int(self._block().get("grid_size"))
        return stored if stored is not None else super()._recorded_grid_size()


# --------------------------------------------------------------------------
# Data type axis: behaviours
# --------------------------------------------------------------------------


class GriddedBehaviour(DataAdapter):
    """Gridded dataset: it has a grid size and a data request."""

    def data_type(self) -> str:
        """Return ``"gridded"``."""
        return GRIDDED

    def grid_size(self) -> int | None:
        """Return the number of grid points the layout records or derives; ``None`` if not recorded."""
        return self._recorded_grid_size()

    def data_request(self) -> dict[str, Any]:
        """Return the data request (grid, area, parameters, ...); an empty dict when none is recorded.

        The request is searched for in the raw ``dataset`` dump of the dataset,
        because it sits deep in the nested dataset description (for example
        ``specific.forward.forward.attrs.data_request``, sometimes inside
        ``sources`` or ``datasets`` lists).  A top-level ``data_request`` wins.
        If several are found, the first with a ``grid`` is returned, and a warning
        is logged when their ``grid`` or ``area`` differ.  Logic ported from
        anemoi-inference ``legacy.py`` (``_legacy_data_request``).
        """
        section = self._dataset_section()
        direct = _as_dict(section.get("data_request"))
        if direct:
            return copy.deepcopy(direct)

        found = [request for request in map(_as_dict, _find_all(section, "data_request")) if request]
        if not found:
            return {}
        if len(found) > 1:
            for key in ("grid", "area"):
                values = {str(request.get(key)) for request in found}
                if len(values) > 1:
                    LOG.warning("data_request %s is ambiguous: %s", key, sorted(values))
            with_grid = [request for request in found if request.get("grid")]
            found = with_grid or found
        return copy.deepcopy(found[0])


class TabularBehaviour(DataAdapter):
    """Tabular dataset (e.g. observations): no grid, so every grid related value is *not applicable* (``None``)."""

    def data_type(self) -> str:
        """Return ``"tabular"``."""
        return TABULAR

    def grid_size(self) -> int | None:
        """Return ``None``: a tabular dataset has no grid."""
        return None

    def data_request(self) -> dict[str, Any] | None:
        """Return ``None``: a tabular dataset has no data request."""
        return None


#: The data behaviours by ``data_type``.
DATA_BEHAVIOURS: dict[str, type[DataAdapter]] = {
    GRIDDED: GriddedBehaviour,
    TABULAR: TabularBehaviour,
}


# --------------------------------------------------------------------------
# Both axes together
# --------------------------------------------------------------------------


def register_data_version(
    version: str,
    reader: type[DataAdapter],
    *,
    names: Iterable[str] | None = None,
) -> dict[str, type[DataAdapter]]:
    """Register the data adapters of one component version: *reader* combined with each data type.

    For every data type a class ``<DataType>V<version>`` is built from the
    behaviour, *reader* and :class:`DataAdapter`, and registered under
    ``("data", version, data_type)``.  A change made in *reader* therefore
    reaches every data type.

    Parameters
    ----------
    version : str
        Component version the reader reads (e.g. ``"1.1"``).
    reader : type[DataAdapter]
        Version reader, usually a subclass of the nearest older reader.
    names : Iterable[str] | None, optional
        Data types to register; every type of :data:`DATA_BEHAVIOURS` when omitted.

    Returns
    -------
    dict[str, type[DataAdapter]]
        The registered classes by data type.

    Examples
    --------
    >>> class DataReaderV1_1(DataReaderV1_0):
    ...     def _recorded_grid_size(self): ...
    >>> classes = register_data_version("1.1", DataReaderV1_1)
    >>> classes["tabular"].version
    '1.1'
    """
    registered: dict[str, type[DataAdapter]] = {}
    for name in DATA_BEHAVIOURS if names is None else names:
        behaviour = DATA_BEHAVIOURS[name]
        adapter_cls = cast(
            "type[DataAdapter]",
            type(
                f"{name.title()}V{version.replace('.', '_')}",
                (behaviour, reader, DataAdapter),
                {
                    "__module__": __name__,
                    "__doc__": f"Data adapter for {name} datasets, version {version}.",
                },
            ),
        )
        registered[name] = ComponentRegistry.register("data", version, name=name)(adapter_cls)
    return registered


def build_data_adapters(
    raw: Mapping[str, object],
    version: str,
    dataset_names: Iterable[str],
) -> dict[str, DataAdapter]:
    """Build one data adapter per dataset: the version reader of *version* and the dataset's behaviour.

    Every behaviour of a version shares its reader, so any of them (here the
    gridded one) can tell which ``data_type`` a dataset records; the adapter of
    that type is then built for the dataset.  A recorded type without
    behaviour reads as gridded, with a warning.

    Parameters
    ----------
    raw : Mapping[str, Any]
        The raw metadata dictionary of the checkpoint.
    version : str
        The data component version (stamped or inferred).
    dataset_names : Iterable[str]
        The datasets to build an adapter for.

    Returns
    -------
    dict[str, DataAdapter]
        The adapter of each dataset, in the order of *dataset_names*.
    """
    probe_cls = cast("type[DataAdapter]", ComponentRegistry.resolve("data", version, name=GRIDDED))
    by_type: dict[str, type[DataAdapter]] = {GRIDDED: probe_cls}
    adapters: dict[str, DataAdapter] = {}
    for name in dataset_names:
        data_type = probe_cls(raw, name).recorded_data_type()
        if data_type not in by_type:
            if data_type in ComponentRegistry.names("data"):
                by_type[data_type] = cast(
                    "type[DataAdapter]",
                    ComponentRegistry.resolve("data", version, name=data_type),
                )
            else:
                warnings.warn(
                    f"Dataset {name!r} has unknown data_type {data_type!r}; reading it as {GRIDDED}.",
                    UserWarning,
                    stacklevel=3,
                )
                data_type = GRIDDED
        adapters[name] = by_type[data_type](raw, name)
    return adapters


# Component version 0.2 (format 2.0) and 1.0 (format 3.0).
_V0_2 = register_data_version("0.2", DataReaderV0_2)
_V1_0 = register_data_version("1.0", DataReaderV1_0)

GriddedV0_2 = _V0_2[GRIDDED]
TabularV0_2 = _V0_2[TABULAR]

GriddedV1_0 = _V1_0[GRIDDED]
TabularV1_0 = _V1_0[TABULAR]
