# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Strict validation of the metadata this package writes (format ``3.1``).

Reading is lenient (see :mod:`anemoi.metadata.components`); writing is not.
:func:`validate_for_write` checks that a metadata dictionary is complete and
well formed before it is stored in a checkpoint, and stamps it with
:data:`CURRENT_VERSION`.

Layout
------

The format is the richer-batch ``3.0`` layout plus a top-level
``component_versions`` map that stamps the version of each structured
component::

    {"version": "3.1", "component_versions": {"data": "1.0", "task": "1.0", "lineage": "1.0"}, ...}

Every structured component has one strict schema per component version,
registered under ``(component, version)`` with :func:`register_write_schema`.
A future version of one component adds a schema for that version only; the
other components and the older schemas are untouched.

* ``data`` ``1.0``: ``metadata_inference.dataset_names`` and, per dataset,
  ``data_type`` (``"gridded"`` or ``"tabular"``), ``grid_size`` (a positive int
  for gridded, ``null`` for tabular), ``data_indices`` and ``variable_types``.
* ``task`` ``1.0``: ``metadata_inference.task`` and, per dataset, the
  ``timesteps`` block; the offset forecaster also stores offsets, rollout
  shift and advance map.  The task is checkpoint level, so every dataset
  must carry the same ``timesteps``.
* ``lineage`` ``1.0``: the top-level ``uuid`` (a non-empty string),
  ``checkpoint_lineage`` (a list, empty for a fresh run, of
  ``{"loading_strategy", "parent_uuid"}`` entries) and, when present,
  ``parent_uuid`` (it must be the ``parent_uuid`` of the last entry; absent or ``None`` only
  when there is no entry or the last entry's parent is ``None``).

The raw dumps (``config``, ``dataset``, ``provenance_training``, ``training``,
...) carry no version.  Only the presence of ``config`` and of ``dataset`` with
one entry per dataset is checked, because the readers need them; their inner
structure is never inspected.

Errors are collected and raised together as :class:`WriteValidationError`.
"""

import copy
from collections.abc import Callable
from collections.abc import Mapping
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StrictInt
from pydantic import StrictStr
from pydantic import ValidationError
from pydantic import model_validator

from anemoi.utils.dates import as_timedelta

from .exceptions import MetadataError
from .formats import COMPONENTS
from .formats import DATA
from .formats import LINEAGE
from .formats import TASK

#: The ``version`` written by this package.
CURRENT_VERSION = "3.1"

#: The component versions written by this package, stamped in ``component_versions``.
COMPONENT_VERSIONS: dict[str, str] = {DATA: "1.0", TASK: "1.0", LINEAGE: "1.0"}

#: Values of ``version`` that are accepted and relabelled :data:`CURRENT_VERSION`.
_RELABELLED_VERSIONS = (None, "3.0")

#: Task names that store offsets, rollout shift and advance map in ``timesteps``.
_OFFSET_TASKS = ("offset-forecaster",)

_VARIABLE_TYPE_KEYS = ("forcing", "target", "prognostic", "diagnostic")


class WriteValidationError(MetadataError):
    """Raised when metadata cannot be written in the current format.

    Attributes
    ----------
    errors : list[str]
        Every problem found, one readable line each.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        listing = "\n".join(f"  - {error}" for error in self.errors)
        super().__init__(f"Metadata is not valid for writing as version {CURRENT_VERSION}:\n{listing}")


# ---------------------------------------------------------------------------
# Schema registry, keyed by (component, version)
# ---------------------------------------------------------------------------

#: A schema checks one component of a metadata dict and appends problems to the list.
_Schema = Callable[[Mapping[str, Any], list[str]], None]

_SCHEMAS: dict[tuple[str, str], _Schema] = {}


def register_write_schema(component: str, version: str) -> Callable[[_Schema], _Schema]:
    """Return a decorator registering the strict write schema of ``(component, version)``.

    Raises
    ------
    ValueError
        If a schema is already registered under that key.
    """

    def decorator(schema: _Schema) -> _Schema:
        key = (component, version)
        if key in _SCHEMAS:
            msg = f"Write schema {component} {version} already registered"
            raise ValueError(msg)
        _SCHEMAS[key] = schema
        return schema

    return decorator


def write_schema_versions(component: str) -> list[str]:
    """Return the component versions that have a write schema, sorted."""
    return sorted(version for (name, version) in _SCHEMAS if name == component)


def _format_errors(prefix: str, error: ValidationError) -> list[str]:
    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"{prefix}{'.' + location if location else ''}: {item['msg']}")
    return lines


# ---------------------------------------------------------------------------
# data 1.0
# ---------------------------------------------------------------------------


class _Block(BaseModel):
    """Base of the schema models: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class _DataIndices(_Block):
    input: dict[StrictStr, StrictInt]
    output: dict[StrictStr, StrictInt]

    @model_validator(mode="after")
    def _check_indices(self) -> "_DataIndices":
        for side in ("input", "output"):
            values = list(getattr(self, side).values())
            if any(value < 0 for value in values):
                msg = f"{side} indices must not be negative"
                raise ValueError(msg)
            if len(set(values)) != len(values):
                msg = f"{side} indices must be unique"
                raise ValueError(msg)
        return self


class _VariableTypes(_Block):
    forcing: list[StrictStr]
    target: list[StrictStr]
    prognostic: list[StrictStr]
    diagnostic: list[StrictStr]


class _DatasetDataV1_0(_Block):
    """The part of ``metadata_inference.<dataset>`` owned by the data component.

    ``timesteps`` belongs to the task component and is only declared here so
    that the block rejects unknown keys.
    """

    data_type: Literal["gridded", "tabular"]
    grid_size: Annotated[StrictInt, Field(gt=0)] | None
    data_indices: _DataIndices
    variable_types: _VariableTypes
    timesteps: dict[str, Any]

    @model_validator(mode="after")
    def _check_consistency(self) -> "_DatasetDataV1_0":
        if self.data_type == "gridded" and self.grid_size is None:
            msg = "grid_size is required for a gridded dataset"
            raise ValueError(msg)
        if self.data_type == "tabular" and self.grid_size is not None:
            msg = "grid_size must be null for a tabular dataset"
            raise ValueError(msg)
        known = set(self.data_indices.input) | set(self.data_indices.output)
        for key in _VARIABLE_TYPE_KEYS:
            unknown = [name for name in getattr(self.variable_types, key) if name not in known]
            if unknown:
                msg = f"variable_types.{key} lists variables missing from data_indices: {unknown}"
                raise ValueError(msg)
        return self


def _inference_block(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    block = metadata.get("metadata_inference")
    return block if isinstance(block, dict) else None


def _dataset_names(metadata: Mapping[str, Any]) -> list[str] | None:
    """Return the dataset names when they are a non-empty list of unique strings, else ``None``."""
    inference = _inference_block(metadata)
    names = None if inference is None else inference.get("dataset_names")
    if not isinstance(names, list) or not names:
        return None
    if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        return None
    return names


def _check_dataset_names(metadata: Mapping[str, Any], errors: list[str]) -> list[str]:
    """Check ``metadata_inference`` and its ``dataset_names``; return the usable names (possibly empty)."""
    if _inference_block(metadata) is None:
        errors.append("metadata_inference: required and must be a dict")
        return []
    names = _dataset_names(metadata)
    if names is None:
        errors.append("metadata_inference.dataset_names: required, a non-empty list of unique strings")
        return []
    return names


@register_write_schema(DATA, "1.0")
def _validate_data_v1_0(metadata: Mapping[str, Any], errors: list[str]) -> None:
    names = _check_dataset_names(metadata, errors)
    inference = _inference_block(metadata) or {}
    for name in names:
        block = inference.get(name)
        prefix = f"data: metadata_inference.{name}"
        if not isinstance(block, dict):
            errors.append(f"{prefix}: required and must be a dict")
            continue
        try:
            _DatasetDataV1_0.model_validate(block)
        except ValidationError as exc:
            errors.extend(_format_errors(prefix, exc))


# ---------------------------------------------------------------------------
# task 1.0
# ---------------------------------------------------------------------------


def _check_timedelta(value: str) -> str:
    try:
        as_timedelta(value)
    except (ValueError, TypeError) as exc:
        msg = f"{value!r} is not a valid time step"
        raise ValueError(msg) from exc
    return value


class _TimestepsV1_0(_Block):
    """``timesteps`` of the generic task."""

    timestep: StrictStr
    relative_date_indices_training: Annotated[list[StrictInt], Field(min_length=1)]
    input_relative_date_indices: Annotated[list[StrictInt], Field(min_length=1)]
    output_relative_date_indices: Annotated[list[StrictInt], Field(min_length=1)]

    @model_validator(mode="after")
    def _check_timestep(self) -> "_TimestepsV1_0":
        _check_timedelta(self.timestep)
        return self


class _OffsetTimestepsV1_0(_TimestepsV1_0):
    """``timesteps`` of the offset forecaster: adds the stored offsets, rollout shift and advance map."""

    input_offsets: Annotated[list[StrictStr], Field(min_length=1)]
    output_offsets: Annotated[list[StrictStr], Field(min_length=1)]
    rollout_shift: StrictStr
    advance_map: dict[StrictStr, list[Annotated[list[StrictInt], Field(min_length=2, max_length=2)]]]

    @model_validator(mode="after")
    def _check_offsets(self) -> "_OffsetTimestepsV1_0":
        for value in (*self.input_offsets, *self.output_offsets, self.rollout_shift):
            _check_timedelta(value)
        return self


def _check_timing_consistency(prefix: str, timesteps: Mapping[str, Any], *, offsets: bool) -> list[str]:
    """Check that a schema-valid ``timesteps`` block is consistent with itself.

    The input and output indices must be among the training indices (which may
    repeat an index, so they are compared as a set); for the offset forecaster
    each offsets list has one offset per index.
    """
    errors = []
    training = set(timesteps["relative_date_indices_training"])
    for kind in ("input", "output"):
        indices = timesteps[f"{kind}_relative_date_indices"]
        missing = sorted(set(indices) - training)
        if missing:
            errors.append(
                f"{prefix}.{kind}_relative_date_indices: {missing} not in relative_date_indices_training "
                f"{sorted(training)}",
            )
        if offsets and len(timesteps[f"{kind}_offsets"]) != len(indices):
            errors.append(
                f"{prefix}.{kind}_offsets: {len(timesteps[f'{kind}_offsets'])} offsets for "
                f"{len(indices)} {kind}_relative_date_indices",
            )
    return errors


def _validate_task_timesteps(
    inference: Mapping[str, Any],
    names: list[str],
    task: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate the ``timesteps`` block of every dataset; return the valid ones by dataset name."""
    model = _OffsetTimestepsV1_0 if task in _OFFSET_TASKS else _TimestepsV1_0
    validated: dict[str, dict[str, Any]] = {}
    for name in names:
        block = inference.get(name)
        prefix = f"task: metadata_inference.{name}.timesteps"
        timesteps = block.get("timesteps") if isinstance(block, dict) else None
        if not isinstance(timesteps, dict):
            if isinstance(block, dict):
                errors.append(f"{prefix}: required and must be a dict")
            continue
        try:
            model.model_validate(timesteps)
        except ValidationError as exc:
            errors.extend(_format_errors(prefix, exc))
        else:
            validated[name] = timesteps
            errors.extend(_check_timing_consistency(prefix, timesteps, offsets=task in _OFFSET_TASKS))
    return validated


@register_write_schema(TASK, "1.0")
def _validate_task_v1_0(metadata: Mapping[str, Any], errors: list[str]) -> None:
    inference = _inference_block(metadata)
    if inference is None:
        errors.append("metadata_inference: required and must be a dict")
        return
    task = inference.get("task")
    if not isinstance(task, str) or not task:
        errors.append("task: metadata_inference.task is required and must be a non-empty string")
        return

    validated = _validate_task_timesteps(inference, _dataset_names(metadata) or [], task, errors)

    first = next(iter(validated.items()), None)
    if first is not None:
        for name, timesteps in validated.items():
            if timesteps != first[1]:
                errors.append(
                    f"task: metadata_inference.{name}.timesteps differs from {first[0]!r}; "
                    "the task is checkpoint level, so all datasets need the same timesteps",
                )


# ---------------------------------------------------------------------------
# lineage 1.0
# ---------------------------------------------------------------------------


class _LineageEntryV1_0(_Block):
    loading_strategy: StrictStr
    parent_uuid: StrictStr | None


class _LineageV1_0(_Block):
    """The top-level keys owned by the lineage component."""

    uuid: Annotated[StrictStr, Field(min_length=1)]
    parent_uuid: StrictStr | None = None
    checkpoint_lineage: list[_LineageEntryV1_0]

    @model_validator(mode="after")
    def _check_parent(self) -> "_LineageV1_0":
        last = self.checkpoint_lineage[-1].parent_uuid if self.checkpoint_lineage else None
        if self.parent_uuid != last:
            msg = f"parent_uuid {self.parent_uuid!r} (absent is None) differs from the last entry's ({last!r})"
            raise ValueError(msg)
        return self


@register_write_schema(LINEAGE, "1.0")
def _validate_lineage_v1_0(metadata: Mapping[str, Any], errors: list[str]) -> None:
    owned = {key: metadata[key] for key in ("uuid", "parent_uuid", "checkpoint_lineage") if key in metadata}
    try:
        _LineageV1_0.model_validate(owned)
    except ValidationError as exc:
        errors.extend(_format_errors("lineage", exc))


# ---------------------------------------------------------------------------
# Raw dumps: presence only
# ---------------------------------------------------------------------------


def _check_raw_presence(metadata: Mapping[str, Any], errors: list[str]) -> None:
    """Check the raw dumps the readers need; their inner structure is never inspected."""
    if not isinstance(metadata.get("config"), dict):
        errors.append("config: required and must be a dict")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("dataset: required and must be a dict")
        return
    errors.extend(
        f"dataset.{name}: required and must be a dict (empty for a tabular dataset)"
        for name in _dataset_names(metadata) or []
        if not isinstance(dataset.get(name), dict)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _check_stamps(metadata: Mapping[str, Any], errors: list[str]) -> bool:
    """Check ``component_versions``; return whether every component has a known stamp."""
    stamps = metadata.get("component_versions")
    if not isinstance(stamps, dict):
        errors.append(f"component_versions: required and must be a dict with a version for {list(COMPONENTS)}")
        return False

    ok = True
    for component in stamps:
        if component not in COMPONENTS:
            errors.append(f"component_versions.{component}: unknown component; known components are {list(COMPONENTS)}")
            ok = False
    for component in COMPONENTS:
        stamp = stamps.get(component)
        if stamp is None:
            errors.append(f"component_versions.{component}: required")
            ok = False
        elif stamp not in write_schema_versions(component):
            errors.append(
                f"component_versions.{component}: unknown version {stamp!r}; "
                f"can write {write_schema_versions(component)}",
            )
            ok = False
    return ok


def validate_for_write(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate *metadata* strictly and return a copy ready to be written.

    Each structured component is checked against the schema of its stamped
    version in ``component_versions``, which is required.  ``version`` is set to
    :data:`CURRENT_VERSION` when it is absent or ``"3.0"``.  Any other version,
    and any stamp without a schema, is rejected: a version is never relabelled,
    so legacy layouts cannot be written as current.

    Parameters
    ----------
    metadata : dict[str, Any]
        The metadata to write.  It is not modified.

    Returns
    -------
    dict[str, Any]
        A new dictionary with ``version`` set to :data:`CURRENT_VERSION`.

    Raises
    ------
    WriteValidationError
        If the metadata is not valid, or has a version or stamp that is unknown.
    TypeError
        If *metadata* is not a dict.
    """
    if not isinstance(metadata, dict):
        msg = f"metadata must be a dict, got {type(metadata).__name__}"
        raise TypeError(msg)

    errors: list[str] = []
    version = metadata.get("version")
    if version not in _RELABELLED_VERSIONS and version != CURRENT_VERSION:
        errors.append(
            f"version: {version!r} cannot be written; only {CURRENT_VERSION!r} is written "
            "(a missing version or '3.0' is relabelled)",
        )

    if _check_stamps(metadata, errors):
        stamps = metadata["component_versions"]
        for component in COMPONENTS:
            _SCHEMAS[(component, stamps[component])](metadata, errors)
        _check_raw_presence(metadata, errors)

    if errors:
        raise WriteValidationError(errors)

    result = copy.deepcopy(metadata)
    result["version"] = CURRENT_VERSION
    return result
