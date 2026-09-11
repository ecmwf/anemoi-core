# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint functions for metadata.

This module provides functions for reading and writing metadata from/to
checkpoint files. It delegates the actual functions handling the I/O
to anemoi-utils, but provides a higher-level interface for working with checkpoint files in
the context of the metadata objects.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

import anemoi.utils.checkpoints as checkpoints_utils

from .exceptions import CheckpointError
from .registry import MetadataRegistry

if TYPE_CHECKING:
    from .base import MetadataContract

LOG = logging.getLogger(__name__)


def _resolve_metadata(
    metadata: "MetadataContract | dict[str, Any]",
) -> "MetadataContract":
    """Coerce *metadata* to a validated :class:`MetadataContract` instance.

    If *metadata* is already a :class:`MetadataContract`, it is returned
    unchanged.  If it is a ``dict``, it is validated through
    :meth:`MetadataRegistry.load` (which handles version detection, defaults,
    and schema validation).

    Parameters
    ----------
    metadata : MetadataContract | dict[str, Any]
        Either a validated instance or a raw dict.

    Returns
    -------
    MetadataContract
        Validated metadata instance guaranteed to have ``schema_version``.

    Raises
    ------
    TypeError
        If *metadata* is neither a MetadataContract nor a dict.
    """
    from .base import MetadataContract as _MC

    if isinstance(metadata, _MC):
        return metadata

    if isinstance(metadata, dict):
        return MetadataRegistry.load(metadata, migrate=False)

    raise TypeError(f"metadata must be a MetadataContract instance or a dict, got {type(metadata).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_metadata(path: str | Path) -> bool:
    """Check if a checkpoint contains metadata.

    Searches by basename so that the top-level archive directory prefix is
    irrelevant.  Falls back to the deprecated ``ai-models.json`` name.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.

    Returns
    -------
    bool
        ``True`` if the checkpoint contains metadata.

    Raises
    ------
    CheckpointError
        If the file exists but is not a valid ZIP archive.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        return False

    try:
        return checkpoints_utils.has_metadata(str(checkpoint_path))
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc


def extract_metadata_dict(path: str | Path) -> dict[str, Any]:
    """Extract raw metadata dictionary from a checkpoint.

    Returns the raw dictionary without validation or migration.
    Use :func:`load_metadata` for typed, validated metadata.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.

    Returns
    -------
    dict[str, Any]
        Raw metadata dictionary.

    Raises
    ------
    CheckpointError
        If metadata is not found or the file is invalid.
    """
    checkpoint_path = Path(path)

    try:
        return checkpoints_utils.load_metadata(str(checkpoint_path), supporting_arrays=False)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc
    except FileNotFoundError as exc:
        raise CheckpointError(f"No metadata found in checkpoint: {checkpoint_path}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Invalid metadata JSON in {checkpoint_path}") from exc


@overload
def load_metadata(
    path: str | Path,
    *,
    migrate: bool,
    supporting_arrays: Literal[False],
) -> "MetadataContract": ...


@overload
def load_metadata(
    path: str | Path,
    *,
    migrate: bool,
    supporting_arrays: Literal[True],
) -> "tuple[MetadataContract, dict[str, Any]]": ...


def load_metadata(
    path: str | Path,
    *,
    migrate: bool = True,
    supporting_arrays: bool = False,
) -> "MetadataContract | tuple[MetadataContract, dict[str, Any]]":
    """Load metadata from a checkpoint file.

    This is the primary function for loading checkpoint metadata.  It handles
    version detection, validation, and optional migration to the latest version.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    migrate : bool, optional
        If ``True`` (default), auto-migrate to the latest schema version.
    supporting_arrays : bool, optional
        If ``True``, also load numpy arrays stored alongside the metadata.
        Returns a tuple of ``(metadata, arrays_dict)``.

    Returns
    -------
    MetadataContract | tuple[MetadataContract, dict[str, Any]]
        Validated metadata instance, or a tuple of ``(metadata, arrays)``
        when *supporting_arrays* is ``True``.

    Raises
    ------
    CheckpointError
        If the file is invalid or metadata is missing.
    UnknownVersionError
        If the metadata version is not recognised.

    Examples
    --------
    >>> metadata = load_metadata("model.ckpt")
    >>> print(metadata.schema_version)
    '1.0'

    >>> metadata, arrays = load_metadata("model.ckpt", supporting_arrays=True)
    >>> print(list(arrays.keys()))
    ['latitudes', 'longitudes']
    """
    checkpoint_path = Path(path)

    try:
        data, arrays = checkpoints_utils.load_metadata(str(checkpoint_path), supporting_arrays=True)
        metadata = MetadataRegistry.load(data, migrate=migrate)

        if supporting_arrays:
            return metadata, arrays
        return metadata

    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc
    except FileNotFoundError as exc:
        raise CheckpointError(f"No metadata found in checkpoint: {checkpoint_path}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Invalid metadata JSON in {checkpoint_path}") from exc


def save_metadata(
    path: str | Path,
    metadata: "MetadataContract | dict[str, Any]",
    *,
    supporting_arrays: dict[str, Any] | None = None,
) -> None:
    """Save metadata to an existing checkpoint file.

    Appends metadata to a PyTorch-style ZIP checkpoint.  If the checkpoint
    already contains metadata, use :func:`replace_metadata` instead.

    The top-level archive directory is discovered automatically (PyTorch
    checkpoints have exactly one).

    When a raw ``dict`` is passed it is validated through
    :meth:`MetadataRegistry.load` to ensure it conforms to a registered schema
    and has ``schema_version`` set.  Prefer passing a :class:`MetadataContract`
    instance directly to avoid the overhead of re-validation.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    metadata : MetadataContract | dict[str, Any]
        Validated metadata instance, or a raw dict that will be validated
        through the registry before writing.
    supporting_arrays : dict[str, Any] | None, optional
        Optional dictionary of numpy arrays to store alongside the metadata.

    Raises
    ------
    CheckpointError
        If the file does not exist, already contains metadata, or if the
        archive structure is invalid (zero or multiple top-level directories).
    TypeError
        If *metadata* is neither a MetadataContract nor a dict.
    """
    checkpoint_path = Path(path)
    metadata_obj = _resolve_metadata(metadata)
    metadata_dict = metadata_obj.to_dict()

    if not checkpoint_path.exists():
        raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoints_utils.save_metadata(str(checkpoint_path), metadata_dict, supporting_arrays=supporting_arrays)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc
    except ValueError as exc:
        raise CheckpointError(f"Checkpoint {checkpoint_path} already contains metadata") from exc


def replace_metadata(
    path: str | Path,
    metadata: "MetadataContract | dict[str, Any]",
    *,
    supporting_arrays: dict[str, Any] | None = None,
) -> None:
    """Replace metadata in an existing checkpoint file.

    Rebuilds the ZIP archive, substituting the existing metadata entry with
    the new one.

    When a raw ``dict`` is passed it is validated through
    :meth:`MetadataRegistry.load` to ensure it conforms to a registered schema
    and has ``schema_version`` set.  Prefer passing a :class:`MetadataContract`
    instance directly.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    metadata : MetadataContract | dict[str, Any]
        Validated metadata instance, or a raw dict that will be validated
        through the registry before writing.
    supporting_arrays : dict[str, Any] | None, optional
        Optional dictionary of numpy arrays to store alongside the metadata.

    Raises
    ------
    CheckpointError
        If the checkpoint file does not exist or contains no metadata.
    TypeError
        If *metadata* is neither a MetadataContract nor a dict.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")

    metadata_obj = _resolve_metadata(metadata)
    metadata_dict = metadata_obj.to_dict()
    metadata_dict.setdefault("version", metadata_obj.schema_version)

    try:
        checkpoints_utils.replace_metadata(str(checkpoint_path), metadata_dict, supporting_arrays=supporting_arrays)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc


def remove_metadata(path: str | Path) -> None:
    """Remove metadata from a checkpoint file.

    Rebuilds the ZIP archive, omitting the metadata JSON and any associated
    supporting array files.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.

    Raises
    ------
    CheckpointError
        If the checkpoint file does not exist.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoints_utils.remove_metadata(str(checkpoint_path))
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc
