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
from typing import Any
from typing import Literal
from typing import overload

import anemoi.utils.checkpoints as checkpoints_utils

from .base import CheckpointAdapter
from .exceptions import CheckpointError
from .write_schema import validate_for_write

LOG = logging.getLogger(__name__)


def _resolve_metadata(
    metadata: "CheckpointAdapter | dict[str, Any]",
) -> "CheckpointAdapter":
    """Coerce *metadata* to a :class:`CheckpointAdapter` instance.

    If *metadata* is already a :class:`CheckpointAdapter`, it is returned
    unchanged.  If it is a ``dict``, it is wrapped in a :class:`CheckpointAdapter`
    (which detects the format and the component versions).  Reading is
    lenient and the content is never altered.

    Parameters
    ----------
    metadata : CheckpointAdapter | dict[str, Any]
        Either an adapter or a raw dict.

    Returns
    -------
    CheckpointAdapter
        Adapter holding the metadata.

    Raises
    ------
    TypeError
        If *metadata* is neither a CheckpointAdapter nor a dict.
    """
    if isinstance(metadata, CheckpointAdapter):
        return metadata

    if isinstance(metadata, dict):
        return CheckpointAdapter(metadata)

    msg = f"metadata must be a CheckpointAdapter instance or a dict, got {type(metadata).__name__}"
    raise TypeError(msg)


def _metadata_to_write(metadata: "CheckpointAdapter | dict[str, Any]", *, validate: bool) -> dict[str, Any]:
    """Return the dict to store: strictly validated and stamped, or unchanged when *validate* is false."""
    if validate and isinstance(metadata, dict):
        # Validate the dict itself: wrapping it in an adapter first would read it leniently.
        return validate_for_write(metadata)
    metadata_dict = _resolve_metadata(metadata).to_dict()
    return validate_for_write(metadata_dict) if validate else metadata_dict


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
        msg = f"Invalid checkpoint file: {checkpoint_path}"
        raise CheckpointError(msg) from exc


def extract_metadata_dict(path: str | Path) -> dict[str, Any]:
    """Extract raw metadata dictionary from a checkpoint.

    Returns the raw dictionary exactly as stored.
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
        msg = f"Invalid checkpoint file: {checkpoint_path}"
        raise CheckpointError(msg) from exc
    except FileNotFoundError as exc:
        msg = f"No metadata found in checkpoint: {checkpoint_path}"
        raise CheckpointError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Invalid metadata JSON in {checkpoint_path}"
        raise CheckpointError(msg) from exc


@overload
def load_metadata(
    path: str | Path,
    *,
    supporting_arrays: Literal[False] = ...,
) -> "CheckpointAdapter": ...


@overload
def load_metadata(
    path: str | Path,
    *,
    supporting_arrays: Literal[True],
) -> "tuple[CheckpointAdapter, dict[str, Any]]": ...


def load_metadata(
    path: str | Path,
    *,
    supporting_arrays: bool = False,
) -> "CheckpointAdapter | tuple[CheckpointAdapter, dict[str, Any]]":
    """Load metadata from a checkpoint file.

    This is the primary function for loading checkpoint metadata.  It detects
    the format and the version of each component and returns a
    :class:`CheckpointAdapter` over the metadata, which is kept unchanged.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    supporting_arrays : bool, optional
        If ``True``, also load numpy arrays stored alongside the metadata.
        Returns a tuple of ``(metadata, arrays_dict)``.

    Returns
    -------
    CheckpointAdapter | tuple[CheckpointAdapter, dict[str, Any]]
        Adapter over the metadata, or a tuple of ``(metadata, arrays)``
        when *supporting_arrays* is ``True``.

    Raises
    ------
    CheckpointError
        If the file is invalid or metadata is missing.

    Examples
    --------
    >>> metadata = load_metadata("model.ckpt")
    >>> print(metadata.version)
    '2.0'

    >>> metadata, arrays = load_metadata("model.ckpt", supporting_arrays=True)
    >>> print(list(arrays.keys()))
    ['latitudes', 'longitudes']
    """
    checkpoint_path = Path(path)

    try:
        data, arrays = checkpoints_utils.load_metadata(str(checkpoint_path), supporting_arrays=True)
        metadata = CheckpointAdapter(data)
    except zipfile.BadZipFile as exc:
        msg = f"Invalid checkpoint file: {checkpoint_path}"
        raise CheckpointError(msg) from exc
    except FileNotFoundError as exc:
        msg = f"No metadata found in checkpoint: {checkpoint_path}"
        raise CheckpointError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Invalid metadata JSON in {checkpoint_path}"
        raise CheckpointError(msg) from exc

    if supporting_arrays:
        return metadata, arrays
    return metadata


def save_metadata(
    path: str | Path,
    metadata: "CheckpointAdapter | dict[str, Any]",
    *,
    supporting_arrays: dict[str, Any] | None = None,
    validate: bool = True,
) -> None:
    """Save metadata to an existing checkpoint file.

    Appends metadata to a PyTorch-style ZIP checkpoint.  If the checkpoint
    already contains metadata, use :func:`replace_metadata` instead.

    The top-level archive directory is discovered automatically (PyTorch
    checkpoints have exactly one).

    By default the metadata is validated strictly and stamped with
    :data:`~anemoi.metadata.write_schema.CURRENT_VERSION` by
    :func:`~anemoi.metadata.write_schema.validate_for_write`.  With
    ``validate=False`` the raw content is written unchanged (for editing tools
    and historical checkpoints).

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    metadata : CheckpointAdapter | dict[str, Any]
        Adapter over the metadata, or a raw dict.
    supporting_arrays : dict[str, Any] | None, optional
        Optional dictionary of numpy arrays to store alongside the metadata.
    validate : bool, optional
        Validate strictly before writing (default).  ``False`` writes the
        metadata as it is.

    Raises
    ------
    CheckpointError
        If the file does not exist, already contains metadata, or if the
        archive structure is invalid (zero or multiple top-level directories).
    WriteValidationError
        If *validate* is true and the metadata is not valid for writing.
    TypeError
        If *metadata* is neither a CheckpointAdapter nor a dict.
    """
    checkpoint_path = Path(path)
    metadata_dict = _metadata_to_write(metadata, validate=validate)

    if not checkpoint_path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise CheckpointError(msg)

    try:
        checkpoints_utils.save_metadata(str(checkpoint_path), metadata_dict, supporting_arrays=supporting_arrays)
    except zipfile.BadZipFile as exc:
        msg = f"Invalid checkpoint file: {checkpoint_path}"
        raise CheckpointError(msg) from exc
    except ValueError as exc:
        msg = f"Checkpoint {checkpoint_path} already contains metadata"
        raise CheckpointError(msg) from exc


def replace_metadata(
    path: str | Path,
    metadata: "CheckpointAdapter | dict[str, Any]",
    *,
    supporting_arrays: dict[str, Any] | None = None,
    validate: bool = True,
) -> None:
    """Replace metadata in an existing checkpoint file.

    Rebuilds the ZIP archive, substituting the existing metadata entry with
    the new one.  By default the metadata is validated strictly and stamped
    with :data:`~anemoi.metadata.write_schema.CURRENT_VERSION` by
    :func:`~anemoi.metadata.write_schema.validate_for_write`.  With
    ``validate=False`` the raw content is written unchanged (for editing tools
    and historical checkpoints).

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    metadata : CheckpointAdapter | dict[str, Any]
        Adapter over the metadata, or a raw dict.
    supporting_arrays : dict[str, Any] | None, optional
        Optional dictionary of numpy arrays to store alongside the metadata.
    validate : bool, optional
        Validate strictly before writing (default).  ``False`` writes the
        metadata as it is.

    Raises
    ------
    CheckpointError
        If the checkpoint file does not exist, contains no metadata, or (with
        ``validate=False``) the metadata has no ``version`` key (required by
        the checkpoint writer).
    WriteValidationError
        If *validate* is true and the metadata is not valid for writing.
    TypeError
        If *metadata* is neither a CheckpointAdapter nor a dict.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise CheckpointError(msg)

    metadata_dict = _metadata_to_write(metadata, validate=validate)

    try:
        checkpoints_utils.replace_metadata(str(checkpoint_path), metadata_dict, supporting_arrays=supporting_arrays)
    except zipfile.BadZipFile as exc:
        msg = f"Invalid checkpoint file: {checkpoint_path}"
        raise CheckpointError(msg) from exc
    except ValueError as exc:
        msg = f"Cannot replace metadata in {checkpoint_path}: {exc}"
        raise CheckpointError(msg) from exc


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
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise CheckpointError(msg)

    try:
        checkpoints_utils.remove_metadata(str(checkpoint_path))
    except zipfile.BadZipFile as exc:
        msg = f"Invalid checkpoint file: {checkpoint_path}"
        raise CheckpointError(msg) from exc
