# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint I/O for metadata.

This module provides functions for reading and writing metadata from/to
checkpoint files. It handles ZIP archive manipulation and supports
both the new metadata format and legacy formats.

PyTorch checkpoint files are ZIP archives with a single top-level directory
(e.g. ``archive/``). Metadata is stored under that directory as
``<top-dir>/anemoi-metadata/anemoi.json``. Matching is done by **basename**
so that the exact top-level prefix does not need to be known in advance.
"""

import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

import numpy as np

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


DEFAULT_NAME = "anemoi.json"
DEFAULT_FOLDER = "anemoi-metadata"
DEPRECATED_NAME = "ai-models.json"

# Convenience aliases used by tests and the public API.
METADATA_PATH = f"{DEFAULT_FOLDER}/{DEFAULT_NAME}"
LEGACY_METADATA_PATH = DEPRECATED_NAME


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_metadata_path(zf: zipfile.ZipFile, *, name: str = DEFAULT_NAME) -> str | None:
    """Find the full in-archive path of a metadata file by matching its basename.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open ZipFile object.
    name : str, optional
        Basename to search for (default: ``DEFAULT_NAME``).

    Returns
    -------
    str | None
        Full path within the archive, or ``None`` if not found.
    """
    matches = [b for b in zf.namelist() if os.path.basename(b) == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise CheckpointError(f"Found multiple '{name}' entries in archive.")
    return None


def _find_metadata_path_with_deprecation(zf: zipfile.ZipFile) -> str | None:
    """Find metadata path, falling back to the deprecated name with a warning.

    Tries ``DEFAULT_NAME`` first; if absent, tries ``DEPRECATED_NAME`` and
    emits a deprecation warning.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open ZipFile object.

    Returns
    -------
    str | None
        Full path within the archive, or ``None`` if neither name is found.
    """
    path = _find_metadata_path(zf, name=DEFAULT_NAME)
    if path is not None:
        return path

    path = _find_metadata_path(zf, name=DEPRECATED_NAME)
    if path is not None:
        LOG.warning(
            "The metadata file '%s' is deprecated. " "New versions of checkpoints will write to '%s' instead.",
            DEPRECATED_NAME,
            DEFAULT_NAME,
        )
        return path

    return None


def _get_top_level_directory(zf: zipfile.ZipFile) -> str:
    """Determine the single top-level directory in a ZIP archive.

    PyTorch checkpoints are required to have exactly one top-level directory.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open ZipFile object.

    Returns
    -------
    str
        The top-level directory name (without trailing slash).

    Raises
    ------
    CheckpointError
        If the archive has zero or more than one top-level directory.
    """
    directories: set[str] = set()
    for entry in zf.namelist():
        directory = os.path.dirname(entry)
        if not directory:
            # Flat entry (no directory prefix) -- skip it.
            continue
        # Walk up to the top-level component.
        while os.path.dirname(directory) not in (".", ""):
            directory = os.path.dirname(directory)
        directories.add(directory)

    if len(directories) != 1:
        raise CheckpointError(f"Expected exactly one top-level directory in checkpoint, " f"found: {directories!r}")
    return list(directories)[0]


def _get_supporting_arrays_paths(
    directory: str,
    supporting_arrays: dict | np.ndarray | None,
) -> dict:
    """Build the ``supporting_arrays_paths`` metadata structure recursively.

    Parameters
    ----------
    directory : str
        Current archive directory (e.g. ``"archive/anemoi-metadata"``).
    supporting_arrays : dict | np.ndarray | None
        Arrays to record.  A ``dict`` triggers recursion; an ``ndarray``
        produces a leaf entry; ``None`` returns ``{}``.

    Returns
    -------
    dict
        Nested dict of ``{"path": ..., "shape": [...], "dtype": "..."}``
        leaf entries.
    """
    if supporting_arrays is None:
        return {}

    if isinstance(supporting_arrays, dict):
        return {
            key: _get_supporting_arrays_paths(f"{directory}/{key}", value) for key, value in supporting_arrays.items()
        }

    # Leaf: a single numpy array.
    return {
        "path": f"{directory}.numpy",
        "shape": list(supporting_arrays.shape),
        "dtype": str(supporting_arrays.dtype),
    }


def _write_array_to_bytes(
    array: dict | np.ndarray | None,
    name: str,
    entry: dict,
    zipf: zipfile.ZipFile,
) -> None:
    """Write a supporting array (or nested dict of arrays) into a ZIP file.

    Parameters
    ----------
    array : dict | np.ndarray | None
        Array data to write.
    name : str
        Current key name (used for logging).
    entry : dict
        Corresponding entry from ``supporting_arrays_paths``.
    zipf : zipfile.ZipFile
        Open ZipFile to write into.
    """
    if array is None:
        return

    if isinstance(array, dict):
        for sub_name, sub_array in array.items():
            _write_array_to_bytes(sub_array, sub_name, entry.get(sub_name, {}), zipf)
        return

    LOG.info(
        "Saving supporting array '%s' to %s (shape=%s, dtype=%s)",
        name,
        entry["path"],
        entry["shape"],
        entry["dtype"],
    )
    zipf.writestr(entry["path"], array.tobytes())


def _load_supporting_arrays(zf: zipfile.ZipFile, entries: dict) -> dict[str, Any]:
    """Load supporting numpy arrays from a ZIP file.

    Recursively handles nested (multi-dataset) structures.  A leaf entry is
    a dict with exactly the keys ``{"path", "shape", "dtype"}``; anything
    else is treated as a nested group.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open ZipFile object.
    entries : dict
        The ``supporting_arrays_paths`` dict (or a sub-dict thereof).

    Returns
    -------
    dict[str, Any]
        Mapping of key → numpy array (or nested dict of arrays).
    """
    result: dict[str, Any] = {}
    for key, entry in entries.items():
        if isinstance(entry, dict) and set(entry.keys()) != {"path", "shape", "dtype"}:
            # Nested group — recurse.
            result[key] = _load_supporting_arrays(zf, entry)
        else:
            result[key] = np.frombuffer(
                zf.read(entry["path"]),
                dtype=entry["dtype"],
            ).reshape(entry["shape"])
    return result


def _edit_metadata(
    path: Path,
    metadata_archive_path: str,
    new_metadata_json: str | None,
    supporting_arrays: dict | None = None,
    *,
    target_archive_path: str | None = None,
) -> None:
    """Rebuild a ZIP archive, replacing or removing the metadata entry.

    Copies every entry from the source archive into a new temporary file,
    skipping the old metadata entry (and its associated array files).  If
    ``new_metadata_json`` is not ``None`` the new metadata is written at
    ``target_archive_path`` (defaults to ``metadata_archive_path`` if not
    specified, effectively migrating deprecated paths to the canonical one).

    Parameters
    ----------
    path : Path
        Path to the checkpoint file (modified in-place).
    metadata_archive_path : str
        In-archive path of the **old** metadata JSON to remove/replace.
    new_metadata_json : str | None
        Serialised JSON to write as the new metadata, or ``None`` to remove.
    supporting_arrays : dict | None, optional
        New supporting arrays to write alongside the new metadata.
    target_archive_path : str | None, optional
        In-archive path where the new metadata should be written.  Defaults
        to ``metadata_archive_path``.  Set this to migrate from a deprecated
        path (e.g. ``ai-models.json``) to the canonical path.
    """
    if target_archive_path is None:
        target_archive_path = metadata_archive_path

    tmp_path = path.with_suffix(f".anemoi-edit-{time.time()}-{os.getpid()}.tmp")

    old_directory = os.path.dirname(metadata_archive_path)

    # Determine which archive entries to skip (old metadata + old arrays).
    skip_paths: set[str] = {metadata_archive_path}

    try:
        with zipfile.ZipFile(path, "r") as src_zf:
            # Collect old array paths by scanning for .numpy files under the
            # same directory as the old metadata.
            for entry in src_zf.namelist():
                if entry.startswith(old_directory) and entry.endswith(".numpy"):
                    skip_paths.add(entry)

            with zipfile.ZipFile(tmp_path, "w") as dst_zf:
                # Copy everything except the entries being replaced,
                # preserving original compression method per entry.
                for entry in src_zf.namelist():
                    if entry not in skip_paths:
                        info = src_zf.getinfo(entry)
                        dst_zf.writestr(info, src_zf.read(entry))

                # Write new metadata at the target path (compressed).
                if new_metadata_json is not None:
                    dst_zf.writestr(
                        target_archive_path,
                        new_metadata_json,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )

                    # Write new supporting arrays (uncompressed -- binary data).
                    if supporting_arrays:
                        metadata_dict = json.loads(new_metadata_json)
                        array_paths = metadata_dict.get("supporting_arrays_paths", {})
                        _write_array_to_bytes(supporting_arrays, "", array_paths, dst_zf)

        tmp_path.replace(path)
        LOG.info("Updated metadata in %s", path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


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
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            return _find_metadata_path_with_deprecation(zf) is not None
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
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            metadata_path = _find_metadata_path_with_deprecation(zf)
            if metadata_path is None:
                raise CheckpointError(f"No metadata found in checkpoint: {checkpoint_path}")
            with zf.open(metadata_path) as f:
                return json.load(f)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Invalid metadata JSON in {checkpoint_path}") from exc


@overload
def load_metadata(
    path: str | Path,
    *,
    migrate: bool = True,
    supporting_arrays: Literal[False],
) -> "MetadataContract": ...


@overload
def load_metadata(
    path: str | Path,
    *,
    migrate: bool = True,
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
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            metadata_archive_path = _find_metadata_path_with_deprecation(zf)
            if metadata_archive_path is None:
                raise CheckpointError(f"No metadata found in checkpoint: {checkpoint_path}")

            with zf.open(metadata_archive_path) as f:
                data = json.load(f)

            metadata = MetadataRegistry.load(data, migrate=migrate)

            if supporting_arrays:
                arrays = _load_supporting_arrays(zf, data.get("supporting_arrays_paths", {}))
                return metadata, arrays

            return metadata

    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc
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
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            if _find_metadata_path_with_deprecation(zf) is not None:
                raise CheckpointError(
                    f"Checkpoint already contains metadata: {checkpoint_path}. " "Use replace_metadata() to overwrite."
                )
            directory = _get_top_level_directory(zf)
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc

    folder = DEFAULT_FOLDER
    name = DEFAULT_NAME

    metadata_dict = metadata_dict.copy()
    metadata_dict["supporting_arrays_paths"] = _get_supporting_arrays_paths(f"{directory}/{folder}", supporting_arrays)

    LOG.info("Saving metadata to %s/%s/%s", directory, folder, name)

    with zipfile.ZipFile(checkpoint_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{directory}/{folder}/{name}",
            json.dumps(metadata_dict),
        )
        _write_array_to_bytes(supporting_arrays, "", metadata_dict["supporting_arrays_paths"], zf)


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

    try:
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            metadata_archive_path = _find_metadata_path_with_deprecation(zf)
            if metadata_archive_path is None:
                raise CheckpointError(f"No metadata found in checkpoint: {checkpoint_path}")
            directory = os.path.dirname(os.path.dirname(metadata_archive_path))
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc

    folder = DEFAULT_FOLDER
    name = DEFAULT_NAME

    metadata_dict = metadata_dict.copy()
    metadata_dict["supporting_arrays_paths"] = _get_supporting_arrays_paths(f"{directory}/{folder}", supporting_arrays)

    new_archive_path = f"{directory}/{folder}/{name}"
    new_metadata_json = json.dumps(metadata_dict)

    _edit_metadata(
        checkpoint_path,
        metadata_archive_path,
        new_metadata_json,
        supporting_arrays,
        target_archive_path=new_archive_path,
    )


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
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            metadata_archive_path = _find_metadata_path_with_deprecation(zf)
            if metadata_archive_path is None:
                # Nothing to remove — treat as a no-op.
                return
    except zipfile.BadZipFile as exc:
        raise CheckpointError(f"Invalid checkpoint file: {checkpoint_path}") from exc

    _edit_metadata(checkpoint_path, metadata_archive_path, new_metadata_json=None)
