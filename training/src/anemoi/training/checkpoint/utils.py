# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utility functions for checkpoint operations."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING
from typing import Any

import torch
from hydra.utils import get_class

from .exceptions import CheckpointLoadError
from .exceptions import CheckpointValidationError

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def target_class(cfg: Any) -> type | None:
    """Return the class that the ``_target_`` of ``cfg`` names.

    Parameters
    ----------
    cfg : Any
        A config with a ``_target_`` key, or ``None``.

    Returns
    -------
    type or None
        ``None`` if the target is absent or cannot be imported.
    """
    target = cfg.get("_target_") if cfg is not None else None
    if not target:
        return None
    try:
        return get_class(target)
    except (ImportError, ValueError):
        return None


def validate_checkpoint(checkpoint_data: dict[str, Any], *, check_tensors: bool = False) -> bool:
    """Validate checkpoint structure and contents.

    Performs validation checks on a loaded checkpoint to ensure
    it contains expected keys and valid data.

    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data dictionary to validate
    check_tensors : bool, optional
        Whether to check tensors for NaN/Inf values (default: False).
        Set to True to enable expensive tensor validation for large
        checkpoints. When False, only structural validation is performed.

    Returns
    -------
    bool
        True if checkpoint is valid

    Raises
    ------
    CheckpointValidationError
        If validation fails

    Examples
    --------
    >>> checkpoint = torch.load('model.ckpt')
    >>> is_valid = validate_checkpoint(checkpoint)
    >>> # Skip expensive tensor checks for large checkpoints
    >>> is_valid = validate_checkpoint(checkpoint, check_tensors=False)
    """
    validation_errors = []

    # Check for empty checkpoint
    if not checkpoint_data:
        validation_errors.append("Checkpoint is empty")

    # Check if at least one model key exists
    _validate_model_keys(checkpoint_data, validation_errors)

    # Check for corrupt tensors (optional, can be expensive for large checkpoints)
    if check_tensors:
        _validate_tensors(checkpoint_data, validation_errors)

    if validation_errors:
        msg = "Checkpoint validation failed"
        raise CheckpointValidationError(
            msg,
            validation_errors,
            {"num_keys": len(checkpoint_data)},
        )

    LOGGER.debug("Checkpoint validation passed with %d keys", len(checkpoint_data))
    return True


def _validate_model_keys(checkpoint_data: dict[str, Any], validation_errors: list[str]) -> None:
    """Validate that checkpoint contains model state."""
    model_keys = ["state_dict", "model_state_dict", "model"]
    if not any(key in checkpoint_data for key in model_keys):
        validation_errors.append(f"No model state found. Expected one of: {model_keys}")


def _validate_tensors(checkpoint_data: dict[str, Any], validation_errors: list[str]) -> None:
    """Validate tensor values in checkpoint."""
    for key, value in checkpoint_data.items():
        if isinstance(value, torch.Tensor):
            _check_tensor_validity(key, value, validation_errors)
        elif isinstance(value, dict):
            _validate_nested_tensors(key, value, validation_errors)


def _check_tensor_validity(name: str, tensor: torch.Tensor, validation_errors: list[str]) -> None:
    """Check a single tensor for NaN/Inf values."""
    if torch.isnan(tensor).any():
        validation_errors.append(f"Tensor '{name}' contains NaN values")
    if torch.isinf(tensor).any():
        validation_errors.append(f"Tensor '{name}' contains infinite values")


def _validate_nested_tensors(
    parent_key: str,
    nested_dict: dict,
    validation_errors: list[str],
    *,
    _depth: int = 0,
    max_depth: int = 20,
) -> None:
    """Validate tensors in nested dictionaries (recursive).

    Parameters
    ----------
    parent_key : str
        Dot-separated key prefix for error messages
    nested_dict : dict
        Dictionary to recurse into
    validation_errors : list[str]
        Accumulator for validation error messages
    _depth : int
        Current recursion depth (internal use)
    max_depth : int
        Maximum recursion depth to prevent unbounded recursion (default: 20)
    """
    if _depth >= max_depth:
        LOGGER.debug("Max validation depth (%d) reached at key '%s', skipping deeper nesting", max_depth, parent_key)
        return

    for sub_key, sub_value in nested_dict.items():
        full_key = f"{parent_key}.{sub_key}"
        if isinstance(sub_value, torch.Tensor):
            _check_tensor_validity(full_key, sub_value, validation_errors)
        elif isinstance(sub_value, dict):
            _validate_nested_tensors(full_key, sub_value, validation_errors, _depth=_depth + 1, max_depth=max_depth)


def get_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Extract metadata from a checkpoint file.

    Notes
    -----
    This loads the **entire** checkpoint into CPU memory via
    ``torch.load`` because PyTorch does not support partial loading.
    For very large checkpoints this may consume significant RAM.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file

    Returns
    -------
    dict
        Checkpoint metadata dictionary

    Raises
    ------
    CheckpointLoadError
        If checkpoint cannot be loaded
    CheckpointValidationError
        If metadata cannot be extracted from the loaded data

    Examples
    --------
    >>> metadata = get_checkpoint_metadata(Path('model.ckpt'))
    >>> print(f"Epoch: {metadata.get('epoch', 'unknown')}")
    """
    if not checkpoint_path.exists():
        raise CheckpointLoadError(
            checkpoint_path,
            FileNotFoundError(f"File not found: {checkpoint_path}"),
            {"exists": False},
        )

    import pickle

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError, ValueError) as e:
        raise CheckpointLoadError(checkpoint_path, e, {"operation": "extract_metadata"}) from e

    try:
        # Extract metadata (non-tensor data)
        metadata: dict[str, Any] = {}
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor | dict) or key in [
                "epoch",
                "global_step",
                "iteration",
                "best_score",
            ]:
                metadata[key] = value
            elif key == "metadata" and isinstance(value, dict):
                metadata.update(value)

        # Add file information
        metadata["file_size_mb"] = checkpoint_path.stat().st_size / (1024 * 1024)
        metadata["file_path"] = str(checkpoint_path)

        # Count parameters if state dict exists
        if "state_dict" in checkpoint:
            metadata["num_parameters"] = len(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            metadata["num_parameters"] = len(checkpoint["model_state_dict"])

    except (KeyError, TypeError, AttributeError) as e:
        msg = f"Failed to extract metadata from checkpoint {checkpoint_path}: {e}"
        raise CheckpointValidationError(msg) from e

    return metadata


def calculate_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate checksum of a file.

    Parameters
    ----------
    file_path : Path
        Path to file
    algorithm : str, optional
        Hash algorithm to use (default: 'sha256')

    Returns
    -------
    str
        Hexadecimal checksum string

    Examples
    --------
    >>> checksum = calculate_checksum(Path('model.ckpt'))
    >>> print(f"SHA256: {checksum}")
    """
    with file_path.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def compare_state_dicts(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[set, set, dict[str, tuple[torch.Size, torch.Size]]]:
    """Compare two state dictionaries.

    Compares keys and shapes between source and target state dictionaries
    to identify missing keys, unexpected keys, and shape mismatches.

    Parameters
    ----------
    source : dict
        Source state dictionary
    target : dict
        Target state dictionary

    Returns
    -------
    tuple
        (missing_keys, unexpected_keys, shape_mismatches)
        where shape_mismatches is {key: (source_shape, target_shape)}

    Examples
    --------
    >>> missing, unexpected, mismatches = compare_state_dicts(
    ...     checkpoint['state_dict'],
    ...     model.state_dict()
    ... )
    """
    source_keys = set(source.keys())
    target_keys = set(target.keys())

    missing_keys = target_keys - source_keys
    unexpected_keys = source_keys - target_keys

    shape_mismatches = {}
    for key in source_keys.intersection(target_keys):
        if not isinstance(source[key], torch.Tensor) or not isinstance(target[key], torch.Tensor):
            continue  # Skip non-tensor entries (e.g., num_batches_tracked)
        source_shape = source[key].shape
        target_shape = target[key].shape

        if source_shape != target_shape:
            shape_mismatches[key] = (source_shape, target_shape)

    return missing_keys, unexpected_keys, shape_mismatches


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size string (e.g., '1.5 GB')

    Examples
    --------
    >>> format_size(1536000000)
    '1.43 GB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def estimate_checkpoint_memory(checkpoint_data: dict[str, Any]) -> int:
    """Estimate memory usage of checkpoint data.

    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data dictionary

    Returns
    -------
    int
        Estimated memory usage in bytes

    Examples
    --------
    >>> mem_bytes = estimate_checkpoint_memory(checkpoint)
    >>> print(f"Estimated memory: {format_size(mem_bytes)}")
    """
    total_bytes = 0

    def estimate_tensor_size(tensor: torch.Tensor) -> int:
        """Estimate memory size of a tensor."""
        return tensor.numel() * tensor.element_size()

    def estimate_dict_size(d: dict) -> int:
        """Recursively estimate dictionary size."""
        size = 0
        for value in d.values():
            if isinstance(value, torch.Tensor):
                size += estimate_tensor_size(value)
            elif isinstance(value, dict):
                size += estimate_dict_size(value)
        return size

    for value in checkpoint_data.values():
        if isinstance(value, torch.Tensor):
            total_bytes += estimate_tensor_size(value)
        elif isinstance(value, dict):
            total_bytes += estimate_dict_size(value)

    return total_bytes
