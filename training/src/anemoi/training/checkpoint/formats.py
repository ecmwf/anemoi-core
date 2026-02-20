# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint format detection and conversion utilities.

This module provides utilities for detecting checkpoint formats and converting
between different checkpoint types (Lightning, PyTorch, state_dict).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any
from typing import Literal

import torch

LOGGER = logging.getLogger(__name__)


def detect_format_from_data(
    checkpoint_data: dict[str, Any],
) -> Literal["lightning", "pytorch", "state_dict"]:
    """Detect checkpoint format from already-loaded data by inspecting dict keys.

    This avoids re-loading the checkpoint from disk when the data is already
    in memory (e.g. after a source stage has loaded it).

    Parameters
    ----------
    checkpoint_data : dict
        Already-loaded checkpoint data dictionary

    Returns
    -------
    str
        Format of the checkpoint: "lightning", "pytorch", or "state_dict"
    """
    if not isinstance(checkpoint_data, dict):
        return "pytorch"

    # Check for Lightning-specific keys (exclude generic training state keys)
    lightning_specific_keys = {
        "pytorch-lightning_version",
        "callbacks",
        "optimizer_states",  # Lightning uses plural
        "lr_schedulers",  # Lightning uses plural
        "loops",
        "hyper_parameters",
    }

    if any(key in checkpoint_data for key in lightning_specific_keys):
        return "lightning"

    # Check for PyTorch-specific structure
    pytorch_keys = {
        "model_state_dict",  # PyTorch uses this specific key
        "optimizer_state_dict",  # PyTorch uses singular
        "scheduler_state_dict",  # PyTorch uses singular
    }

    if any(key in checkpoint_data for key in pytorch_keys):
        return "pytorch"

    # If it's just a dict of tensors, it's a state dict
    if checkpoint_data and all(isinstance(v, torch.Tensor) for v in checkpoint_data.values()):
        return "state_dict"

    # Default to pytorch for structured checkpoints that don't match specific patterns
    return "pytorch"


_SUPPORTED_EXTENSIONS = frozenset({".ckpt", ".pt", ".pth", ".bin"})


def detect_checkpoint_format(
    checkpoint_path: Path | str,
) -> Literal["lightning", "pytorch", "state_dict"]:
    """Detect the format of a checkpoint file.

    Uses file extension and structure inspection to determine format.
    For already-loaded data, prefer :func:`detect_format_from_data`.

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to the checkpoint file

    Returns
    -------
    str
        Format of the checkpoint: "lightning", "pytorch", or "state_dict"

    Raises
    ------
    CheckpointLoadError
        If the checkpoint file cannot be loaded (corrupted, empty, etc.)
    CheckpointConfigError
        If the file extension is not supported
    """
    path = Path(checkpoint_path)

    # Check file extension first
    extension = path.suffix.lower()

    if extension not in _SUPPORTED_EXTENSIONS:
        from .exceptions import CheckpointConfigError

        msg = (
            f"Unsupported checkpoint file extension '{extension}' for {path}. "
            f"Supported extensions: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )
        raise CheckpointConfigError(msg)

    # For supported extensions, load and inspect structure
    try:
        # SECURITY: weights_only=False is required for Anemoi checkpoints that
        # contain non-tensor metadata (hyper_parameters, callbacks, etc.).
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError) as e:
        from .exceptions import CheckpointLoadError

        raise CheckpointLoadError(path, e) from e

    return detect_format_from_data(checkpoint)


def load_checkpoint(
    checkpoint_path: Path | str,
    checkpoint_format: Literal["lightning", "pytorch", "state_dict"] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint file in any supported format.

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to the checkpoint file
    checkpoint_format : str, optional
        Format of the checkpoint. If None, will auto-detect from data.

    Returns
    -------
    dict
        Loaded checkpoint data
    """
    path = Path(checkpoint_path)

    if not path.exists():
        from .exceptions import CheckpointNotFoundError

        raise CheckpointNotFoundError(path)

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

    except FileNotFoundError:
        from .exceptions import CheckpointNotFoundError

        raise CheckpointNotFoundError(path) from None

    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError, ValueError) as e:
        from .exceptions import CheckpointLoadError

        raise CheckpointLoadError(path, e) from e

    if checkpoint_format is None:
        LOGGER.debug("Auto-detecting checkpoint format from loaded data")
        detect_format_from_data(data)

    return data


def extract_state_dict(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    """Extract the state dict from a checkpoint.

    Handles different checkpoint structures.

    Parameters
    ----------
    checkpoint_data : dict
        Loaded checkpoint data

    Returns
    -------
    dict
        Extracted state dictionary
    """
    if not isinstance(checkpoint_data, dict):
        from .exceptions import CheckpointValidationError

        msg = (
            "Cannot extract state dict: checkpoint data is not a dictionary. "
            f"Got {type(checkpoint_data).__name__} instead. "
            "This might indicate a corrupted or incompatible checkpoint file."
        )
        raise CheckpointValidationError(msg)

    # Try common state dict keys in order of preference
    for key in ["state_dict", "model_state_dict", "model"]:
        if key in checkpoint_data:
            state_dict = checkpoint_data[key]
            if not isinstance(state_dict, dict):
                from .exceptions import CheckpointValidationError

                msg = (
                    f"State dict under key '{key}' is not a dictionary. "
                    f"Got {type(state_dict).__name__} instead. "
                    "This indicates an incompatible checkpoint format."
                )
                raise CheckpointValidationError(msg)
            return state_dict

    # Check if the checkpoint itself looks like a state dict
    if checkpoint_data and all(isinstance(v, torch.Tensor) for v in checkpoint_data.values()):
        # Looks like a raw state dict
        return checkpoint_data

    # If we get here, provide helpful guidance
    available_keys = list(checkpoint_data.keys())[:10]  # Show first 10 keys
    from .exceptions import CheckpointValidationError

    error_msg = (
        "Cannot find model state in checkpoint. "
        "Expected one of: 'state_dict', 'model_state_dict', 'model' "
        f"or a raw state dictionary with tensor values.\n"
        f"Found keys: {available_keys}"
    )

    if available_keys:
        if any("model" in key.lower() for key in available_keys):
            error_msg += "\nSuggestion: Found model-related keys - this checkpoint might use a non-standard format"
        elif any(key.endswith(("_state_dict", "_dict")) for key in available_keys):
            error_msg += "\nSuggestion: Found state_dict-like keys - check if the structure is nested differently"
        else:
            error_msg += "\nSuggestion: This may not be a valid model checkpoint file"

    raise CheckpointValidationError(error_msg)


def save_checkpoint(
    checkpoint_data: dict[str, Any],
    checkpoint_path: Path | str,
    checkpoint_format: Literal["lightning", "pytorch", "state_dict"] = "pytorch",  # noqa: ARG001
    anemoi_metadata: dict[str, Any] | None = None,
    supporting_arrays: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint in the specified format with optional Anemoi metadata.

    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data to save
    checkpoint_path : Path or str
        Path where to save the checkpoint
    checkpoint_format : str
        Format to save in: "lightning", "pytorch", or "state_dict"
    anemoi_metadata : dict, optional
        Anemoi-specific metadata to save alongside the checkpoint.
        Will be saved using anemoi.utils.checkpoints.save_metadata.
    supporting_arrays : dict, optional
        Supporting arrays to save with metadata (e.g., statistics, indices)
    """
    path = Path(checkpoint_path)

    # Ensure parent directory exists
    _ensure_directory_exists(path.parent)

    # Save checkpoint in the appropriate format
    _save_checkpoint_file(checkpoint_data, path)

    # Save Anemoi metadata if provided
    _handle_anemoi_metadata(path, anemoi_metadata, supporting_arrays)


def _ensure_directory_exists(directory: Path) -> None:
    """Ensure directory exists, with helpful error messages on failure.

    Parameters
    ----------
    directory : Path
        Directory path to create

    Raises
    ------
    OSError
        If directory cannot be created with helpful suggestions
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = _build_directory_error_message(directory, e)
        raise OSError(msg) from e


def _build_directory_error_message(directory: Path, error: OSError) -> str:
    """Build helpful error message for directory creation failures.

    Parameters
    ----------
    directory : Path
        Directory that failed to create
    error : OSError
        The original error

    Returns
    -------
    str
        Detailed error message with suggestions
    """
    error_type = type(error).__name__
    error_str = str(error)

    if "Permission denied" in error_str or error_type == "PermissionError":
        return (
            f"Cannot create directory {directory}: Permission denied.\n"
            "Suggestions:\n"
            "  • Check directory permissions\n"
            "  • Try running with appropriate privileges\n"
            "  • Use a different save location where you have write access"
        )
    if "No space left on device" in error_str:
        return (
            f"Cannot create directory {directory}: No space left on device.\n"
            "Suggestions:\n"
            "  • Free up disk space\n"
            "  • Use a different save location with more available space\n"
            "  • Check disk usage with 'df -h'"
        )
    return (
        f"Cannot create directory {directory}: {error}\n"
        f"Error type: {error_type}\n"
        "Check directory path and file system permissions."
    )


def _save_checkpoint_file(
    checkpoint_data: dict[str, Any],
    path: Path,
) -> None:
    """Save checkpoint file using ``torch.save``.

    All checkpoint formats are saved via ``torch.save``. Alternative
    serialisation backends (e.g. safetensors) can be added in the
    future by extending this function.

    Parameters
    ----------
    checkpoint_data : dict
        Data to save
    path : Path
        File path to save to

    Raises
    ------
    OSError
        If file cannot be saved
    """
    try:
        torch.save(checkpoint_data, path)
    except OSError as e:
        msg = _build_save_error_message(path, e)
        raise OSError(msg) from e
    except Exception as e:
        # Catch any other errors during serialization
        msg = (
            f"Failed to serialize checkpoint data for saving to {path}: {e}\n"
            f"Error type: {type(e).__name__}\n"
            "This might indicate incompatible data types in the checkpoint."
        )
        raise RuntimeError(msg) from e


def _build_save_error_message(path: Path, error: OSError) -> str:
    """Build helpful error message for save failures.

    Parameters
    ----------
    path : Path
        File path that failed to save
    error : OSError
        The original error

    Returns
    -------
    str
        Detailed error message with suggestions
    """
    error_type = type(error).__name__
    error_str = str(error)

    if "Permission denied" in error_str or error_type == "PermissionError":
        return (
            f"Cannot save checkpoint to {path}: Permission denied.\n"
            "Suggestions:\n"
            "  • Check file and directory permissions\n"
            "  • Try running with appropriate privileges\n"
            "  • Use a different save location where you have write access"
        )
    if "No space left on device" in error_str:
        return (
            f"Cannot save checkpoint to {path}: No space left on device.\n"
            "Suggestions:\n"
            "  • Free up disk space\n"
            "  • Use a different save location with more available space\n"
            "  • Check disk usage with 'df -h'"
        )
    return f"Failed to save checkpoint to {path}: {error}"


def _handle_anemoi_metadata(
    path: Path,
    anemoi_metadata: dict[str, Any] | None,
    supporting_arrays: dict[str, Any] | None,
) -> None:
    """Save Anemoi metadata if provided.

    Parameters
    ----------
    path : Path
        Checkpoint file path
    anemoi_metadata : dict, optional
        Anemoi-specific metadata to save
    supporting_arrays : dict, optional
        Supporting arrays to save with metadata
    """
    if anemoi_metadata is not None:
        try:
            from anemoi.utils.checkpoints import save_metadata

            LOGGER.debug("Saving Anemoi metadata for checkpoint %s", path)
            save_metadata(path, anemoi_metadata, supporting_arrays=supporting_arrays)
            LOGGER.debug("Anemoi metadata saved successfully")

        except ImportError:
            LOGGER.warning(
                "anemoi.utils.checkpoints.save_metadata not available. "
                "Anemoi metadata will not be saved. Install anemoi-utils to enable metadata support.",
            )
        except (OSError, RuntimeError):
            LOGGER.exception("Failed to save Anemoi metadata for %s", path)
            # Don't fail the entire checkpoint save if metadata saving fails
            LOGGER.warning("Checkpoint saved successfully but metadata saving failed")


def _extract_training_state(
    lightning_checkpoint: dict[str, Any],
    pytorch_checkpoint: dict[str, Any],
    warnings: list[str],
) -> None:
    """Extract training state from Lightning checkpoint.

    Parameters
    ----------
    lightning_checkpoint : dict
        Source Lightning checkpoint
    pytorch_checkpoint : dict
        Target PyTorch checkpoint to populate
    warnings : list
        List to append warnings to
    """
    if "optimizer_states" in lightning_checkpoint:
        pytorch_checkpoint["optimizer_state_dict"] = lightning_checkpoint["optimizer_states"]
    else:
        warnings.append("No 'optimizer_states' found in Lightning checkpoint")

    if "lr_schedulers" in lightning_checkpoint:
        pytorch_checkpoint["scheduler_state_dict"] = lightning_checkpoint["lr_schedulers"]
    else:
        warnings.append("No 'lr_schedulers' found in Lightning checkpoint")

    if "epoch" in lightning_checkpoint:
        pytorch_checkpoint["epoch"] = lightning_checkpoint["epoch"]
    if "global_step" in lightning_checkpoint:
        pytorch_checkpoint["global_step"] = lightning_checkpoint["global_step"]


def convert_lightning_to_pytorch(
    lightning_checkpoint: dict[str, Any],
    extract_model_only: bool = True,
) -> dict[str, Any]:
    """Convert a Lightning checkpoint to PyTorch format.

    Parameters
    ----------
    lightning_checkpoint : dict
        Lightning checkpoint data
    extract_model_only : bool
        If True, extract only model weights. If False, keep optimizer/scheduler state.

    Returns
    -------
    dict
        PyTorch format checkpoint
    """
    if not isinstance(lightning_checkpoint, dict):
        from .exceptions import CheckpointValidationError

        msg = (
            f"Cannot convert checkpoint: expected dictionary, got {type(lightning_checkpoint).__name__}. "
            "Input must be a loaded Lightning checkpoint."
        )
        raise CheckpointValidationError(msg)

    pytorch_checkpoint = {}
    warnings = []

    # Extract model state
    if "state_dict" in lightning_checkpoint:
        pytorch_checkpoint["model_state_dict"] = lightning_checkpoint["state_dict"]
    else:
        warnings.append("No 'state_dict' found in Lightning checkpoint - model weights may be missing")

    if not extract_model_only:
        # Keep training state if requested
        _extract_training_state(lightning_checkpoint, pytorch_checkpoint, warnings)

    # Log warnings about missing components
    if warnings:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Lightning to PyTorch conversion completed with warnings: %s", "; ".join(warnings))

    # Validate that we got at least something useful
    if not pytorch_checkpoint:
        from .exceptions import CheckpointValidationError

        available_keys = list(lightning_checkpoint.keys())[:10]
        msg = (
            "Lightning checkpoint conversion failed: no recognizable Lightning components found.\n"
            f"Available keys: {available_keys}\n"
            "Expected keys: 'state_dict' (required), 'optimizer_states', 'lr_schedulers' (optional)"
        )
        raise CheckpointValidationError(msg)

    return pytorch_checkpoint


def is_format_available(checkpoint_format: Literal["lightning", "pytorch", "safetensors", "state_dict"]) -> bool:
    """Check if a checkpoint format is available for use.

    The ``"lightning"``, ``"pytorch"`` and ``"state_dict"`` formats are
    always available because they only depend on PyTorch.
    ``"safetensors"`` requires the optional ``safetensors`` package.

    Parameters
    ----------
    checkpoint_format : str
        Format to check: "lightning", "pytorch", "safetensors", or "state_dict"

    Returns
    -------
    bool
        True if the format is available
    """
    if checkpoint_format in {"lightning", "pytorch", "state_dict"}:
        return True

    if checkpoint_format == "safetensors":
        try:
            import safetensors  # noqa: F401
        except ImportError:
            return False
        else:
            return True

    return False
