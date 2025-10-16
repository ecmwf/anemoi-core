# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Exception classes for checkpoint operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint operations.

    All checkpoint-related exceptions should inherit from this class
    to allow for unified error handling in the pipeline.

    Parameters
    ----------
    message : str
        Error message describing the issue
    details : dict, optional
        Additional error details for debugging
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize checkpoint error.

        Parameters
        ----------
        message : str
            Error message describing the issue
        details : dict, optional
            Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

        # Log the error with details
        if self.details:
            LOGGER.error(
                "%s: %s. Details: %s",
                self.__class__.__name__,
                message,
                self.details,
            )
        else:
            LOGGER.error("%s: %s", self.__class__.__name__, message)

    def __str__(self) -> str:
        """String representation of the error.

        Returns
        -------
        str
            Error message with optional details
        """
        # Handle None message gracefully
        message = str(self.message) if self.message is not None else "Unknown error"

        if self.details:
            return f"{message} (Details: {self.details})"
        return message

    def __reduce__(self) -> tuple:
        """Support for pickling.

        Returns
        -------
        tuple
            Tuple of (constructor, args) for pickle reconstruction
        """
        return (self.__class__, (self.message, self.details))


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint file cannot be found.

    This exception is raised when a specified checkpoint path
    does not exist or cannot be accessed.

    Parameters
    ----------
    path : str or Path
        Path to the missing checkpoint
    details : dict, optional
        Additional error details
    """

    def __init__(self, path: Any, details: dict[str, Any] | None = None):
        """Initialize checkpoint not found error.

        Parameters
        ----------
        path : str or Path
            Path to the missing checkpoint
        details : dict, optional
            Additional error details
        """
        path = Path(path) if not isinstance(path, Path) else path

        # Create helpful error message with suggestions
        message = f"Checkpoint not found: {path}"

        # Add helpful suggestions
        suggestions = []
        if path.parent.exists():
            # Look for similar files in the directory
            similar_files = [
                f for f in path.parent.glob("*") if f.suffix in [".ckpt", ".pt", ".pth", ".bin", ".safetensors"]
            ]
            if similar_files:
                suggestions.append(
                    f"Found similar files in {path.parent}: {[f.name for f in similar_files[:3]]}",
                )
        else:
            suggestions.append(f"Directory does not exist: {path.parent}")

        if suggestions:
            message += "\nSuggestions:\n  • " + "\n  • ".join(suggestions)

        # Add context about which pipeline stage failed if available
        stage_context = details.get("pipeline_stage") if details else None
        if stage_context:
            message = f"Pipeline stage '{stage_context}' failed: {message}"

        error_details = {"path": str(path)}
        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.path = path


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint loading fails.

    This exception is raised when a checkpoint file exists but
    cannot be loaded due to corruption, format issues, or I/O errors.

    Parameters
    ----------
    path : str or Path
        Path to the checkpoint that failed to load
    original_error : Exception
        The original exception that caused the load failure
    details : dict, optional
        Additional error details
    """

    def __init__(
        self,
        path: Any,
        original_error: Exception,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint load error.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint that failed to load
        original_error : Exception
            The original exception that caused the load failure
        details : dict, optional
            Additional error details
        """
        path = Path(path) if not isinstance(path, Path) else path

        # Create helpful error message based on error type
        error_type = type(original_error).__name__
        base_message = f"Failed to load checkpoint from: {path}"

        # Add specific guidance based on error type
        if "UnpicklingError" in error_type or "EOFError" in error_type:
            suggestion = "The checkpoint file appears to be corrupted. Try re-downloading or using a backup."
        elif "FileNotFoundError" in error_type:
            suggestion = "The checkpoint file was not found. Check that the path is correct and the file exists."
        elif "PermissionError" in error_type:
            suggestion = "Permission denied accessing the checkpoint file. Check file permissions."
        elif "OutOfMemoryError" in error_type or "CUDA out of memory" in str(
            original_error,
        ):
            suggestion = (
                "Not enough memory to load checkpoint. Try loading on CPU first or use a machine with more RAM."
            )
        else:
            suggestion = f"Original error: {original_error}"

        message = f"{base_message}\nError: {suggestion}"

        # Add context about which pipeline stage failed if available
        stage_context = details.get("pipeline_stage") if details else None
        if stage_context:
            message = f"Pipeline stage '{stage_context}' failed: {message}"

        error_details = {
            "path": str(path),
            "original_error": str(original_error),
            "error_type": error_type,
        }
        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.path = path
        self.original_error = original_error

    def __reduce__(self) -> tuple:
        """Support for pickling.

        Returns
        -------
        tuple
            Tuple of (constructor, args) for pickle reconstruction
        """
        # Reconstruct with original constructor arguments
        details_without_auto = {
            k: v for k, v in self.details.items() if k not in ("path", "original_error", "error_type")
        }
        return (
            self.__class__,
            (self.path, self.original_error, details_without_auto or None),
        )


class CheckpointIncompatibleError(CheckpointError):
    """Raised when checkpoint is incompatible with model.

    This exception is raised when a checkpoint cannot be applied to
    a model due to architectural differences, shape mismatches, or
    missing/unexpected keys.

    Parameters
    ----------
    message : str
        Description of the incompatibility
    missing_keys : list, optional
        List of keys missing in checkpoint
    unexpected_keys : list, optional
        List of unexpected keys in checkpoint
    shape_mismatches : dict, optional
        Dictionary of shape mismatches {key: (expected, actual)}
    details : dict, optional
        Additional error details
    """

    def _format_missing_keys(
        self,
        missing_keys: list[str],
        suggestions: list[str],
    ) -> str:
        """Format missing keys section of error message."""
        msg = f"\nMissing keys in checkpoint ({len(missing_keys)}): {missing_keys[:5]}"
        if len(missing_keys) > 5:
            msg += f" ... and {len(missing_keys) - 5} more"
        suggestions.extend(
            (
                "Try using 'strict=False' in your loader configuration",
                "Check if you're using the right checkpoint for this model architecture",
            ),
        )
        return msg

    def _format_unexpected_keys(
        self,
        unexpected_keys: list[str],
        suggestions: list[str],
    ) -> str:
        """Format unexpected keys section of error message."""
        msg = f"\nUnexpected keys in checkpoint ({len(unexpected_keys)}): {unexpected_keys[:5]}"
        if len(unexpected_keys) > 5:
            msg += f" ... and {len(unexpected_keys) - 5} more"
        suggestions.append("Try using 'strict=False' in your loader configuration")
        return msg

    def _format_shape_mismatches(
        self,
        shape_mismatches: dict[str, tuple],
        suggestions: list[str],
    ) -> str:
        """Format shape mismatches section of error message."""
        msg = f"\nShape mismatches ({len(shape_mismatches)}):"
        for key, (expected, actual) in list(shape_mismatches.items())[:3]:
            msg += f"\n  • {key}: expected {expected}, got {actual}"
        if len(shape_mismatches) > 3:
            msg += f"\n  ... and {len(shape_mismatches) - 3} more mismatches"
        suggestions.extend(
            (
                "Model architecture may not match the checkpoint",
                "Consider using transfer learning loader instead",
            ),
        )
        return msg

    def _build_message(
        self,
        message: str,
        missing_keys: list[str] | None,
        unexpected_keys: list[str] | None,
        shape_mismatches: dict[str, tuple] | None,
        details: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Build detailed error message with incompatibility information."""
        detailed_message = message
        suggestions: list[str] = []

        if missing_keys:
            detailed_message += self._format_missing_keys(missing_keys, suggestions)
        if unexpected_keys:
            detailed_message += self._format_unexpected_keys(unexpected_keys, suggestions)
        if shape_mismatches:
            detailed_message += self._format_shape_mismatches(shape_mismatches, suggestions)

        # Add suggestions
        if suggestions:
            detailed_message += "\nSuggestions:"
            for suggestion in suggestions[:3]:
                detailed_message += f"\n  • {suggestion}"

        # Add pipeline stage context
        stage_context = details.get("pipeline_stage") if details else None
        if stage_context:
            detailed_message = f"Pipeline stage '{stage_context}' failed: {detailed_message}"

        return detailed_message, suggestions

    def _build_error_details(
        self,
        missing_keys: list[str] | None,
        unexpected_keys: list[str] | None,
        shape_mismatches: dict[str, tuple] | None,
        details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build error details dictionary."""
        error_details = {}

        if missing_keys:
            error_details["missing_keys"] = missing_keys
            error_details["num_missing"] = len(missing_keys)
        if unexpected_keys:
            error_details["unexpected_keys"] = unexpected_keys
            error_details["num_unexpected"] = len(unexpected_keys)
        if shape_mismatches:
            error_details["shape_mismatches"] = shape_mismatches
            error_details["num_mismatches"] = len(shape_mismatches)
        if details:
            error_details.update(details)

        return error_details

    def __init__(
        self,
        message: str,
        missing_keys: list[str] | None = None,
        unexpected_keys: list[str] | None = None,
        shape_mismatches: dict[str, tuple] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint incompatible error.

        Parameters
        ----------
        message : str
            Description of the incompatibility
        missing_keys : list, optional
            List of keys missing in checkpoint
        unexpected_keys : list, optional
            List of unexpected keys in checkpoint
        shape_mismatches : dict, optional
            Dictionary of shape mismatches {key: (expected, actual)}
        details : dict, optional
            Additional error details
        """
        detailed_message, _ = self._build_message(
            message,
            missing_keys,
            unexpected_keys,
            shape_mismatches,
            details,
        )
        error_details = self._build_error_details(
            missing_keys,
            unexpected_keys,
            shape_mismatches,
            details,
        )

        super().__init__(detailed_message, error_details)
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []
        self.shape_mismatches = shape_mismatches or {}


class CheckpointSourceError(CheckpointError):
    """Raised when checkpoint source operations fail.

    This exception is raised when fetching a checkpoint from a
    source (S3, HTTP, etc.) fails due to network issues, authentication
    problems, or source unavailability.

    Parameters
    ----------
    source_type : str
        Type of checkpoint source (e.g., 's3', 'http', 'local')
    source_path : str
        Source path or URL that failed
    original_error : Exception, optional
        The original exception if available
    details : dict, optional
        Additional error details
    """

    def __init__(
        self,
        source_type: str,
        source_path: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint source error.

        Parameters
        ----------
        source_type : str
            Type of checkpoint source
        source_path : str
            Source path or URL that failed
        original_error : Exception, optional
            The original exception if available
        details : dict, optional
            Additional error details
        """
        message = f"Failed to fetch checkpoint from {source_type} source: {source_path}"
        if original_error:
            message += f". {original_error!s}"

        error_details = {"source_type": source_type, "source_path": source_path}

        if original_error:
            error_details["original_error"] = str(original_error)
            error_details["error_type"] = type(original_error).__name__

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.source_type = source_type
        self.source_path = source_path
        self.original_error = original_error


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint validation fails.

    This exception is raised when a checkpoint does not meet
    expected validation criteria such as required keys, format
    version, or integrity checks.

    Parameters
    ----------
    message : str
        Description of validation failure
    validation_errors : list, optional
        List of specific validation errors
    details : dict, optional
        Additional error details
    """

    def __init__(
        self,
        message: str,
        validation_errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint validation error.

        Parameters
        ----------
        message : str
            Description of validation failure
        validation_errors : list, optional
            List of specific validation errors
        details : dict, optional
            Additional error details
        """
        error_details = {}

        if validation_errors:
            error_details["validation_errors"] = validation_errors
            error_details["num_errors"] = len(validation_errors)

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.validation_errors = validation_errors or []


class CheckpointTimeoutError(CheckpointError):
    """Raised when checkpoint operation times out.

    This exception is raised when a checkpoint operation exceeds
    the configured timeout duration.

    Parameters
    ----------
    operation : str
        Description of the operation that timed out
    timeout : float
        Timeout duration in seconds
    details : dict, optional
        Additional error details
    """

    def __init__(
        self,
        operation: str,
        timeout: float,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint timeout error.

        Parameters
        ----------
        operation : str
            Description of the operation that timed out
        timeout : float
            Timeout duration in seconds
        details : dict, optional
            Additional error details
        """
        message = f"Checkpoint operation timed out after {timeout}s: {operation}"

        error_details = {"operation": operation, "timeout_seconds": timeout}

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.operation = operation
        self.timeout = timeout


class CheckpointConfigError(CheckpointError):
    """Raised when checkpoint configuration is invalid.

    This exception is raised when the checkpoint configuration
    contains invalid values, missing required fields, or
    conflicting settings.

    Parameters
    ----------
    message : str
        Description of configuration error
    config_path : str, optional
        Path to the problematic configuration field
    details : dict, optional
        Additional error details
    """

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint configuration error.

        Parameters
        ----------
        message : str
            Description of configuration error
        config_path : str, optional
            Path to the problematic configuration field
        details : dict, optional
            Additional error details
        """
        error_details = {}

        if config_path:
            error_details["config_path"] = config_path
            message = f"{message} (at {config_path})"

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.config_path = config_path
