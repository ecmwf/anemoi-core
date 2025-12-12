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

from typing import Any


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize checkpoint error."""
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint file cannot be found."""

    def __init__(self, path: Any, details: dict[str, Any] | None = None):
        """Initialize checkpoint not found error."""
        from pathlib import Path

        path = Path(path) if not isinstance(path, Path) else path
        message = f"Checkpoint not found: {path}"

        # Add helpful suggestions
        suggestions = []
        if path.parent.exists():
            # Look for similar files in the directory
            similar_files = [f for f in path.parent.glob("*") if f.suffix in [".ckpt", ".pt", ".pth", ".bin"]]
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
    """Raised when checkpoint loading fails."""

    def __init__(
        self,
        path: Any,
        original_error: Exception,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint load error."""
        from pathlib import Path

        path = Path(path) if not isinstance(path, Path) else path
        message = f"Failed to load checkpoint from: {path}. Error: {original_error}"

        error_details = {
            "path": str(path),
            "original_error": str(original_error),
            "error_type": type(original_error).__name__,
        }
        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.path = path
        self.original_error = original_error


class CheckpointIncompatibleError(CheckpointError):
    """Raised when checkpoint is incompatible with model."""

    def __init__(
        self,
        message: str,
        missing_keys: list[str] | None = None,
        unexpected_keys: list[str] | None = None,
        shape_mismatches: dict[str, tuple] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint incompatible error."""
        detailed_message = self._build_error_message(
            message,
            missing_keys,
            unexpected_keys,
            shape_mismatches,
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

    def _build_error_message(
        self,
        message: str,
        missing_keys: list[str] | None,
        unexpected_keys: list[str] | None,
        shape_mismatches: dict[str, tuple] | None,
    ) -> str:
        """Build detailed error message."""
        detailed_message = message

        if missing_keys:
            detailed_message += f"\nMissing keys: {missing_keys[:5]}"
            if len(missing_keys) > 5:
                detailed_message += f" ... and {len(missing_keys) - 5} more"

        if unexpected_keys:
            detailed_message += f"\nUnexpected keys: {unexpected_keys[:5]}"
            if len(unexpected_keys) > 5:
                detailed_message += f" ... and {len(unexpected_keys) - 5} more"

        if shape_mismatches:
            detailed_message += f"\nShape mismatches: {len(shape_mismatches)} keys"

        return detailed_message

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


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint validation fails."""

    def __init__(
        self,
        message: str,
        validation_errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint validation error."""
        error_details = {}

        if validation_errors:
            error_details["validation_errors"] = validation_errors
            error_details["num_errors"] = len(validation_errors)

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.validation_errors = validation_errors or []


class CheckpointConfigError(CheckpointError):
    """Raised when checkpoint configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint configuration error."""
        error_details = {}

        if config_path:
            error_details["config_path"] = config_path
            message = f"{message} (at {config_path})"

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.config_path = config_path


class CheckpointSourceError(CheckpointError):
    """Raised when checkpoint source operations fail.

    This exception is raised when fetching a checkpoint from a
    source (S3, HTTP, etc.) fails due to network issues, authentication
    problems, or source unavailability.
    """

    def __init__(
        self,
        message: str,
        source_path: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize checkpoint source error."""
        error_details = {"source_path": source_path}

        if original_error:
            error_details["original_error"] = str(original_error)

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.source_path = source_path
        self.original_error = original_error


class CheckpointTimeoutError(CheckpointError):
    """Raised when checkpoint operation times out.

    This exception is raised when a checkpoint operation exceeds
    the configured timeout duration.
    """

    def __init__(self, operation: str, timeout: float, details: dict[str, Any] | None = None):
        """Initialize checkpoint timeout error."""
        message = f"Checkpoint operation timed out after {timeout}s: {operation}"

        error_details = {"operation": operation, "timeout_seconds": timeout}

        if details:
            error_details.update(details)

        super().__init__(message, error_details)
        self.operation = operation
        self.timeout = timeout
