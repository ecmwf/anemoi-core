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
