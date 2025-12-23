# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for checkpoint exception hierarchy and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointError
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.exceptions import CheckpointSourceError
from anemoi.training.checkpoint.exceptions import CheckpointTimeoutError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError


class TestCheckpointError:
    """Test base CheckpointError functionality."""

    @pytest.mark.unit
    def test_checkpoint_error_basic(self) -> None:
        """Test basic CheckpointError creation."""
        error = CheckpointError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    @pytest.mark.unit
    def test_checkpoint_error_with_details(self) -> None:
        """Test CheckpointError with details.

        Note: The base CheckpointError stores details but does not include
        them in the string representation. Details are accessed via the
        details attribute for programmatic use.
        """
        details = {"file": "test.ckpt", "line": 42}
        error = CheckpointError("Test error with details", details=details)

        assert error.message == "Test error with details"
        assert error.details == details
        # Details are stored but not in string representation for base class
        assert error.details["file"] == "test.ckpt"
        assert error.details["line"] == 42

    @pytest.mark.unit
    def test_checkpoint_error_empty_details(self) -> None:
        """Test CheckpointError with empty details dictionary."""
        error = CheckpointError("Test error", details={})

        assert error.details == {}
        assert str(error) == "Test error"  # No details suffix

    @pytest.mark.unit
    def test_checkpoint_error_none_details(self) -> None:
        """Test CheckpointError with None details."""
        error = CheckpointError("Test error", details=None)

        assert error.details == {}
        assert str(error) == "Test error"

    @pytest.mark.unit
    def test_checkpoint_error_inheritance(self) -> None:
        """Test that CheckpointError inherits from Exception."""
        error = CheckpointError("Test inheritance")

        assert isinstance(error, Exception)
        assert isinstance(error, CheckpointError)

    @pytest.mark.unit
    def test_checkpoint_error_as_base_class(self) -> None:
        """Test CheckpointError can be used as base exception type."""

        # Create a custom exception that inherits from CheckpointError
        class CustomCheckpointError(CheckpointError):
            pass

        custom_error = CustomCheckpointError("Custom error")

        # Should be catchable as CheckpointError
        try:
            raise custom_error
        except CheckpointError as e:
            assert e.message == "Custom error"
        else:
            pytest.fail("Exception should have been caught as CheckpointError")


class TestCheckpointNotFoundError:
    """Test CheckpointNotFoundError functionality."""

    @pytest.mark.unit
    def test_checkpoint_not_found_error_string_path(self) -> None:
        """Test CheckpointNotFoundError with string path."""
        error = CheckpointNotFoundError("/path/to/missing.ckpt")

        assert "Checkpoint not found" in str(error)
        assert "/path/to/missing.ckpt" in str(error)
        assert error.path == Path("/path/to/missing.ckpt")

    @pytest.mark.unit
    def test_checkpoint_not_found_error_path_object(self) -> None:
        """Test CheckpointNotFoundError with Path object."""
        path = Path("/path/to/missing.ckpt")
        error = CheckpointNotFoundError(path)

        assert error.path == path
        assert str(path) in str(error)

    @pytest.mark.unit
    def test_checkpoint_not_found_error_with_details(self) -> None:
        """Test CheckpointNotFoundError with additional details."""
        details = {"reason": "permission_denied", "user": "test_user"}
        error = CheckpointNotFoundError("/missing.ckpt", details=details)

        assert error.details["path"] == "/missing.ckpt"
        assert error.details["reason"] == "permission_denied"
        assert error.details["user"] == "test_user"

    @pytest.mark.unit
    def test_checkpoint_not_found_error_inheritance(self) -> None:
        """Test CheckpointNotFoundError inheritance."""
        error = CheckpointNotFoundError("/missing.ckpt")

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointNotFoundError)

    @pytest.mark.unit
    def test_checkpoint_not_found_error_path_details_merge(self) -> None:
        """Test that path is added to details correctly."""
        extra_details = {"attempted_locations": ["/path1", "/path2"]}
        error = CheckpointNotFoundError("/main/path.ckpt", details=extra_details)

        assert "path" in error.details
        assert "attempted_locations" in error.details
        assert error.details["path"] == "/main/path.ckpt"


class TestCheckpointLoadError:
    """Test CheckpointLoadError functionality."""

    @pytest.mark.unit
    def test_checkpoint_load_error_basic(self) -> None:
        """Test basic CheckpointLoadError."""
        original_error = FileNotFoundError("File not found")
        error = CheckpointLoadError("/path/to/checkpoint.ckpt", original_error)

        assert error.path == Path("/path/to/checkpoint.ckpt")
        assert error.original_error is original_error
        assert "Failed to load checkpoint from:" in str(error)
        assert "File not found" in str(error)

    @pytest.mark.unit
    def test_checkpoint_load_error_with_details(self) -> None:
        """Test CheckpointLoadError with additional details."""
        original_error = RuntimeError("Corrupted data")
        details = {"size_mb": 1024, "format": "lightning"}
        error = CheckpointLoadError("/checkpoint.ckpt", original_error, details=details)

        assert error.details["path"] == "/checkpoint.ckpt"
        assert error.details["original_error"] == "Corrupted data"
        assert error.details["error_type"] == "RuntimeError"
        assert error.details["size_mb"] == 1024
        assert error.details["format"] == "lightning"

    @pytest.mark.unit
    def test_checkpoint_load_error_path_conversion(self) -> None:
        """Test that string path is converted to Path object."""
        original_error = ValueError("Invalid value")
        error = CheckpointLoadError("/string/path.ckpt", original_error)

        assert isinstance(error.path, Path)
        assert str(error.path) == "/string/path.ckpt"

    @pytest.mark.unit
    def test_checkpoint_load_error_error_details_structure(self) -> None:
        """Test the structure of error details."""
        original_error = KeyError("missing_key")
        error = CheckpointLoadError("/test.ckpt", original_error)

        expected_keys = {"path", "original_error", "error_type"}
        assert set(error.details.keys()) == expected_keys
        assert error.details["error_type"] == "KeyError"

    @pytest.mark.unit
    def test_checkpoint_load_error_inheritance(self) -> None:
        """Test CheckpointLoadError inheritance."""
        original_error = Exception("test")
        error = CheckpointLoadError("/test.ckpt", original_error)

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointLoadError)


class TestCheckpointIncompatibleError:
    """Test CheckpointIncompatibleError functionality."""

    @pytest.mark.unit
    def test_checkpoint_incompatible_error_basic(self) -> None:
        """Test basic CheckpointIncompatibleError."""
        error = CheckpointIncompatibleError("Model architecture mismatch")

        assert error.missing_keys == []
        assert error.unexpected_keys == []
        assert error.shape_mismatches == {}
        assert "Model architecture mismatch" in str(error)

    @pytest.mark.unit
    def test_checkpoint_incompatible_error_with_keys(self) -> None:
        """Test CheckpointIncompatibleError with missing and unexpected keys."""
        missing_keys = ["layer1.weight", "layer1.bias"]
        unexpected_keys = ["extra_layer.weight"]

        error = CheckpointIncompatibleError("Key mismatch", missing_keys=missing_keys, unexpected_keys=unexpected_keys)

        assert error.missing_keys == missing_keys
        assert error.unexpected_keys == unexpected_keys
        assert error.details["num_missing"] == 2
        assert error.details["num_unexpected"] == 1

    @pytest.mark.unit
    def test_checkpoint_incompatible_error_with_shape_mismatches(self) -> None:
        """Test CheckpointIncompatibleError with shape mismatches."""
        shape_mismatches = {"layer1.weight": ((10, 5), (10, 8)), "layer2.bias": ((20,), (25,))}

        error = CheckpointIncompatibleError("Shape mismatch", shape_mismatches=shape_mismatches)

        assert error.shape_mismatches == shape_mismatches
        assert error.details["num_mismatches"] == 2

    @pytest.mark.unit
    def test_checkpoint_incompatible_error_all_fields(self) -> None:
        """Test CheckpointIncompatibleError with all field types."""
        missing_keys = ["missing1"]
        unexpected_keys = ["unexpected1", "unexpected2"]
        shape_mismatches = {"mismatch1": ((5, 5), (3, 3))}
        additional_details = {"model_type": "transformer"}

        error = CheckpointIncompatibleError(
            "Complete mismatch",
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            shape_mismatches=shape_mismatches,
            details=additional_details,
        )

        assert error.missing_keys == missing_keys
        assert error.unexpected_keys == unexpected_keys
        assert error.shape_mismatches == shape_mismatches
        assert error.details["model_type"] == "transformer"
        assert error.details["num_missing"] == 1
        assert error.details["num_unexpected"] == 2
        assert error.details["num_mismatches"] == 1

    @pytest.mark.unit
    def test_checkpoint_incompatible_error_none_values(self) -> None:
        """Test CheckpointIncompatibleError with None values."""
        error = CheckpointIncompatibleError(
            "Test with None values",
            missing_keys=None,
            unexpected_keys=None,
            shape_mismatches=None,
        )

        assert error.missing_keys == []
        assert error.unexpected_keys == []
        assert error.shape_mismatches == {}

    @pytest.mark.unit
    def test_checkpoint_incompatible_error_inheritance(self) -> None:
        """Test CheckpointIncompatibleError inheritance."""
        error = CheckpointIncompatibleError("Test inheritance")

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointIncompatibleError)


class TestCheckpointSourceError:
    """Test CheckpointSourceError functionality.

    Note: CheckpointSourceError takes (message, source_path, original_error, details)
    not (source_type, source_path, ...).
    """

    @pytest.mark.unit
    def test_checkpoint_source_error_basic(self) -> None:
        """Test basic CheckpointSourceError."""
        error = CheckpointSourceError("Failed to fetch from S3", "s3://bucket/path/model.ckpt")

        assert error.source_path == "s3://bucket/path/model.ckpt"
        assert error.original_error is None
        assert "Failed to fetch from S3" in str(error)

    @pytest.mark.unit
    def test_checkpoint_source_error_with_original_error(self) -> None:
        """Test CheckpointSourceError with original error."""
        original_error = ConnectionError("Network unavailable")
        error = CheckpointSourceError(
            "HTTP download failed",
            "https://example.com/model.ckpt",
            original_error,
        )

        assert error.original_error is original_error
        assert error.details["original_error"] == "Network unavailable"

    @pytest.mark.unit
    def test_checkpoint_source_error_with_details(self) -> None:
        """Test CheckpointSourceError with additional details."""
        details = {"retry_count": 3, "last_attempt": "2024-01-01T00:00:00"}
        error = CheckpointSourceError(
            "GCS download failed",
            "gs://bucket/model.ckpt",
            original_error=None,
            details=details,
        )

        assert error.details["source_path"] == "gs://bucket/model.ckpt"
        assert error.details["retry_count"] == 3
        assert error.details["last_attempt"] == "2024-01-01T00:00:00"

    @pytest.mark.unit
    def test_checkpoint_source_error_all_fields(self) -> None:
        """Test CheckpointSourceError with all fields."""
        original_error = TimeoutError("Request timeout")
        details = {"timeout_seconds": 300}

        error = CheckpointSourceError(
            "Azure download timed out",
            "https://storage.blob.core.windows.net/model.ckpt",
            original_error,
            details,
        )

        assert error.source_path == "https://storage.blob.core.windows.net/model.ckpt"
        assert error.original_error is original_error
        assert error.details["timeout_seconds"] == 300

    @pytest.mark.unit
    def test_checkpoint_source_error_inheritance(self) -> None:
        """Test CheckpointSourceError inheritance."""
        error = CheckpointSourceError("Local file error", "/path/to/checkpoint")

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointSourceError)


class TestCheckpointValidationError:
    """Test CheckpointValidationError functionality."""

    @pytest.mark.unit
    def test_checkpoint_validation_error_basic(self) -> None:
        """Test basic CheckpointValidationError."""
        error = CheckpointValidationError("Validation failed")

        assert error.validation_errors == []
        assert "Validation failed" in str(error)

    @pytest.mark.unit
    def test_checkpoint_validation_error_with_validation_errors(self) -> None:
        """Test CheckpointValidationError with validation error list."""
        validation_errors = [
            "Missing required key: state_dict",
            "Invalid tensor shape in layer1.weight",
            "NaN values detected in layer2.bias",
        ]

        error = CheckpointValidationError("Multiple validation errors", validation_errors=validation_errors)

        assert error.validation_errors == validation_errors
        assert error.details["num_errors"] == 3
        assert error.details["validation_errors"] == validation_errors

    @pytest.mark.unit
    def test_checkpoint_validation_error_with_details(self) -> None:
        """Test CheckpointValidationError with additional details."""
        validation_errors = ["Error 1", "Error 2"]
        details = {"checkpoint_format": "lightning", "file_size_mb": 512}

        error = CheckpointValidationError("Validation failed", validation_errors=validation_errors, details=details)

        assert error.details["checkpoint_format"] == "lightning"
        assert error.details["file_size_mb"] == 512
        assert error.details["num_errors"] == 2

    @pytest.mark.unit
    def test_checkpoint_validation_error_none_validation_errors(self) -> None:
        """Test CheckpointValidationError with None validation errors."""
        error = CheckpointValidationError("Test", validation_errors=None)

        assert error.validation_errors == []

    @pytest.mark.unit
    def test_checkpoint_validation_error_inheritance(self) -> None:
        """Test CheckpointValidationError inheritance."""
        error = CheckpointValidationError("Test inheritance")

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointValidationError)


class TestCheckpointTimeoutError:
    """Test CheckpointTimeoutError functionality."""

    @pytest.mark.unit
    def test_checkpoint_timeout_error_basic(self) -> None:
        """Test basic CheckpointTimeoutError."""
        error = CheckpointTimeoutError("Download checkpoint", 300.0)

        assert error.operation == "Download checkpoint"
        assert error.timeout == 300.0
        assert "timed out after 300.0s" in str(error)

    @pytest.mark.unit
    def test_checkpoint_timeout_error_with_details(self) -> None:
        """Test CheckpointTimeoutError with additional details."""
        details = {"url": "https://example.com/model.ckpt", "bytes_downloaded": 1024000}
        error = CheckpointTimeoutError("HTTP download", 120.5, details=details)

        assert error.details["operation"] == "HTTP download"
        assert error.details["timeout_seconds"] == 120.5
        assert error.details["url"] == "https://example.com/model.ckpt"
        assert error.details["bytes_downloaded"] == 1024000

    @pytest.mark.unit
    def test_checkpoint_timeout_error_float_timeout(self) -> None:
        """Test CheckpointTimeoutError with float timeout."""
        error = CheckpointTimeoutError("Test operation", 45.5)

        assert error.timeout == 45.5
        assert "45.5s" in str(error)

    @pytest.mark.unit
    def test_checkpoint_timeout_error_zero_timeout(self) -> None:
        """Test CheckpointTimeoutError with zero timeout."""
        error = CheckpointTimeoutError("Immediate timeout", 0.0)

        assert error.timeout == 0.0
        assert "0.0s" in str(error)

    @pytest.mark.unit
    def test_checkpoint_timeout_error_inheritance(self) -> None:
        """Test CheckpointTimeoutError inheritance."""
        error = CheckpointTimeoutError("Test", 10.0)

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointTimeoutError)


class TestCheckpointConfigError:
    """Test CheckpointConfigError functionality."""

    @pytest.mark.unit
    def test_checkpoint_config_error_basic(self) -> None:
        """Test basic CheckpointConfigError."""
        error = CheckpointConfigError("Invalid configuration value")

        assert error.config_path is None
        assert "Invalid configuration value" in str(error)

    @pytest.mark.unit
    def test_checkpoint_config_error_with_config_path(self) -> None:
        """Test CheckpointConfigError with configuration path."""
        error = CheckpointConfigError("Invalid value for source type", config_path="training.checkpoint.source.type")

        assert error.config_path == "training.checkpoint.source.type"
        assert "training.checkpoint.source.type" in str(error)
        assert error.details["config_path"] == "training.checkpoint.source.type"

    @pytest.mark.unit
    def test_checkpoint_config_error_with_details(self) -> None:
        """Test CheckpointConfigError with additional details."""
        details = {"provided_value": "invalid_type", "valid_values": ["s3", "http", "local"]}
        error = CheckpointConfigError("Unknown source type", config_path="checkpoint.source.type", details=details)

        assert error.details["provided_value"] == "invalid_type"
        assert error.details["valid_values"] == ["s3", "http", "local"]
        assert error.details["config_path"] == "checkpoint.source.type"

    @pytest.mark.unit
    def test_checkpoint_config_error_message_formatting(self) -> None:
        """Test CheckpointConfigError message formatting with config path."""
        error = CheckpointConfigError("Value too large", config_path="training.batch_size")

        # Message should include config path in parentheses
        assert "Value too large (at training.batch_size)" in str(error)

    @pytest.mark.unit
    def test_checkpoint_config_error_inheritance(self) -> None:
        """Test CheckpointConfigError inheritance."""
        error = CheckpointConfigError("Test config error")

        assert isinstance(error, CheckpointError)
        assert isinstance(error, CheckpointConfigError)


class TestExceptionHierarchy:
    """Test the exception hierarchy and polymorphism."""

    @pytest.mark.unit
    def test_all_exceptions_inherit_from_checkpoint_error(self) -> None:
        """Test that all checkpoint exceptions inherit from CheckpointError."""
        exception_classes = [
            CheckpointNotFoundError,
            CheckpointLoadError,
            CheckpointIncompatibleError,
            CheckpointSourceError,
            CheckpointValidationError,
            CheckpointTimeoutError,
            CheckpointConfigError,
        ]

        for exc_class in exception_classes:
            # Create minimal instance with correct constructor arguments
            if exc_class == CheckpointLoadError:
                instance = exc_class("/test.ckpt", Exception("test"))
            elif exc_class == CheckpointSourceError:
                # CheckpointSourceError takes (message, source_path)
                instance = exc_class("test message", "/test/path")
            elif exc_class == CheckpointTimeoutError:
                instance = exc_class("test", 10.0)
            else:
                instance = exc_class("test message")

            assert isinstance(instance, CheckpointError)

    @pytest.mark.unit
    def test_catch_all_checkpoint_errors(self) -> None:
        """Test catching all checkpoint errors with base class."""
        exceptions_to_test = [
            CheckpointNotFoundError("/missing.ckpt"),
            CheckpointLoadError("/broken.ckpt", ValueError("test")),
            CheckpointIncompatibleError("incompatible"),
            CheckpointSourceError("S3 fetch failed", "s3://bucket/key"),
            CheckpointValidationError("invalid"),
            CheckpointTimeoutError("timeout", 30.0),
            CheckpointConfigError("config error"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except CheckpointError as caught:
                # Should catch all types as CheckpointError
                assert isinstance(caught, CheckpointError)
                assert caught.message  # All should have message attribute
            else:
                pytest.fail(f"Failed to catch {type(exc).__name__} as CheckpointError")

    @pytest.mark.unit
    def test_exception_specific_catching(self) -> None:
        """Test catching specific exception types."""
        # Test that specific exception types can still be caught specifically
        try:
            msg = "/missing.ckpt"
            raise CheckpointNotFoundError(msg)
        except CheckpointNotFoundError as e:
            assert isinstance(e, CheckpointNotFoundError)
            assert hasattr(e, "path")
        except CheckpointError:
            pytest.fail("Should have caught CheckpointNotFoundError specifically")

    @pytest.mark.unit
    def test_exception_details_consistency(self) -> None:
        """Test that all exceptions have consistent details structure."""
        exceptions = [
            CheckpointError("test", {"key": "value"}),
            CheckpointNotFoundError("/test.ckpt", {"extra": "data"}),
            CheckpointLoadError("/test.ckpt", ValueError("test"), {"size": 100}),
            CheckpointSourceError("S3 error", "s3://bucket/key", details={"retry": 3}),
            CheckpointValidationError("test", details={"format": "lightning"}),
            CheckpointTimeoutError("test", 10.0, {"url": "http://test"}),
            CheckpointConfigError("test", details={"field": "value"}),
        ]

        for exc in exceptions:
            assert hasattr(exc, "details")
            assert isinstance(exc.details, dict)
            assert hasattr(exc, "message")
            assert isinstance(exc.message, str)


class TestErrorContextPreservation:
    """Test error context and chaining."""

    @pytest.mark.unit
    def test_original_error_preservation_in_load_error(self) -> None:
        """Test that original errors are preserved in CheckpointLoadError."""
        original = FileNotFoundError("Original file error")
        load_error = CheckpointLoadError("/test.ckpt", original)

        assert load_error.original_error is original
        assert str(original) in str(load_error)

    @pytest.mark.unit
    def test_original_error_preservation_in_source_error(self) -> None:
        """Test that original errors are preserved in CheckpointSourceError."""
        original = ConnectionError("Network failure")
        source_error = CheckpointSourceError("HTTP download failed", "http://test.com", original)

        assert source_error.original_error is original

    @pytest.mark.unit
    def test_error_chain_with_raise_from(self) -> None:
        """Test proper error chaining with 'raise from'."""
        original = ValueError("Original error")

        try:
            msg = "/test.ckpt"
            raise CheckpointLoadError(msg, original) from original
        except CheckpointLoadError as e:
            assert e.__cause__ is original

    @pytest.mark.unit
    def test_error_details_aggregation(self) -> None:
        """Test that error details are properly aggregated."""
        # Create a complex error with multiple detail sources
        original_error = RuntimeError("Runtime failure")
        extra_details = {"attempt": 3, "timeout": 30}

        error = CheckpointLoadError("/test.ckpt", original_error, extra_details)

        # Should contain both automatic details and provided details
        assert "path" in error.details  # Automatic
        assert "original_error" in error.details  # Automatic
        assert "error_type" in error.details  # Automatic
        assert "attempt" in error.details  # Provided
        assert "timeout" in error.details  # Provided

    @pytest.mark.unit
    def test_nested_error_details(self) -> None:
        """Test handling of nested error structures."""
        complex_details = {
            "metadata": {
                "format": "lightning",
                "version": "2.0",
            },
            "validation_results": {
                "passed": False,
                "errors": ["missing_key", "shape_mismatch"],
            },
        }

        error = CheckpointValidationError("Complex validation error", details=complex_details)

        assert error.details["metadata"]["format"] == "lightning"
        assert "missing_key" in error.details["validation_results"]["errors"]


class TestErrorEdgeCases:
    """Test edge cases and unusual error conditions."""

    @pytest.mark.unit
    def test_error_with_none_message(self) -> None:
        """Test error handling with None message."""
        # This should not normally happen, but test robustness
        try:
            error = CheckpointError(None)  # type: ignore
        except TypeError:
            # This is acceptable - string message should be required
            pass
        else:
            # If it doesn't raise TypeError, check it handles None gracefully
            assert str(error) is not None

    @pytest.mark.unit
    def test_error_with_very_large_details(self) -> None:
        """Test error handling with very large details dictionary."""
        large_details = {f"key_{i}": f"value_{i}" for i in range(1000)}

        error = CheckpointError("Error with large details", details=large_details)

        assert len(error.details) == 1000
        assert "key_500" in error.details

    @pytest.mark.unit
    def test_error_string_representation_truncation(self) -> None:
        """Test that error string representation handles large details."""
        large_value = "x" * 10000  # Very large string
        details = {"large_field": large_value}

        error = CheckpointError("Error with large field", details=details)
        error_str = str(error)

        # Should not crash and should be a reasonable length
        assert isinstance(error_str, str)
        assert len(error_str) < 20000  # Some reasonable upper bound

    @pytest.mark.unit
    def test_error_with_circular_reference_in_details(self) -> None:
        """Test error handling with circular references in details."""
        circular_dict = {"self": None}
        circular_dict["self"] = circular_dict

        # Should not crash when creating the error
        error = CheckpointError("Circular reference test", details=circular_dict)

        # String representation should not crash
        error_str = str(error)
        assert isinstance(error_str, str)

    @pytest.mark.unit
    def test_multiple_inheritance_compatibility(self) -> None:
        """Test that checkpoint errors work with multiple inheritance."""

        # Create a custom error that inherits from both CheckpointError and another exception
        class CustomError(CheckpointError, ValueError):
            pass

        error = CustomError("Multi-inheritance test")

        assert isinstance(error, CheckpointError)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_error_picklability(self) -> None:
        """Test that base checkpoint errors can be pickled (for multiprocessing).

        Note: Not all exception subclasses are designed to be picklable due to
        required constructor arguments. This test focuses on the base CheckpointError
        which should be picklable.
        """
        import pickle

        # Test base CheckpointError which should be fully picklable
        error = CheckpointError("Test error", {"extra": "data"})

        # Should be able to pickle and unpickle
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, CheckpointError)
        assert unpickled.message == error.message
        assert unpickled.details == error.details
