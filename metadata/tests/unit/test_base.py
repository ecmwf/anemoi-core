# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for base.py.

Covers MetadataContract, the requires_version decorator, the version property,
to_dict() / from_dict() serialisation, and frozen immutability.
"""

import pytest
from packaging.version import Version

from anemoi.metadata.base import requires_version
from anemoi.metadata.exceptions import VersionError
from anemoi.metadata.versions.v1 import MetadataV1

# ---------------------------------------------------------------------------
# requires_version decorator
# ---------------------------------------------------------------------------


class TestRequiresVersion:
    """requires_version gates methods by minimum schema version.

    The decorator calls ``self.version`` and ``self.schema_version``, so it
    must be applied to methods on :class:`MetadataContract` subclasses (which
    provide those attributes).  Tests use ``MetadataV1`` directly as the
    host class.
    """

    def test_raises_for_old_version(self, sample_metadata_v1):
        """Method decorated with requires_version("2.0") raises on V1."""

        # Patch a method onto MetadataV1 at the class level for this test.
        @requires_version("2.0")
        def new_feature(self):
            return "available"

        with pytest.raises(VersionError):
            new_feature(sample_metadata_v1)

    def test_passes_for_sufficient_version(self, sample_metadata_v1):
        """Method decorated with requires_version("1.0") passes on V1."""

        @requires_version("1.0")
        def v1_feature(self):
            return "available"

        assert v1_feature(sample_metadata_v1) == "available"

    def test_error_message_contains_version(self, sample_metadata_v1):
        """VersionError message includes the required version string."""

        @requires_version("3.0.0")
        def future_feature(self):
            return "nope"

        with pytest.raises(VersionError, match="3.0.0"):
            future_feature(sample_metadata_v1)

    def test_error_message_contains_method_name(self, sample_metadata_v1):
        """VersionError message includes the method name."""

        @requires_version("5.0.0")
        def my_special_method(self):
            return "nope"

        with pytest.raises(VersionError, match="my_special_method"):
            my_special_method(sample_metadata_v1)

    def test_decorator_preserves_function_name(self):
        """@requires_version preserves the wrapped function's __name__."""

        @requires_version("2.0")
        def compute_something(self):
            return 42

        assert compute_something.__name__ == "compute_something"

    def test_boundary_version_passes(self, sample_metadata_v1):
        """Exact version match (not strictly greater) passes the gate."""

        @requires_version("1.0")
        def exact_version_method(self):
            return "ok"

        assert exact_version_method(sample_metadata_v1) == "ok"


# ---------------------------------------------------------------------------
# version property
# ---------------------------------------------------------------------------


class TestVersionProperty:
    """MetadataContract.version returns a packaging.version.Version."""

    def test_version_is_version_instance(self, sample_metadata_v1):
        """version property returns a packaging.version.Version."""
        assert isinstance(sample_metadata_v1.version, Version)

    def test_version_value(self, sample_metadata_v1):
        """version property parses '1.0.0' correctly."""
        assert sample_metadata_v1.version == Version("1.0")

    def test_version_comparison(self, sample_metadata_v1):
        """version supports comparison with other Version objects."""
        assert sample_metadata_v1.version >= Version("1.0")
        assert sample_metadata_v1.version < Version("2.0")


# ---------------------------------------------------------------------------
# to_dict / from_dict
# ---------------------------------------------------------------------------


class TestToDictFromDict:
    """MetadataContract.to_dict() and from_dict() serialise/deserialise correctly."""

    def test_to_dict_returns_dict(self, sample_metadata_v1):
        """to_dict() returns a plain dict."""
        result = sample_metadata_v1.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_schema_version(self, sample_metadata_v1):
        """to_dict() output contains schema_version."""
        result = sample_metadata_v1.to_dict()
        assert result["schema_version"] == "1.0"

    def test_to_dict_is_json_serialisable(self, sample_metadata_v1):
        """to_dict() output is fully JSON-serialisable."""
        import json

        d = sample_metadata_v1.to_dict()
        # Should not raise
        json.dumps(d)

    def test_from_dict_returns_metadata_v1(self, sample_v1_dict):
        """from_dict() returns a MetadataV1 instance."""
        result = MetadataV1.from_dict(sample_v1_dict)
        assert isinstance(result, MetadataV1)

    def test_from_dict_validates_data(self, sample_v1_dict):
        """from_dict() validates the data (seed is preserved)."""
        result = MetadataV1.from_dict(sample_v1_dict)
        assert result.metadata_inference.seed == 42

    def test_round_trip_schema_version(self, sample_metadata_v1):
        """schema_version survives to_dict() -> from_dict()."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.schema_version == sample_metadata_v1.schema_version

    def test_round_trip_created_at(self, sample_metadata_v1):
        """created_at survives to_dict() -> from_dict()."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.created_at == sample_metadata_v1.created_at

    def test_round_trip_permissive_sections(self, sample_metadata_v1):
        """Permissive dict sections survive round-trip."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.environment == sample_metadata_v1.environment
        assert restored.provenance == sample_metadata_v1.provenance


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


class TestMetadataContractFrozen:
    """MetadataContract (and subclasses) are frozen (immutable)."""

    def test_schema_version_assignment_raises(self, sample_metadata_v1):
        """Assigning to schema_version raises an error."""
        with pytest.raises(Exception):
            sample_metadata_v1.schema_version = "9.9.9"

    def test_created_at_assignment_raises(self, sample_metadata_v1):
        """Assigning to created_at raises an error."""
        from datetime import datetime

        with pytest.raises(Exception):
            sample_metadata_v1.created_at = datetime.now()

    def test_metadata_inference_assignment_raises(self, sample_metadata_v1):
        """Assigning to metadata_inference raises an error."""
        with pytest.raises(Exception):
            sample_metadata_v1.metadata_inference = None
