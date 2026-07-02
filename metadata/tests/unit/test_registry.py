# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for MetadataRegistry.

Covers version listing, latest-version detection, schema-version extraction
from raw dicts, class retrieval, and the load() entry point.

Note: We do not attempt to register additional versions in tests because
re-registration raises ValueError and the registry is module-level state.
All tests work with the already-registered "1.0".
"""

import pytest

from anemoi.metadata.exceptions import UnknownVersionError
from anemoi.metadata.registry import MetadataRegistry
from anemoi.metadata.versions.v1 import MetadataV1


class TestListVersions:
    """MetadataRegistry.list_versions() returns a sorted list."""

    def test_returns_list(self):
        """list_versions() returns a list."""
        versions = MetadataRegistry.list_versions()
        assert isinstance(versions, list)

    def test_contains_v1(self):
        """list_versions() includes '1.0.0'."""
        assert "1.0" in MetadataRegistry.list_versions()

    def test_returns_copy(self):
        """list_versions() returns a copy, not the internal list."""
        v1 = MetadataRegistry.list_versions()
        v2 = MetadataRegistry.list_versions()
        assert v1 is not v2

    def test_sorted_ascending(self):
        """Versions are returned in ascending semver order."""
        versions = MetadataRegistry.list_versions()
        from packaging.version import Version

        assert versions == sorted(versions, key=Version)


class TestLatestVersion:
    """MetadataRegistry.latest_version() returns the highest version."""

    def test_returns_string(self):
        """latest_version() returns a string."""
        assert isinstance(MetadataRegistry.latest_version(), str)

    def test_returns_v1_as_latest(self):
        """V1 ('1.0') is the latest registered version (V0 < V1)."""
        assert MetadataRegistry.latest_version() == "1.0"

    def test_latest_is_last_in_sorted_list(self):
        """latest_version() matches the last entry in list_versions()."""
        assert MetadataRegistry.latest_version() == MetadataRegistry.list_versions()[-1]


class TestDetectVersion:
    """MetadataRegistry.detect_version() extracts schema_version from a dict."""

    def test_detects_v1(self):
        """Detects '1.0.0' from a dict with schema_version='1.0.0'."""
        data = {"schema_version": "1.0", "other": "stuff"}
        assert MetadataRegistry.detect_version(data) == "1.0"

    def test_warns_on_missing_schema_version(self):
        """Warns and defaults to '0.0' (V0) when schema_version is absent."""
        with pytest.warns(UserWarning, match="schema_version"):
            result = MetadataRegistry.detect_version({"no_version": True})
        assert result == "0.0"

    def test_warns_on_empty_dict(self):
        """Warns and defaults to '0.0' (V0) for an empty dict."""
        with pytest.warns(UserWarning):
            result = MetadataRegistry.detect_version({})
        assert result == "0.0"

    def test_returns_exact_string(self):
        """Returns the exact string stored in schema_version."""
        data = {"schema_version": "1.0"}
        result = MetadataRegistry.detect_version(data)
        assert result == "1.0"
        assert isinstance(result, str)


class TestGetVersion:
    """MetadataRegistry.get_version() retrieves the registered class."""

    def test_get_v1_returns_metadata_v1(self):
        """get_version('1.0.0') returns MetadataV1."""
        cls = MetadataRegistry.get_version("1.0")
        assert cls is MetadataV1

    def test_unknown_version_raises(self):
        """get_version() raises UnknownVersionError for unregistered versions."""
        with pytest.raises(UnknownVersionError):
            MetadataRegistry.get_version("99.0.0")

    def test_unknown_version_error_message(self):
        """UnknownVersionError message contains the bad version string."""
        with pytest.raises(UnknownVersionError, match="99.0.0"):
            MetadataRegistry.get_version("99.0.0")

    def test_get_latest_returns_metadata_v1(self):
        """get_latest() returns MetadataV1 (the highest registered version)."""
        from anemoi.metadata.versions.v1 import MetadataV1

        cls = MetadataRegistry.get_latest()
        assert cls is MetadataV1


class TestLoad:
    """MetadataRegistry.load() validates and optionally migrates metadata."""

    def test_load_migrate_false_returns_v1(self, sample_v1_dict):
        """load() with migrate=False returns a MetadataV1 instance."""
        meta = MetadataRegistry.load(sample_v1_dict, migrate=False)
        assert isinstance(meta, MetadataV1)

    def test_load_migrate_true_noop_with_single_version(self, sample_v1_dict):
        """load() with migrate=True is a no-op when only V1 is registered."""
        meta = MetadataRegistry.load(sample_v1_dict, migrate=True)
        assert isinstance(meta, MetadataV1)

    def test_load_default_returns_v1(self, sample_v1_dict):
        """load() returns V1 by default (only version registered)."""
        meta = MetadataRegistry.load(sample_v1_dict)
        assert isinstance(meta, MetadataV1)

    def test_load_preserves_schema_version(self, sample_v1_dict):
        """Loaded metadata has the correct schema_version."""
        meta = MetadataRegistry.load(sample_v1_dict, migrate=False)
        assert meta.schema_version == "1.0"

    def test_load_missing_schema_version_warns_and_defaults(self, sample_v1_dict):
        """load() warns and defaults to '1.0' when schema_version is absent but metadata_inference is present.

        A dict without ``schema_version`` but with ``metadata_inference`` is
        treated as a transitional V1 checkpoint (written before
        ``schema_version`` was added).
        """
        del sample_v1_dict["schema_version"]
        with pytest.warns(UserWarning, match="schema_version"):
            result = MetadataRegistry.load(sample_v1_dict, migrate=False)
        assert result.schema_version == "1.0"

    def test_load_unknown_version_raises(self, sample_v1_dict):
        """load() raises UnknownVersionError for an unregistered version."""
        sample_v1_dict["schema_version"] = "99.0.0"
        with pytest.raises(UnknownVersionError):
            MetadataRegistry.load(sample_v1_dict)

    def test_load_validates_inference_metadata(self, sample_v1_dict):
        """Loaded metadata has a fully validated InferenceMetadata block."""
        meta = MetadataRegistry.load(sample_v1_dict)
        assert meta.metadata_inference.seed == 42
        assert "era5_1deg" in meta.metadata_inference.datasets


class TestRegisterMinor:
    """MetadataRegistry.register_minor() registers minor versions."""

    def test_register_minor_reuses_class(self):
        """register_minor shares the schema class with the base version."""
        # Use a version unlikely to conflict with real registrations.
        MetadataRegistry.register_minor("1.99")
        try:
            assert MetadataRegistry.get_version("1.99") is MetadataV1
        finally:
            # Clean up to avoid polluting other tests.
            del MetadataRegistry._versions["1.99"]
            MetadataRegistry._sorted_versions = None

    def test_register_minor_duplicate_raises(self):
        """register_minor raises ValueError for an already-registered version."""
        with pytest.raises(ValueError, match="already registered"):
            MetadataRegistry.register_minor("1.0")

    def test_register_minor_unknown_base_raises(self):
        """register_minor raises ValueError for an unregistered base version."""
        with pytest.raises(ValueError, match="not registered"):
            MetadataRegistry.register_minor("99.1")

    def test_minor_version_appears_in_list(self):
        """Registered minor version appears in list_versions()."""
        MetadataRegistry.register_minor("1.98")
        try:
            assert "1.98" in MetadataRegistry.list_versions()
        finally:
            del MetadataRegistry._versions["1.98"]
            MetadataRegistry._sorted_versions = None

    def test_minor_version_can_load_v1_data(self, sample_v1_dict):
        """Data with a minor version string loads via the shared schema class."""
        MetadataRegistry.register_minor("1.97")
        try:
            sample_v1_dict["schema_version"] = "1.97"
            meta = MetadataRegistry.load(sample_v1_dict, migrate=False)
            assert isinstance(meta, MetadataV1)
            assert meta.schema_version == "1.97"
        finally:
            del MetadataRegistry._versions["1.97"]
            MetadataRegistry._sorted_versions = None
