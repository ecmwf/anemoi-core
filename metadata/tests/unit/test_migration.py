# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for MetadataMigrator.

Only version 1.0 is registered, so multi-step migration tests are limited.
This module tests the no-op paths, error paths, has_migration(), and the
registration mechanism.
"""

import pytest

from anemoi.metadata.exceptions import MigrationError
from anemoi.metadata.migration import MetadataMigrator
from anemoi.metadata.registry import MetadataRegistry


class TestHasMigration:
    """MetadataMigrator.has_migration() checks for registered pairs."""

    def test_no_migration_registered(self):
        """No real migrations exist yet (only V1 registered)."""
        assert MetadataMigrator.has_migration("1.0", "2.0") is False

    def test_same_version_pair_returns_false(self):
        """has_migration() returns False for a same-version pair."""
        assert MetadataMigrator.has_migration("1.0", "1.0") is False

    def test_completely_unknown_versions_return_false(self):
        """has_migration() returns False for completely unknown versions."""
        assert MetadataMigrator.has_migration("99.0", "100.0") is False


class TestSameVersionMigration:
    """Migrating to the same version is a no-op."""

    def test_same_version_returns_same_instance(self, sample_metadata_v1):
        """migrate() returns the same object when source == target."""
        result = MetadataMigrator.migrate(sample_metadata_v1, "1.0")
        assert result is sample_metadata_v1


class TestMigrateToLatest:
    """MetadataMigrator.migrate_to_latest() with single registered version."""

    def test_latest_is_noop_when_already_latest(self, sample_metadata_v1):
        """migrate_to_latest() is a no-op when metadata is already latest."""
        result = MetadataMigrator.migrate_to_latest(sample_metadata_v1)
        assert result is sample_metadata_v1

    def test_latest_version_matches_registry(self, sample_metadata_v1):
        """Result has the latest schema version from the registry."""
        result = MetadataMigrator.migrate_to_latest(sample_metadata_v1)
        # With only V1 registered, latest is "1.0" and it's a no-op
        assert result.schema_version == MetadataRegistry.latest_version()

    def test_none_schema_version_raises(self, sample_metadata_v1):
        """migrate() raises MigrationError if schema_version is None."""
        # Create a metadata with None version (legacy checkpoint)
        from anemoi.metadata.versions.v1 import MetadataV1

        data = sample_metadata_v1.model_dump()
        data.pop("schema_version", None)
        legacy = MetadataV1.model_validate(data)
        assert legacy.schema_version is None

        with pytest.raises(MigrationError, match="schema_version"):
            MetadataMigrator.migrate(legacy, "1.0")


class TestDowngradeRejected:
    """Downgrade migrations raise errors."""

    def test_migrate_to_unknown_version_raises(self, sample_metadata_v1):
        """Migrating to an unknown version raises MigrationError."""
        with pytest.raises(MigrationError):
            MetadataMigrator.migrate(sample_metadata_v1, "99.0")


class TestMigrationRegistration:
    """MetadataMigrator.register_migration() prevents duplicate registration."""

    def test_duplicate_registration_raises(self):
        """Registering the same migration pair twice raises ValueError."""

        @MetadataMigrator.register_migration("98.0", "99.0")
        def _dummy(old):
            return old

        with pytest.raises(ValueError, match="already registered"):

            @MetadataMigrator.register_migration("98.0", "99.0")
            def _dummy2(old):
                return old
