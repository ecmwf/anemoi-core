# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for MetadataMigrator.

Only version 1.0 is registered, so multi-step migration tests are limited.
This module tests the no-op paths, error paths, has_migration(), and the
registration mechanism.
"""

import sys

import pytest
from pydantic import Field

from anemoi.metadata.base import MetadataContract
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

    def test_duplicate_registration_raises(self, registry_state):
        """Registering the same migration pair twice raises ValueError."""

        @MetadataMigrator.register_migration("98.0", "99.0")
        def _dummy(old):
            return old

        with pytest.raises(ValueError, match="already registered"):

            @MetadataMigrator.register_migration("98.0", "99.0")
            def _dummy2(old):
                return old


class TestMinorVersionAutobridge:
    """Schema-sharing minor versions are automatically bridged with no-op version bumps."""

    def test_minor_version_noop_migration(self, registry_state, sample_metadata_v1):
        """Migrating to a schema-sharing minor uses copy_with for version bump."""
        # Register a minor version that shares the V1 schema
        MetadataRegistry.register_minor("1.1")

        # Migrate from 1.0 to 1.1 without a registered migration function
        result = MetadataMigrator.migrate(sample_metadata_v1, "1.1")

        assert result.schema_version == "1.1"
        assert type(result) is type(sample_metadata_v1)
        # Content should be identical except for schema_version
        assert result.metadata_inference == sample_metadata_v1.metadata_inference

    def test_load_with_minor_version_migration(self, registry_state, sample_v1_dict):
        """MetadataRegistry.load migrates to a schema-sharing minor successfully."""
        MetadataRegistry.register_minor("1.1")

        # Load should auto-detect "1.0" and migrate to latest ("1.1")
        result = MetadataRegistry.load(sample_v1_dict, migrate=True)

        assert result.schema_version == "1.1"


class _FakeMetadata97(MetadataContract):
    """Minimal concrete contract used to fake registry versions in tests."""

    schema_version: str | None = "97.0"
    data: dict = Field(default_factory=dict)

    def get_dataset_names(self):
        return []

    def get_task(self):
        return None

    def get_timestep(self, dataset_name=None):
        return "6h"

    def get_input_relative_date_indices(self, dataset_name=None):
        return []

    def get_output_relative_date_indices(self, dataset_name=None):
        return []

    def get_variable_indices(self, dataset_name=None):
        return {}

    def get_output_variable_indices(self, dataset_name=None):
        return {}

    def get_variable_types(self, dataset_name=None):
        return {}

    def get_tensor_shapes(self, dataset_name=None):
        return {}

    def get_variables_metadata(self, dataset_name=None):
        return {}

    def get_grid_points(self, dataset_name=None):
        return None

    def get_data_request(self, dataset_name=None):
        return {}

    def get_precision(self):
        return None

    def get_provenance(self):
        return {}

    def get_data_frequency(self, dataset_name=None):
        return None

    def get_sources(self, dataset_name=None):
        return []

    def get_open_dataset_args(self, dataset_name=None):
        return {}

    def get_dataloader_config(self, partition="training", dataset_name=None):
        return {}


class _FakeMetadata98(_FakeMetadata97):
    schema_version: str | None = "98.0"


class _FakeMetadata99(_FakeMetadata97):
    schema_version: str | None = "99.0"


class TestDirectMultiHop:
    """Direct multi-hop migrations are preferred over intermediate steps."""

    def test_direct_jump_preferred(self, registry_state):
        """A registered direct jump is used instead of intermediate steps."""
        FakeMetadata97 = _FakeMetadata97
        FakeMetadata98 = _FakeMetadata98
        FakeMetadata99 = _FakeMetadata99

        # Register versions
        MetadataRegistry._versions["97.0"] = FakeMetadata97
        MetadataRegistry._versions["98.0"] = FakeMetadata98
        MetadataRegistry._versions["99.0"] = FakeMetadata99
        MetadataRegistry._sorted_versions = None

        # Register a direct 97.0 -> 99.0 migration (no 97.0 -> 98.0 migration)
        @MetadataMigrator.register_migration("97.0", "99.0")
        def migrate_97_to_99(old):
            return FakeMetadata99.model_validate({"schema_version": "99.0", "data": old.data})

        # Create a 97.0 instance
        v97 = FakeMetadata97(schema_version="97.0", data={"test": "value"})

        # Migrate to 99.0 should use the direct jump
        result = MetadataMigrator.migrate(v97, "99.0")

        assert result.schema_version == "99.0"
        assert result.data == {"test": "value"}


class TestAllowStopPartialMigration:
    """allow_stop executes the resolvable prefix and warns when stopping short."""

    def _register_fakes(self):
        """Register the three fake versions with a single 97.0 -> 98.0 migration."""
        MetadataRegistry._versions["97.0"] = _FakeMetadata97
        MetadataRegistry._versions["98.0"] = _FakeMetadata98
        MetadataRegistry._versions["99.0"] = _FakeMetadata99
        MetadataRegistry._sorted_versions = None

        @MetadataMigrator.register_migration("97.0", "98.0")
        def migrate_97_to_98(old):
            return _FakeMetadata98.model_validate({"schema_version": "98.0", "data": old.data})

    def test_partial_migration_stops_at_furthest(self, registry_state):
        """With allow_stop, the resolvable prefix is executed before stopping."""
        self._register_fakes()
        v97 = _FakeMetadata97(schema_version="97.0", data={"test": "value"})

        # 97.0 -> 98.0 exists; 98.0 -> 99.0 does not and schemas differ.
        with pytest.warns(UserWarning, match="stopped at version 98.0"):
            result = MetadataMigrator.migrate(v97, "99.0", allow_stop=True)

        assert result.schema_version == "98.0"
        assert result.data == {"test": "value"}

    def test_warning_names_target_and_missing_step(self, registry_state):
        """The warning identifies the missing step and the original target."""
        self._register_fakes()
        v97 = _FakeMetadata97(schema_version="97.0", data={})

        with pytest.warns(UserWarning, match=r"98\.0 to 99\.0.*target was 99\.0"):
            MetadataMigrator.migrate_as_possible(v97, "99.0")

    def test_without_allow_stop_raises(self, registry_state):
        """Without allow_stop the same gap raises MigrationError at the gap."""
        self._register_fakes()
        v97 = _FakeMetadata97(schema_version="97.0", data={})

        with pytest.raises(MigrationError, match="from 98.0 to 99.0"):
            MetadataMigrator.migrate(v97, "99.0")

    def test_no_warning_on_complete_path(self, registry_state, sample_metadata_v1):
        """No warning is emitted when the target is reached."""
        MetadataRegistry.register_minor("1.1")

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            result = MetadataMigrator.migrate(sample_metadata_v1, "1.1", allow_stop=True)
        assert result.schema_version == "1.1"


class TestRegisterMinorValidation:
    """register_minor validates the version format."""

    def test_single_component_raises(self, registry_state):
        """register_minor('1') raises ValueError."""
        with pytest.raises(ValueError, match="does not match the required format"):
            MetadataRegistry.register_minor("1")

    def test_non_numeric_raises(self, registry_state):
        """register_minor('abc') raises ValueError."""
        with pytest.raises(ValueError, match="does not match the required format"):
            MetadataRegistry.register_minor("abc")

    def test_valid_format_accepted(self, registry_state):
        """register_minor('1.1') succeeds with valid format."""
        # Should not raise
        MetadataRegistry.register_minor("1.1")
        assert "1.1" in MetadataRegistry.list_versions()


class TestUnderscoreModuleSkip:
    """Underscore-prefixed modules in migrations/ are not imported."""

    def test_example_not_imported(self):
        """_example.py is not imported by migrations/__init__.py."""
        # Check that the _example module is not in sys.modules
        assert "anemoi.metadata.migrations._example" not in sys.modules
