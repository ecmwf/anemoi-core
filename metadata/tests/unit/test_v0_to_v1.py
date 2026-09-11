# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for v0_to_v1 migration.

Verifies that the V0 → V1 migration correctly synthesises metadata_inference,
preserves provenance, and maintains contract equivalence for shared accessors.
"""

from anemoi.metadata.migration import MetadataMigrator
from anemoi.metadata.versions.v0 import MetadataV0


def assert_contract_equivalent(old, new, methods):
    """Assert that contract accessor methods return equal values.

    Parameters
    ----------
    old : MetadataContract
        Original metadata instance.
    new : MetadataContract
        Migrated metadata instance.
    methods : list[str]
        List of method names to compare.
    """
    for method_name in methods:
        old_method = getattr(old, method_name)
        new_method = getattr(new, method_name)
        old_result = old_method()
        new_result = new_method()
        assert old_result == new_result, (
            f"{method_name}() differs after migration: " f"{old_result} (old) vs {new_result} (new)"
        )


class TestV0ToV1Migration:
    """Test the v0_to_v1 migration function."""

    def test_provenance_preserved(self, sample_v0_dict):
        """V0's provenance_training is correctly migrated to V1's provenance."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        # V0 had provenance_training, V1 should have it in provenance
        assert v1.get_provenance() == sample_v0_dict["provenance_training"]
        assert v1.get_provenance() != {}

    def test_schema_version_set(self, sample_v0_dict):
        """Migrated instance has schema_version '1.0'."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        assert v1.schema_version == "1.0"

    def test_original_schema_version_set(self, sample_v0_dict):
        """Migrated instance has original_schema_version '0.0'."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        assert v1.original_schema_version == "0.0"

    def test_contract_equivalence(self, sample_v0_dict):
        """Contract accessors return equivalent values before/after migration."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        # These accessors should return the same values after migration
        methods_to_check = [
            "get_provenance",
            "get_data_frequency",
            "get_timestep",
            "get_precision",
            "get_sources",
            "get_variable_indices",
            "get_output_variable_indices",
        ]

        assert_contract_equivalent(v0, v1, methods_to_check)

    def test_metadata_inference_synthesised(self, sample_v0_dict):
        """The metadata_inference block is synthesised from V0 data."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        # V1 should have a metadata_inference block
        assert v1.metadata_inference is not None
        assert v1.metadata_inference.dataset_names == ["data"]

        # Check that variable indices were synthesised
        data_config = v1.metadata_inference.datasets["data"]
        assert "2t" in data_config.data_indices.input
        assert "msl" in data_config.data_indices.input
        assert "10u" in data_config.data_indices.input
        assert "10v" in data_config.data_indices.input
        assert "lsm" in data_config.data_indices.input

    def test_variable_types_synthesised(self, sample_v0_dict):
        """Variable types are correctly synthesised from V0 config."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        var_types = v1.get_variable_types("data")
        assert "lsm" in var_types["forcing"]
        assert "2t" in var_types["prognostic"]
        assert "msl" in var_types["prognostic"]

    def test_timestep_preserved(self, sample_v0_dict):
        """Timestep is preserved from V0 config.data.timestep."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        assert v1.get_timestep("data") == "6h"

    def test_grid_points_synthesised(self, sample_v0_dict):
        """Grid size is synthesised from V0 dataset.shape."""
        v0 = MetadataV0.model_validate(sample_v0_dict)
        v1 = MetadataMigrator.migrate(v0, "1.0")

        # dataset.shape is [samples, variables, ensemble, grid_points]
        # The last element should be the grid size
        assert v1.get_grid_points("data") == 40320
