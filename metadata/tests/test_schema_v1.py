# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the MetadataV1 top-level schema.

Covers construction, round-tripping, immutability, required-field
enforcement, and permissive section handling.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from anemoi.metadata.versions.v1 import InferenceMetadata
from anemoi.metadata.versions.v1 import MetadataV1


class TestMetadataV1Construction:
    """MetadataV1 accepts valid dicts and typed objects."""

    def test_accepts_full_v1_dict(self, sample_v1_dict):
        """Full V1 dict validates without errors."""
        meta = MetadataV1.model_validate(sample_v1_dict)
        assert meta.schema_version == "1.0"

    def test_metadata_inference_is_typed(self, sample_metadata_v1):
        """metadata_inference is coerced to InferenceMetadata."""
        assert isinstance(sample_metadata_v1.metadata_inference, InferenceMetadata)

    def test_created_at_is_datetime(self, sample_metadata_v1):
        """created_at is parsed as a datetime from an ISO string."""
        assert isinstance(sample_metadata_v1.created_at, datetime)

    def test_created_at_value(self, sample_metadata_v1):
        """created_at has the expected year."""
        assert sample_metadata_v1.created_at.year == 2024

    def test_schema_version_stored(self, sample_metadata_v1):
        """schema_version is stored as-is."""
        assert sample_metadata_v1.schema_version == "1.0"


class TestMetadataV1ExtraKeys:
    """MetadataV1 preserves unknown top-level keys (extra='allow')."""

    def test_extra_top_level_key_preserved(self, sample_v1_dict):
        """An unknown top-level key survives validation."""
        sample_v1_dict["custom_flag"] = "experiment-v3"
        meta = MetadataV1.model_validate(sample_v1_dict)
        assert meta.model_dump()["custom_flag"] == "experiment-v3"

    def test_multiple_extra_keys_preserved(self, sample_v1_dict):
        """Multiple unknown keys are all preserved."""
        sample_v1_dict["notes"] = "ablation"
        sample_v1_dict["run_group"] = "sweep-001"
        meta = MetadataV1.model_validate(sample_v1_dict)
        dumped = meta.model_dump()
        assert dumped["notes"] == "ablation"
        assert dumped["run_group"] == "sweep-001"


class TestMetadataV1RoundTrip:
    """MetadataV1 round-trips through to_dict() / from_dict()."""

    def test_round_trip_schema_version(self, sample_metadata_v1):
        """schema_version survives to_dict() -> from_dict()."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.schema_version == sample_metadata_v1.schema_version

    def test_round_trip_seed(self, sample_metadata_v1):
        """seed inside metadata_inference survives round-trip."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.metadata_inference.seed == sample_metadata_v1.metadata_inference.seed

    def test_round_trip_dataset_names(self, sample_metadata_v1):
        """dataset_names list survives round-trip."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.metadata_inference.dataset_names == sample_metadata_v1.metadata_inference.dataset_names

    def test_round_trip_permissive_sections(self, sample_metadata_v1):
        """Permissive dict sections survive round-trip intact."""
        restored = MetadataV1.from_dict(sample_metadata_v1.to_dict())
        assert restored.training == sample_metadata_v1.training
        assert restored.dataset == sample_metadata_v1.dataset
        assert restored.environment == sample_metadata_v1.environment
        assert restored.provenance == sample_metadata_v1.provenance

    def test_to_dict_is_json_serialisable(self, sample_metadata_v1):
        """to_dict() output contains only JSON-serialisable types."""
        import json

        d = sample_metadata_v1.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)


class TestMetadataV1Frozen:
    """MetadataV1 is immutable (frozen=True)."""

    def test_assignment_raises(self, sample_metadata_v1):
        """Direct attribute assignment raises an error."""
        with pytest.raises(Exception):
            sample_metadata_v1.schema_version = "2.0"

    def test_nested_model_is_also_frozen(self, sample_metadata_v1):
        """Nested InferenceMetadata is also frozen."""
        with pytest.raises(Exception):
            sample_metadata_v1.metadata_inference.seed = 999


class TestMetadataV1RequiredFields:
    """MetadataV1 rejects dicts missing required fields."""

    def test_missing_schema_version_defaults_to_none(self, sample_v1_dict):
        """Missing schema_version defaults to None (legacy compat)."""
        del sample_v1_dict["schema_version"]
        v1 = MetadataV1.model_validate(sample_v1_dict)
        assert v1.schema_version is None

    def test_missing_metadata_inference_raises(self, sample_v1_dict):
        """Missing metadata_inference raises ValidationError."""
        del sample_v1_dict["metadata_inference"]
        with pytest.raises(ValidationError):
            MetadataV1.model_validate(sample_v1_dict)

    def test_missing_created_at_defaults_to_none(self, sample_v1_dict):
        """Missing created_at defaults to None (legacy compat)."""
        del sample_v1_dict["created_at"]
        v1 = MetadataV1.model_validate(sample_v1_dict)
        assert v1.created_at is None


class TestMetadataV1PermissiveSections:
    """Permissive dict sections accept arbitrary nested structures."""

    def test_config_accepts_deep_nesting(self, sample_v1_dict):
        """config section accepts deeply nested dicts."""
        sample_v1_dict["config"] = {
            "model": {
                "encoder": {"layers": 8, "hidden": 512},
                "decoder": {"layers": 4},
            },
            "optimizer": {"type": "adam", "betas": [0.9, 0.999]},
        }
        meta = MetadataV1.model_validate(sample_v1_dict)
        assert meta.config["model"]["encoder"]["layers"] == 8

    def test_training_accepts_arbitrary_keys(self, sample_v1_dict):
        """training section accepts any keys."""
        sample_v1_dict["training"] = {
            "global_step": 200_000,
            "custom_metric": {"val_loss": 0.025, "train_loss": 0.018},
        }
        meta = MetadataV1.model_validate(sample_v1_dict)
        assert meta.training["custom_metric"]["val_loss"] == 0.025

    def test_environment_accepts_any_structure(self, sample_v1_dict):
        """environment section accepts any structure."""
        sample_v1_dict["environment"] = {
            "python_version": "3.12.0",
            "cuda_version": "12.1",
            "packages": {"numpy": "1.26.0", "torch": "2.3.0"},
        }
        meta = MetadataV1.model_validate(sample_v1_dict)
        assert meta.environment["packages"]["numpy"] == "1.26.0"

    def test_permissive_sections_default_to_empty_dict(self, sample_v1_dict):
        """Permissive sections default to {} when absent."""
        for key in ("config", "training", "dataset", "environment", "provenance"):
            sample_v1_dict.pop(key, None)
        meta = MetadataV1.model_validate(sample_v1_dict)
        assert meta.config == {}
        assert meta.training == {}
        assert meta.dataset == {}
        assert meta.environment == {}
        assert meta.provenance == {}
