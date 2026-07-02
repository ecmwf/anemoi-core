# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the Metadata user-facing interface (Phase 2).

Covers every public property and method on :class:`~anemoi.metadata.interface.Metadata`,
including typed inference accessors, safe ``get``/``__getitem__`` access for
permissive sections, mixin delegation, and checkpoint loading.
"""

from datetime import datetime

import pytest

from anemoi.metadata.base import MetadataContract
from anemoi.metadata.interface import Metadata
from anemoi.metadata.versions.v1 import MetadataV1

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMetadataConstruction:
    """Metadata can be built from a dict or a checkpoint file."""

    def test_from_dict_returns_metadata_instance(self, sample_v1_dict):
        """from_dict() returns a Metadata instance."""
        m = Metadata.from_dict(sample_v1_dict)
        assert isinstance(m, Metadata)

    def test_from_dict_migrate_false_still_works(self, sample_v1_dict):
        """from_dict(migrate=False) still returns a Metadata instance."""
        m = Metadata.from_dict(sample_v1_dict, migrate=False)
        assert isinstance(m, Metadata)

    def test_from_checkpoint_returns_metadata_instance(self, tmp_checkpoint):
        """from_checkpoint() returns a Metadata instance."""
        m = Metadata.from_checkpoint(tmp_checkpoint)
        assert isinstance(m, Metadata)

    def test_from_checkpoint_migrate_false(self, tmp_checkpoint):
        """from_checkpoint(migrate=False) still returns a Metadata instance."""
        m = Metadata.from_checkpoint(tmp_checkpoint, migrate=False)
        assert isinstance(m, Metadata)

    def test_from_checkpoint_schema_version(self, tmp_checkpoint):
        """from_checkpoint() auto-migrates to the latest schema version."""
        from anemoi.metadata.registry import MetadataRegistry

        m = Metadata.from_checkpoint(tmp_checkpoint)
        assert m.schema_version == MetadataRegistry.latest_version()


# ---------------------------------------------------------------------------
# Envelope properties
# ---------------------------------------------------------------------------


class TestEnvelopeProperties:
    """schema_version, created_at, raw, and to_dict() expose the envelope."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_schema_version_is_string(self, meta):
        """schema_version returns a str."""
        assert isinstance(meta.schema_version, str)

    def test_schema_version_value(self, meta):
        """schema_version returns the latest version after auto-migration."""
        from anemoi.metadata.registry import MetadataRegistry

        assert meta.schema_version == MetadataRegistry.latest_version()

    def test_created_at_is_datetime(self, meta):
        """created_at returns a datetime instance."""
        assert isinstance(meta.created_at, datetime)

    def test_created_at_year(self, meta):
        """created_at has the expected year from the fixture."""
        assert meta.created_at.year == 2024

    def test_raw_is_base_metadata(self, meta):
        """raw returns a MetadataContract instance."""
        assert isinstance(meta.raw, MetadataContract)

    def test_raw_is_metadata_v1(self, meta):
        """raw is MetadataV1 (only version registered)."""

        assert isinstance(meta.raw, MetadataV1)

    def test_to_dict_returns_dict(self, meta):
        """to_dict() returns a plain dict."""
        assert isinstance(meta.to_dict(), dict)

    def test_to_dict_contains_schema_version(self, meta):
        """to_dict() output contains schema_version."""
        from anemoi.metadata.registry import MetadataRegistry

        assert meta.to_dict()["schema_version"] == MetadataRegistry.latest_version()

    def test_to_dict_is_json_serialisable(self, meta):
        """to_dict() output is fully JSON-serialisable."""
        import json

        json.dumps(meta.to_dict())  # must not raise

    def test_to_dict_contains_metadata_inference(self, meta):
        """to_dict() output contains the metadata_inference key."""
        assert "metadata_inference" in meta.to_dict()


# ---------------------------------------------------------------------------
# Raw access to version-specific internals
# ---------------------------------------------------------------------------


class TestRawAccess:
    """metadata.raw exposes version-specific internals for advanced users.

    The ``inference`` property has been removed from the public interface.
    Users who need the typed V1 ``InferenceMetadata`` block can access it
    via ``metadata.raw.metadata_inference``.
    """

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_raw_metadata_inference_seed(self, meta):
        """raw.metadata_inference.seed matches the fixture value."""
        from anemoi.metadata.versions.v1 import InferenceMetadata

        assert isinstance(meta.raw.metadata_inference, InferenceMetadata)
        assert meta.raw.metadata_inference.seed == 42

    def test_raw_metadata_inference_run_id(self, meta):
        """raw.metadata_inference.run_id matches the fixture value."""
        assert meta.raw.metadata_inference.run_id == "train-abc123"

    def test_raw_metadata_inference_task(self, meta):
        """raw.metadata_inference.task matches the fixture value."""
        assert meta.raw.metadata_inference.task == "medium-range"

    def test_raw_metadata_inference_dataset_names(self, meta):
        """raw.metadata_inference.dataset_names contains the expected dataset."""
        assert "era5_1deg" in meta.raw.metadata_inference.dataset_names

    def test_inference_not_on_interface(self, meta):
        """The 'inference' property no longer exists on the public interface."""
        assert not hasattr(meta, "inference")


# ---------------------------------------------------------------------------
# dataset_names
# ---------------------------------------------------------------------------


class TestDatasetNames:
    """metadata.dataset_names returns the ordered list of dataset names."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_dataset_names_is_list(self, meta):
        """dataset_names returns a list."""
        assert isinstance(meta.dataset_names, list)

    def test_dataset_names_contains_era5(self, meta):
        """dataset_names contains 'era5_1deg' from the fixture."""
        assert "era5_1deg" in meta.dataset_names

    def test_dataset_names_length(self, meta):
        """dataset_names has exactly one entry for the single-dataset fixture."""
        assert len(meta.dataset_names) == 1

    def test_dataset_names_element_type(self, meta):
        """Every element of dataset_names is a str."""
        assert all(isinstance(n, str) for n in meta.dataset_names)


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------


class TestTask:
    """metadata.task returns the optional task label."""

    def test_task_returns_string_when_set(self, sample_v1_dict):
        """task returns a str when the fixture sets it."""
        m = Metadata.from_dict(sample_v1_dict)
        assert isinstance(m.task, str)

    def test_task_value(self, sample_v1_dict):
        """task returns 'medium-range' from the fixture."""
        m = Metadata.from_dict(sample_v1_dict)
        assert m.task == "medium-range"

    def test_task_returns_none_when_absent(self, sample_v1_dict):
        """task returns None when the field is not set."""
        sample_v1_dict["metadata_inference"]["task"] = None
        m = Metadata.from_dict(sample_v1_dict)
        assert m.task is None


# ---------------------------------------------------------------------------
# Per-dataset contract accessors
# ---------------------------------------------------------------------------


class TestPerDatasetAccessors:
    """Contract methods expose per-dataset inference data without version coupling.

    The ``dataset_config()`` method has been removed from the public interface.
    Per-dataset data is now accessed via the contract methods:
    ``get_variable_indices()``, ``get_variable_types()``,
    ``get_tensor_shapes()``, ``get_timestep()``, etc.
    """

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_get_variable_indices_default(self, meta):
        """raw.get_variable_indices() returns the input index mapping."""
        indices = meta.raw.get_variable_indices()
        assert isinstance(indices, dict)
        assert "2t" in indices

    def test_get_variable_indices_named(self, meta):
        """raw.get_variable_indices('era5_1deg') returns the correct mapping."""
        indices = meta.raw.get_variable_indices("era5_1deg")
        assert "2t" in indices

    def test_get_variable_indices_nonexistent_raises(self, meta):
        """raw.get_variable_indices('nonexistent') raises KeyError."""
        with pytest.raises(KeyError):
            meta.raw.get_variable_indices("nonexistent")

    def test_get_output_variable_indices_default(self, meta):
        """raw.get_output_variable_indices() returns the output index mapping."""
        indices = meta.raw.get_output_variable_indices()
        assert isinstance(indices, dict)
        assert "2t" in indices

    def test_get_variable_types_default(self, meta):
        """raw.get_variable_types() returns the category dict."""
        vt = meta.raw.get_variable_types()
        assert set(vt.keys()) == {"forcing", "prognostic", "diagnostic", "target"}

    def test_get_tensor_shapes_default(self, meta):
        """raw.get_tensor_shapes() returns the shape dict."""
        shapes = meta.raw.get_tensor_shapes()
        assert set(shapes.keys()) == {
            "variables",
            "input_timesteps",
            "ensemble",
            "grid",
        }

    def test_get_timestep_named(self, meta):
        """raw.get_timestep('era5_1deg') returns '6h'."""
        assert meta.raw.get_timestep("era5_1deg") == "6h"

    def test_dataset_config_not_on_interface(self, meta):
        """The 'dataset_config' method no longer exists on the public interface."""
        assert not hasattr(meta, "dataset_config")


# ---------------------------------------------------------------------------
# timestep
# ---------------------------------------------------------------------------


class TestTimestep:
    """metadata.timestep returns the frequency string from the first dataset."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_timestep_is_string(self, meta):
        """timestep returns a str."""
        assert isinstance(meta.timestep, str)

    def test_timestep_value(self, meta):
        """timestep returns '6h' from the fixture."""
        assert meta.timestep == "6h"


# ---------------------------------------------------------------------------
# multi_step_input / multi_step_output
# ---------------------------------------------------------------------------


class TestMultiStep:
    """multi_step_input and multi_step_output return the correct counts."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_multi_step_input_is_int(self, meta):
        """multi_step_input returns an int."""
        assert isinstance(meta.multi_step_input, int)

    def test_multi_step_input_value(self, meta):
        """multi_step_input is 2 (fixture has [-1, 0])."""
        assert meta.multi_step_input == 2

    def test_multi_step_output_is_int(self, meta):
        """multi_step_output returns an int."""
        assert isinstance(meta.multi_step_output, int)

    def test_multi_step_output_value(self, meta):
        """multi_step_output is 1 (fixture has [1])."""
        assert meta.multi_step_output == 1

    def test_multi_step_input_matches_index_length(self, meta):
        """multi_step_input equals len(input_relative_date_indices)."""
        indices = meta.raw.get_input_relative_date_indices()
        assert meta.multi_step_input == len(indices)

    def test_multi_step_output_matches_index_length(self, meta):
        """multi_step_output equals len(output_relative_date_indices)."""
        indices = meta.raw.get_output_relative_date_indices()
        assert meta.multi_step_output == len(indices)


# ---------------------------------------------------------------------------
# VariablesMixin delegation
# ---------------------------------------------------------------------------


class TestVariablesMixinDelegation:
    """Metadata delegates variable methods to VariablesMixin correctly."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_variables_returns_list(self, meta):
        """variables returns a list."""
        assert isinstance(meta.variables, list)

    def test_variables_contains_expected_names(self, meta):
        """variables contains all expected variable names from the fixture."""
        expected = {"2t", "msl", "10u", "10v", "lsm"}
        assert set(meta.variables) == expected

    def test_num_variables_is_int(self, meta):
        """num_variables returns an int."""
        assert isinstance(meta.num_variables, int)

    def test_num_variables_matches_variables_length(self, meta):
        """num_variables equals len(variables)."""
        assert meta.num_variables == len(meta.variables)

    def test_num_variables_value(self, meta):
        """num_variables is 5 for the single-dataset fixture."""
        assert meta.num_variables == 5

    def test_variable_categories_returns_dict(self, meta):
        """variable_categories() returns a dict."""
        assert isinstance(meta.variable_categories(), dict)

    def test_variable_categories_has_expected_keys(self, meta):
        """variable_categories() has forcing, prognostic, diagnostic, target."""
        cats = meta.variable_categories()
        assert set(cats.keys()) == {"forcing", "prognostic", "diagnostic", "target"}

    def test_variable_categories_prognostic_correct(self, meta):
        """prognostic category contains the expected variables."""
        cats = meta.variable_categories()
        assert set(cats["prognostic"]) == {"2t", "msl", "10u", "10v"}

    def test_variable_categories_forcing_correct(self, meta):
        """forcing category contains lsm."""
        cats = meta.variable_categories()
        assert "lsm" in cats["forcing"]

    def test_variable_categories_diagnostic_empty(self, meta):
        """diagnostic category is empty in the fixture."""
        cats = meta.variable_categories()
        assert cats["diagnostic"] == []


# ---------------------------------------------------------------------------
# get() – safe section accessor
# ---------------------------------------------------------------------------


class TestGetMethod:
    """metadata.get() safely accesses permissive sections and nested keys."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_get_section_returns_dict(self, meta):
        """get('config') returns the config dict."""
        result = meta.get("config")
        assert isinstance(result, dict)

    def test_get_section_config_has_model_key(self, meta):
        """get('config') dict contains the 'model' key from the fixture."""
        assert "model" in meta.get("config")

    def test_get_section_training_returns_dict(self, meta):
        """get('training') returns the training dict."""
        result = meta.get("training")
        assert isinstance(result, dict)

    def test_get_section_with_existing_key(self, meta):
        """get('config', 'model') returns the nested model dict."""
        result = meta.get("config", "model")
        assert isinstance(result, dict)
        assert result["type"] == "graphtransformer"

    def test_get_section_with_nonexistent_key_returns_none(self, meta):
        """get('config', 'nonexistent') returns None by default."""
        assert meta.get("config", "nonexistent") is None

    def test_get_nonexistent_section_returns_none(self, meta):
        """get('nonexistent_section') returns None by default."""
        assert meta.get("nonexistent_section") is None

    def test_get_nonexistent_section_with_custom_default(self, meta):
        """get('nonexistent_section', default=42) returns 42."""
        assert meta.get("nonexistent_section", default=42) == 42

    def test_get_nonexistent_section_with_string_default(self, meta):
        """get('nonexistent_section', default='fallback') returns 'fallback'."""
        assert meta.get("nonexistent_section", default="fallback") == "fallback"

    def test_get_environment_section(self, meta):
        """get('environment') returns the environment dict."""
        result = meta.get("environment")
        assert isinstance(result, dict)
        assert "python_version" in result

    def test_get_provenance_section(self, meta):
        """get('provenance') returns the provenance dict."""
        result = meta.get("provenance")
        assert isinstance(result, dict)

    def test_get_dataset_section(self, meta):
        """get('dataset') returns the dataset dict."""
        result = meta.get("dataset")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    """metadata[section] returns the section or raises AttributeError."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_getitem_config_returns_dict(self, meta):
        """metadata['config'] returns the config dict."""
        assert isinstance(meta["config"], dict)

    def test_getitem_training_returns_dict(self, meta):
        """metadata['training'] returns the training dict."""
        assert isinstance(meta["training"], dict)

    def test_getitem_environment_returns_dict(self, meta):
        """metadata['environment'] returns the environment dict."""
        assert isinstance(meta["environment"], dict)

    def test_getitem_nonexistent_raises_key_error(self, meta):
        """metadata['nonexistent'] raises KeyError."""
        with pytest.raises(KeyError):
            _ = meta["nonexistent"]

    def test_getitem_schema_version_returns_string(self, meta):
        """metadata['schema_version'] returns the latest version string."""
        from anemoi.metadata.registry import MetadataRegistry

        assert meta["schema_version"] == MetadataRegistry.latest_version()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    """repr(metadata) returns a sensible, non-crashing string."""

    @pytest.fixture()
    def meta(self, sample_v1_dict):
        """Return a Metadata instance from the sample V1 dict."""
        return Metadata.from_dict(sample_v1_dict)

    def test_repr_returns_string(self, meta):
        """repr() returns a str."""
        assert isinstance(repr(meta), str)

    def test_repr_contains_version(self, meta):
        """repr() contains the schema version."""
        from anemoi.metadata.registry import MetadataRegistry

        assert MetadataRegistry.latest_version() in repr(meta)

    def test_repr_contains_dataset_name(self, meta):
        """repr() contains the dataset name."""
        assert "era5_1deg" in repr(meta)

    def test_repr_does_not_raise(self, meta):
        """repr() does not raise any exception."""
        repr(meta)  # must not raise

    def test_str_does_not_raise(self, meta):
        """str() does not raise any exception."""
        str(meta)  # must not raise


class TestEnvelopePropertyAnnotations:
    """schema_version and original_schema_version can be None; is_legacy is robust."""

    def test_schema_version_none_for_missing_field(self, sample_v1_dict):
        """schema_version returns None when not set."""
        # Remove schema_version to simulate a hand-built instance
        v1_dict_no_version = sample_v1_dict.copy()
        del v1_dict_no_version["schema_version"]
        from anemoi.metadata.versions.v1 import MetadataV1

        raw = MetadataV1.model_validate(v1_dict_no_version)
        meta = Metadata(raw)
        assert meta.schema_version is None

    def test_original_schema_version_none_for_missing_field(self, sample_v1_dict):
        """original_schema_version returns None when not set."""
        # Native V1 instances don't have original_schema_version set
        v1_dict_no_orig = sample_v1_dict.copy()
        # original_schema_version defaults to None if not present
        from anemoi.metadata.versions.v1 import MetadataV1

        raw = MetadataV1.model_validate(v1_dict_no_orig)
        meta = Metadata(raw)
        # For native V1, original_schema_version is None
        assert meta.original_schema_version is None

    def test_is_legacy_false_for_v1(self, sample_v1_dict):
        """is_legacy returns False for a native V1 checkpoint."""
        m = Metadata.from_dict(sample_v1_dict, migrate=False)
        assert m.is_legacy is False

    def test_is_legacy_false_for_none_version(self, sample_v1_dict):
        """is_legacy returns False when original_version is None."""
        v1_dict_no_version = sample_v1_dict.copy()
        del v1_dict_no_version["schema_version"]
        from anemoi.metadata.versions.v1 import MetadataV1

        raw = MetadataV1.model_validate(v1_dict_no_version)
        meta = Metadata(raw)
        assert meta.is_legacy is False

    def test_is_legacy_true_for_v0(self):
        """is_legacy returns True for a V0 checkpoint (original_schema_version "0.0")."""
        # Construct a MetadataV1 that was migrated from V0
        from anemoi.metadata.versions.v1 import MetadataV1

        # Build a minimal V1 dict with original_schema_version = "0.0"
        v0_migrated_dict = {
            "schema_version": "1.0",
            "original_schema_version": "0.0",
            "metadata_inference": {
                "seed": 42,
                "run_id": "legacy-run",
                "task": None,
                "dataset_names": ["data"],
                "data": {
                    "data_indices": {
                        "input": {"2t": 0},
                        "output": {"2t": 0},
                    },
                    "variable_types": {
                        "prognostic": ["2t"],
                        "forcing": [],
                        "diagnostic": [],
                        "target": ["2t"],
                    },
                    "timesteps": {
                        "timestep": "6h",
                        "input_relative_date_indices": [0],
                        "output_relative_date_indices": [1],
                        "relative_date_indices_training": [0, 1],
                    },
                    "shapes": {
                        "variables": 1,
                        "input_timesteps": 1,
                        "ensemble": 1,
                        "grid": 1000,
                    },
                },
            },
        }
        raw = MetadataV1.model_validate(v0_migrated_dict)
        meta = Metadata(raw)
        assert meta.is_legacy is True
