# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""End-to-end integration tests for the anemoi.metadata package.

Exercises complete workflows: dict → Metadata → to_dict() round-trips,
checkpoint save/load cycles, multi-dataset access, and pipeline scenarios
that mirror real inference usage.
"""

import json
import zipfile
from datetime import datetime

import pytest

from anemoi.metadata.checkpoint import save_metadata
from anemoi.metadata.interface import Metadata
from anemoi.metadata.versions.v1 import MetadataV1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_block(
    var_offset: int = 0,
    timestep: str = "6h",
    input_indices: list[int] | None = None,
    output_indices: list[int] | None = None,
) -> dict:
    """Build a per-dataset inference block for inline test data.

    Parameters
    ----------
    var_offset : int, optional
        Index offset so that two datasets have non-overlapping indices.
    timestep : str, optional
        Frequency string (e.g. ``"6h"``).
    input_indices : list[int] | None, optional
        Relative date indices for input; defaults to ``[-1, 0]``.
    output_indices : list[int] | None, optional
        Relative date indices for output; defaults to ``[1]``.

    Returns
    -------
    dict
        Per-dataset block matching what anemoi-training writes.
    """
    if input_indices is None:
        input_indices = [-1, 0]
    if output_indices is None:
        output_indices = [1]

    return {
        "data_indices": {
            "input": {
                "2t": var_offset + 0,
                "msl": var_offset + 1,
                "10u": var_offset + 2,
                "10v": var_offset + 3,
                "lsm": var_offset + 4,
            },
            "output": {
                "2t": var_offset + 0,
                "msl": var_offset + 1,
                "10u": var_offset + 2,
                "10v": var_offset + 3,
            },
        },
        "variable_types": {
            "prognostic": ["2t", "msl", "10u", "10v"],
            "forcing": ["lsm"],
            "diagnostic": [],
            "target": ["2t", "msl", "10u", "10v"],
        },
        "timesteps": {
            "timestep": timestep,
            "input_relative_date_indices": input_indices,
            "output_relative_date_indices": output_indices,
            "relative_date_indices_training": sorted(set(input_indices) | set(output_indices)),
        },
        "shapes": {
            "variables": 5,
            "input_timesteps": len(input_indices),
            "ensemble": 1,
            "grid": 40320,
        },
    }


def _make_v1_dict(
    dataset_names: list[str] | None = None,
    dataset_blocks: dict | None = None,
    task: str | None = "medium-range",
    seed: int = 42,
    run_id: str = "train-e2e-001",
    timestep: str = "6h",
) -> dict:
    """Build a complete V1 metadata dict for inline tests.

    Parameters
    ----------
    dataset_names : list[str] | None, optional
        Dataset names; defaults to ``["era5_1deg"]``.
    dataset_blocks : dict | None, optional
        Mapping of dataset name to block dict.  When ``None`` a single
        ``"era5_1deg"`` block is generated automatically.
    task : str, optional
        Task label.
    seed : int, optional
        Training seed.
    run_id : str, optional
        Training run identifier.
    timestep : str, optional
        Frequency string passed to auto-generated blocks.

    Returns
    -------
    dict
        Complete V1 metadata dict ready for ``Metadata.from_dict()``.
    """
    if dataset_names is None:
        dataset_names = ["era5_1deg"]
    if dataset_blocks is None:
        dataset_blocks = {
            name: _make_dataset_block(var_offset=i * 10, timestep=timestep) for i, name in enumerate(dataset_names)
        }

    inference: dict = {
        "seed": seed,
        "run_id": run_id,
        "task": task,
        "dataset_names": dataset_names,
        **dataset_blocks,
    }

    return {
        "schema_version": "1.0",
        "created_at": "2024-06-01T12:00:00",
        "metadata_inference": inference,
        "config": {
            "model": {"type": "graphtransformer", "num_layers": 16},
            "training": {"lr": 3e-4, "batch_size": 4},
        },
        "training": {
            "global_step": 100_000,
            "epoch": 50,
            "loss": 0.0312,
        },
        "dataset": {
            "name": dataset_names[0],
            "resolution": 1.0,
            "grid_points": 40320,
        },
        "environment": {
            "python_version": "3.11.9",
            "pytorch_version": "2.3.0",
            "anemoi_training_version": "0.4.0",
        },
        "provenance": {
            "git_sha": "deadbeef",
            "hostname": "gpu-node-01",
        },
    }


# ---------------------------------------------------------------------------
# Full round-trip: dict → Metadata → to_dict() → Metadata
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    """dict → Metadata → to_dict() → Metadata preserves all values."""

    @pytest.fixture()
    def original(self, sample_v1_dict):
        """Return the first Metadata loaded from the fixture dict."""
        return Metadata.from_dict(sample_v1_dict)

    @pytest.fixture()
    def restored(self, original):
        """Return a second Metadata loaded from original.to_dict()."""
        return Metadata.from_dict(original.to_dict())

    def test_schema_version_preserved(self, original, restored):
        """schema_version survives the round-trip."""
        assert restored.schema_version == original.schema_version

    def test_created_at_preserved(self, original, restored):
        """created_at survives the round-trip."""
        assert restored.created_at == original.created_at

    def test_dataset_names_preserved(self, original, restored):
        """dataset_names survives the round-trip."""
        assert restored.dataset_names == original.dataset_names

    def test_task_preserved(self, original, restored):
        """task survives the round-trip."""
        assert restored.task == original.task

    def test_timestep_preserved(self, original, restored):
        """timestep survives the round-trip."""
        assert restored.timestep == original.timestep

    def test_multi_step_input_preserved(self, original, restored):
        """multi_step_input survives the round-trip."""
        assert restored.multi_step_input == original.multi_step_input

    def test_multi_step_output_preserved(self, original, restored):
        """multi_step_output survives the round-trip."""
        assert restored.multi_step_output == original.multi_step_output

    def test_variables_preserved(self, original, restored):
        """variables list survives the round-trip."""
        assert set(restored.variables) == set(original.variables)

    def test_num_variables_preserved(self, original, restored):
        """num_variables survives the round-trip."""
        assert restored.num_variables == original.num_variables

    def test_variable_categories_preserved(self, original, restored):
        """variable_categories() output survives the round-trip."""
        orig_cats = original.variable_categories()
        rest_cats = restored.variable_categories()
        for key in ("forcing", "prognostic", "diagnostic", "target"):
            assert set(rest_cats[key]) == set(orig_cats[key])

    def test_config_section_preserved(self, original, restored):
        """config section survives the round-trip."""
        assert restored.get("config") == original.get("config")

    def test_training_section_preserved(self, original, restored):
        """training section survives the round-trip."""
        assert restored.get("training") == original.get("training")

    def test_environment_section_preserved(self, original, restored):
        """environment section survives the round-trip."""
        assert restored.get("environment") == original.get("environment")

    def test_provenance_section_preserved(self, original, restored):
        """provenance section survives the round-trip."""
        assert restored.get("provenance") == original.get("provenance")

    def test_to_dict_output_is_json_serialisable(self, original):
        """to_dict() output is fully JSON-serialisable."""
        json.dumps(original.to_dict())  # must not raise

    def test_double_round_trip_stable(self, original):
        """Two consecutive round-trips produce identical dicts."""
        first = original.to_dict()
        second = Metadata.from_dict(first).to_dict()
        assert first == second


# ---------------------------------------------------------------------------
# Checkpoint round-trip: Metadata → save → load → verify
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Metadata saved to a checkpoint file can be loaded back intact."""

    @pytest.fixture()
    def saved_checkpoint(self, empty_ckpt, sample_v1_dict):
        """Save a Metadata instance to a temp checkpoint and return the path."""
        raw = MetadataV1.model_validate(sample_v1_dict)
        ckpt = empty_ckpt("e2e_model.ckpt")
        save_metadata(ckpt, raw)
        return ckpt

    def test_checkpoint_file_exists(self, saved_checkpoint):
        """The checkpoint file is created on disk."""
        assert saved_checkpoint.exists()

    def test_checkpoint_is_valid_zip(self, saved_checkpoint):
        """The checkpoint file is a valid ZIP archive."""
        assert zipfile.is_zipfile(saved_checkpoint)

    def test_load_returns_metadata_instance(self, saved_checkpoint):
        """from_checkpoint() returns a Metadata instance."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        assert isinstance(m, Metadata)

    def test_schema_version_after_checkpoint(self, saved_checkpoint):
        """schema_version is the latest after checkpoint round-trip (auto-migrated)."""
        from anemoi.metadata.registry import MetadataRegistry

        m = Metadata.from_checkpoint(saved_checkpoint)
        assert m.schema_version == MetadataRegistry.latest_version()

    def test_dataset_names_after_checkpoint(self, saved_checkpoint):
        """dataset_names is correct after checkpoint round-trip."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        assert "era5_1deg" in m.dataset_names

    def test_timestep_after_checkpoint(self, saved_checkpoint):
        """timestep is correct after checkpoint round-trip."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        assert m.timestep == "6h"

    def test_multi_step_input_after_checkpoint(self, saved_checkpoint):
        """multi_step_input is correct after checkpoint round-trip."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        assert m.multi_step_input == 2

    def test_multi_step_output_after_checkpoint(self, saved_checkpoint):
        """multi_step_output is correct after checkpoint round-trip."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        assert m.multi_step_output == 1

    def test_variables_after_checkpoint(self, saved_checkpoint):
        """variables are correct after checkpoint round-trip."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        expected = {"2t", "msl", "10u", "10v", "lsm"}
        assert set(m.variables) == expected

    def test_inference_accessible_after_checkpoint(self, saved_checkpoint):
        """Inference data is accessible via raw after checkpoint round-trip."""
        from anemoi.metadata.versions.v1 import InferenceMetadata

        m = Metadata.from_checkpoint(saved_checkpoint)
        assert isinstance(m.raw.metadata_inference, InferenceMetadata)

    def test_get_config_after_checkpoint(self, saved_checkpoint):
        """get('config') works after checkpoint round-trip."""
        m = Metadata.from_checkpoint(saved_checkpoint)
        assert isinstance(m.get("config"), dict)

    def test_fixture_checkpoint_round_trip(self, tmp_checkpoint):
        """The conftest tmp_checkpoint fixture loads correctly via from_checkpoint."""
        from anemoi.metadata.registry import MetadataRegistry

        m = Metadata.from_checkpoint(tmp_checkpoint)
        assert m.schema_version == MetadataRegistry.latest_version()
        assert m.timestep == "6h"
        assert m.multi_step_input == 2


# ---------------------------------------------------------------------------
# Multi-dataset: two datasets, per-dataset access
# ---------------------------------------------------------------------------


class TestMultiDataset:
    """Metadata with two datasets exposes each by name independently."""

    @pytest.fixture()
    def multi_v1_dict(self):
        """Return a V1 dict with two datasets: era5_1deg and cerra_025deg."""
        return _make_v1_dict(
            dataset_names=["era5_1deg", "cerra_025deg"],
            dataset_blocks={
                "era5_1deg": _make_dataset_block(var_offset=0),
                "cerra_025deg": _make_dataset_block(var_offset=10),
            },
            seed=7,
            run_id="train-multi-xyz",
        )

    @pytest.fixture()
    def meta(self, multi_v1_dict):
        """Return a Metadata instance from the two-dataset dict."""
        return Metadata.from_dict(multi_v1_dict)

    def test_dataset_names_has_two_entries(self, meta):
        """dataset_names contains both dataset names."""
        assert len(meta.dataset_names) == 2

    def test_dataset_names_contains_era5(self, meta):
        """dataset_names contains 'era5_1deg'."""
        assert "era5_1deg" in meta.dataset_names

    def test_dataset_names_contains_cerra(self, meta):
        """dataset_names contains 'cerra_025deg'."""
        assert "cerra_025deg" in meta.dataset_names

    def test_default_variable_indices_is_first_dataset(self, meta):
        """get_variable_indices() with no args returns the first dataset's indices."""
        first_name = meta.dataset_names[0]
        assert meta.raw.get_variable_indices() == meta.raw.get_variable_indices(first_name)

    def test_era5_input_indices_start_at_zero(self, meta):
        """era5_1deg variable indices start at 0."""
        indices = meta.raw.get_variable_indices("era5_1deg")
        assert min(indices.values()) == 0

    def test_cerra_input_indices_start_at_ten(self, meta):
        """cerra_025deg variable indices start at 10 (var_offset=10)."""
        indices = meta.raw.get_variable_indices("cerra_025deg")
        assert min(indices.values()) == 10

    def test_era5_and_cerra_indices_do_not_overlap(self, meta):
        """era5_1deg and cerra_025deg input indices are disjoint."""
        era5_vals = set(meta.raw.get_variable_indices("era5_1deg").values())
        cerra_vals = set(meta.raw.get_variable_indices("cerra_025deg").values())
        assert era5_vals.isdisjoint(cerra_vals)

    def test_era5_variable_names(self, meta):
        """era5_1deg has the expected variable names."""
        indices = meta.raw.get_variable_indices("era5_1deg")
        assert set(indices.keys()) == {
            "2t",
            "msl",
            "10u",
            "10v",
            "lsm",
        }

    def test_cerra_variable_names(self, meta):
        """cerra_025deg has the expected variable names."""
        indices = meta.raw.get_variable_indices("cerra_025deg")
        assert set(indices.keys()) == {
            "2t",
            "msl",
            "10u",
            "10v",
            "lsm",
        }

    def test_nonexistent_dataset_raises_key_error(self, meta):
        """get_variable_indices('unknown') raises KeyError."""
        with pytest.raises(KeyError):
            meta.raw.get_variable_indices("unknown")

    def test_multi_dataset_round_trip(self, multi_v1_dict):
        """Two-dataset metadata survives a full dict round-trip."""
        original = Metadata.from_dict(multi_v1_dict)
        restored = Metadata.from_dict(original.to_dict())
        assert restored.dataset_names == original.dataset_names
        for name in original.dataset_names:
            orig_indices = original.raw.get_variable_indices(name)
            rest_indices = restored.raw.get_variable_indices(name)
            assert orig_indices == rest_indices

    def test_multi_dataset_conftest_fixture(self, multi_dataset_inference_dict):
        """The conftest multi_dataset_inference_dict fixture loads correctly."""
        v1_dict = {
            "schema_version": "1.0",
            "created_at": "2024-06-01T12:00:00",
            "metadata_inference": multi_dataset_inference_dict,
        }
        m = Metadata.from_dict(v1_dict)
        assert len(m.dataset_names) == 2
        assert "era5_1deg" in m.dataset_names
        assert "cerra_025deg" in m.dataset_names


# ---------------------------------------------------------------------------
# Pipeline: create → save → load → use inference + get()
# ---------------------------------------------------------------------------


class TestPipeline:
    """Full inference pipeline: create metadata, persist, load, query."""

    @pytest.fixture()
    def pipeline_checkpoint(self, empty_ckpt):
        """Create a checkpoint from inline data and return its path."""
        v1_dict = _make_v1_dict(
            dataset_names=["era5_1deg"],
            seed=99,
            run_id="pipeline-test-001",
            task="medium-range",
            timestep="6h",
        )
        raw = MetadataV1.model_validate(v1_dict)
        ckpt = empty_ckpt("pipeline.ckpt")
        save_metadata(ckpt, raw)
        return ckpt

    @pytest.fixture()
    def loaded(self, pipeline_checkpoint):
        """Load Metadata from the pipeline checkpoint."""
        return Metadata.from_checkpoint(pipeline_checkpoint)

    # -- Basic load ----------------------------------------------------------

    def test_loaded_is_metadata_instance(self, loaded):
        """Loaded object is a Metadata instance."""
        assert isinstance(loaded, Metadata)

    def test_loaded_schema_version(self, loaded):
        """Loaded metadata has the latest schema version after auto-migration."""
        from anemoi.metadata.registry import MetadataRegistry

        assert loaded.schema_version == MetadataRegistry.latest_version()

    # -- Inference properties (via contract methods) -------------------------

    def test_inference_seed(self, loaded):
        """raw.metadata_inference.seed matches the value written to the checkpoint."""
        assert loaded.raw.metadata_inference.seed == 99

    def test_inference_run_id(self, loaded):
        """raw.metadata_inference.run_id matches the value written to the checkpoint."""
        assert loaded.raw.metadata_inference.run_id == "pipeline-test-001"

    def test_inference_task(self, loaded):
        """task matches the value written to the checkpoint."""
        assert loaded.task == "medium-range"

    def test_inference_dataset_names(self, loaded):
        """dataset_names contains 'era5_1deg'."""
        assert "era5_1deg" in loaded.dataset_names

    # -- Timestep / multi-step -----------------------------------------------

    def test_timestep(self, loaded):
        """timestep is '6h'."""
        assert loaded.timestep == "6h"

    def test_multi_step_input(self, loaded):
        """multi_step_input is 2."""
        assert loaded.multi_step_input == 2

    def test_multi_step_output(self, loaded):
        """multi_step_output is 1."""
        assert loaded.multi_step_output == 1

    # -- Variables -----------------------------------------------------------

    def test_variables_accessible(self, loaded):
        """variables list is accessible and non-empty."""
        assert len(loaded.variables) > 0

    def test_num_variables(self, loaded):
        """num_variables is 5."""
        assert loaded.num_variables == 5

    # -- get() on permissive sections ----------------------------------------

    def test_get_config_returns_dict(self, loaded):
        """get('config') returns a dict."""
        assert isinstance(loaded.get("config"), dict)

    def test_get_config_model_key(self, loaded):
        """get('config', 'model') returns the model sub-dict."""
        model = loaded.get("config", "model")
        assert isinstance(model, dict)
        assert model["type"] == "graphtransformer"

    def test_get_training_global_step(self, loaded):
        """get('training', 'global_step') returns the correct value."""
        assert loaded.get("training", "global_step") == 100_000

    def test_get_environment_python_version(self, loaded):
        """get('environment', 'python_version') returns the stored version."""
        assert loaded.get("environment", "python_version") == "3.11.9"

    def test_get_missing_key_returns_none(self, loaded):
        """get('config', 'missing_key') returns None."""
        assert loaded.get("config", "missing_key") is None

    def test_get_missing_section_returns_default(self, loaded):
        """get('missing_section', default='N/A') returns 'N/A'."""
        assert loaded.get("missing_section", default="N/A") == "N/A"

    # -- per-dataset contract accessors --------------------------------------

    def test_variable_indices_accessible(self, loaded):
        """get_variable_indices() is accessible after checkpoint load."""
        indices = loaded.raw.get_variable_indices()
        assert isinstance(indices, dict)
        assert len(indices) > 0

    def test_dataset_timestep_via_contract(self, loaded):
        """get_timestep() returns '6h' after checkpoint load."""
        assert loaded.raw.get_timestep() == "6h"

    # -- raw -----------------------------------------------------------------

    def test_raw_is_metadata_v1(self, loaded):
        """raw is a MetadataV1 instance (only version registered)."""
        from anemoi.metadata.versions.v1 import MetadataV1

        assert isinstance(loaded.raw, MetadataV1)

    # -- repr ----------------------------------------------------------------

    def test_repr_does_not_raise(self, loaded):
        """repr() does not raise after checkpoint load."""
        repr(loaded)  # must not raise


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty permissive sections, single-step models, etc."""

    def test_minimal_v1_dict_no_permissive_sections(self):
        """Metadata loads from a dict with no permissive sections.

        V1 data is auto-migrated to the latest version on load.
        """
        from anemoi.metadata.registry import MetadataRegistry

        minimal = {
            "schema_version": "1.0",
            "created_at": "2024-01-01T00:00:00",
            "metadata_inference": {
                "seed": 1,
                "run_id": "minimal-run",
                "task": None,
                "dataset_names": ["data"],
                "data": _make_dataset_block(),
            },
        }
        m = Metadata.from_dict(minimal)
        assert m.schema_version == MetadataRegistry.latest_version()
        assert m.task is None
        assert m.dataset_names == ["data"]

    def test_get_on_empty_permissive_section_returns_empty_dict(self):
        """get('config') returns {} when config was not provided."""
        minimal = {
            "schema_version": "1.0",
            "created_at": "2024-01-01T00:00:00",
            "metadata_inference": {
                "seed": 1,
                "run_id": "minimal-run",
                "task": None,
                "dataset_names": ["data"],
                "data": _make_dataset_block(),
            },
        }
        m = Metadata.from_dict(minimal)
        assert m.get("config") == {}

    def test_single_step_input_model(self):
        """Metadata with a single input step reports multi_step_input == 1."""
        v1_dict = _make_v1_dict(
            dataset_blocks={
                "era5_1deg": _make_dataset_block(
                    input_indices=[0],
                    output_indices=[1],
                )
            }
        )
        m = Metadata.from_dict(v1_dict)
        assert m.multi_step_input == 1

    def test_multi_step_output_three(self):
        """Metadata with three output steps reports multi_step_output == 3."""
        v1_dict = _make_v1_dict(
            dataset_blocks={
                "era5_1deg": _make_dataset_block(
                    input_indices=[-1, 0],
                    output_indices=[1, 2, 3],
                )
            }
        )
        m = Metadata.from_dict(v1_dict)
        assert m.multi_step_output == 3

    def test_task_none_is_valid(self):
        """Metadata with task=None is valid and accessible."""
        v1_dict = _make_v1_dict(task=None)
        v1_dict["metadata_inference"]["task"] = None
        m = Metadata.from_dict(v1_dict)
        assert m.task is None

    def test_extra_top_level_keys_preserved_in_to_dict(self):
        """Extra top-level keys in the V1 dict survive to_dict()."""
        v1_dict = _make_v1_dict()
        v1_dict["custom_experiment_tag"] = "ablation-v7"
        m = Metadata.from_dict(v1_dict)
        assert m.to_dict()["custom_experiment_tag"] == "ablation-v7"

    def test_checkpoint_created_at_is_datetime(self, tmp_checkpoint):
        """created_at is a datetime after loading from a checkpoint."""
        m = Metadata.from_checkpoint(tmp_checkpoint)
        assert isinstance(m.created_at, datetime)

    def test_select_variables_prognostic_via_interface(self, sample_v1_dict):
        """select_variables(include=['prognostic']) works via Metadata."""
        m = Metadata.from_dict(sample_v1_dict)
        result = m.select_variables(include=["prognostic"])
        assert set(result) == {"2t", "msl", "10u", "10v"}

    def test_select_variables_exclude_forcing_via_interface(self, sample_v1_dict):
        """select_variables(exclude=['forcing']) works via Metadata."""
        m = Metadata.from_dict(sample_v1_dict)
        result = m.select_variables(exclude=["forcing"])
        assert "lsm" not in result

    def test_validate_environment_returns_list_via_interface(self, sample_v1_dict):
        """validate_environment() is accessible via Metadata."""
        m = Metadata.from_dict(sample_v1_dict)
        result = m.validate_environment()
        assert isinstance(result, list)

    def test_get_environment_info_via_interface(self, sample_v1_dict):
        """get_environment_info() is accessible via Metadata."""
        m = Metadata.from_dict(sample_v1_dict)
        info = m.get_environment_info()
        assert isinstance(info, dict)
        assert "python_version" in info
