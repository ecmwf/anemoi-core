# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the typed inference metadata models in versions/v1.py.

Covers :class:`InferenceMetadata`, :class:`DatasetInferenceConfig`,
:class:`DataIndices`, :class:`VariableTypes`, :class:`TimestepConfig`,
and :class:`TensorShapes`.
"""

import pytest
from pydantic import ValidationError

from anemoi.metadata.versions.v1 import DataIndices
from anemoi.metadata.versions.v1 import DatasetInferenceConfig
from anemoi.metadata.versions.v1 import InferenceMetadata
from anemoi.metadata.versions.v1 import TensorShapes
from anemoi.metadata.versions.v1 import TimestepConfig
from anemoi.metadata.versions.v1 import VariableTypes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dataset_block() -> dict:
    """Return the smallest valid per-dataset block."""
    return {
        "data_indices": {
            "input": {"2t": 0, "msl": 1},
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
            "input_relative_date_indices": [-1, 0],
            "output_relative_date_indices": [1],
            "relative_date_indices_training": [-1, 0, 1],
        },
        "shapes": {
            "variables": 2,
            "input_timesteps": 2,
        },
    }


def _flat_inference_dict(dataset_names=None) -> dict:
    """Return a flat inference dict (as training writes it)."""
    if dataset_names is None:
        dataset_names = ["era5_1deg"]
    d = {
        "seed": 42,
        "run_id": "train-abc123",
        "task": "medium-range",
        "dataset_names": dataset_names,
    }
    for name in dataset_names:
        d[name] = _minimal_dataset_block()
    return d


# ---------------------------------------------------------------------------
# InferenceMetadata - flat input (model_validator reshapes)
# ---------------------------------------------------------------------------


class TestInferenceMetadataFlatInput:
    """InferenceMetadata validates the flat checkpoint dict format."""

    def test_flat_dict_is_accepted(self, sample_inference_dict):
        """Flat dict with dataset entries at top level is reshaped and validated."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        assert meta.seed == 42
        assert meta.run_id == "train-abc123"
        assert meta.task == "medium-range"
        assert meta.dataset_names == ["era5_1deg"]

    def test_flat_dict_datasets_key_populated(self, sample_inference_dict):
        """After reshaping, datasets dict contains the expected dataset name."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        assert "era5_1deg" in meta.datasets

    def test_flat_dict_dataset_config_is_typed(self, sample_inference_dict):
        """Dataset entry is coerced to DatasetInferenceConfig."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        cfg = meta.datasets["era5_1deg"]
        assert isinstance(cfg, DatasetInferenceConfig)

    def test_flat_dict_data_indices_correct(self, sample_inference_dict):
        """Variable indices are preserved through reshaping."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        indices = meta.datasets["era5_1deg"].data_indices
        assert indices.input["2t"] == 0
        assert indices.input["lsm"] == 4

    def test_flat_dict_variable_types_correct(self, sample_inference_dict):
        """Variable type lists are preserved through reshaping."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        vt = meta.datasets["era5_1deg"].variable_types
        assert "2t" in vt.prognostic
        assert "lsm" in vt.forcing

    def test_flat_dict_timestep_config_correct(self, sample_inference_dict):
        """Timestep configuration is preserved through reshaping."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        ts = meta.datasets["era5_1deg"].timesteps
        assert ts.timestep == "6h"
        assert ts.input_relative_date_indices == [-1, 0]
        assert ts.output_relative_date_indices == [1]

    def test_flat_dict_shapes_correct(self, sample_inference_dict):
        """Tensor shapes are preserved through reshaping."""
        meta = InferenceMetadata.model_validate(sample_inference_dict)
        shapes = meta.datasets["era5_1deg"].shapes
        assert shapes.variables == 5
        assert shapes.input_timesteps == 2
        assert shapes.grid == 40320


# ---------------------------------------------------------------------------
# InferenceMetadata - structured input (datasets key already present)
# ---------------------------------------------------------------------------


class TestInferenceMetadataStructuredInput:
    """InferenceMetadata accepts already-structured dicts unchanged."""

    def test_structured_dict_accepted(self):
        """Dict with 'datasets' key is passed through without reshaping."""
        data = {
            "seed": 1,
            "run_id": "run-xyz",
            "dataset_names": ["era5_1deg"],
            "datasets": {
                "era5_1deg": _minimal_dataset_block(),
            },
        }
        meta = InferenceMetadata.model_validate(data)
        assert meta.seed == 1
        assert "era5_1deg" in meta.datasets

    def test_structured_dict_preserves_task_none(self):
        """task field defaults to None when absent."""
        data = {
            "seed": 1,
            "run_id": "run-xyz",
            "dataset_names": ["era5_1deg"],
            "datasets": {"era5_1deg": _minimal_dataset_block()},
        }
        meta = InferenceMetadata.model_validate(data)
        assert meta.task is None


# ---------------------------------------------------------------------------
# Extra fields
# ---------------------------------------------------------------------------


class TestInferenceMetadataExtraFields:
    """Extra fields at the root level are preserved (extra='allow')."""

    def test_extra_root_fields_preserved(self):
        """Unknown top-level keys survive round-trip."""
        data = _flat_inference_dict()
        data["custom_training_flag"] = True
        data["experiment_name"] = "ablation-v3"
        meta = InferenceMetadata.model_validate(data)
        dumped = meta.model_dump()
        assert dumped["custom_training_flag"] is True
        assert dumped["experiment_name"] == "ablation-v3"

    def test_extra_fields_in_flat_form_preserved(self):
        """Extra keys that are not dataset names are kept at root level."""
        data = _flat_inference_dict()
        data["notes"] = "baseline run"
        meta = InferenceMetadata.model_validate(data)
        assert meta.model_dump()["notes"] == "baseline run"


# ---------------------------------------------------------------------------
# Multi-dataset
# ---------------------------------------------------------------------------


class TestInferenceMetadataMultiDataset:
    """InferenceMetadata handles multiple datasets correctly."""

    def test_two_datasets_validate(self, multi_dataset_inference_dict):
        """Two datasets are both present in the validated datasets dict."""
        meta = InferenceMetadata.model_validate(multi_dataset_inference_dict)
        assert set(meta.datasets.keys()) == {"era5_1deg", "cerra_025deg"}

    def test_two_datasets_independent_indices(self, multi_dataset_inference_dict):
        """Each dataset has its own independent index mapping."""
        meta = InferenceMetadata.model_validate(multi_dataset_inference_dict)
        era5_idx = meta.datasets["era5_1deg"].data_indices.input["2t"]
        cerra_idx = meta.datasets["cerra_025deg"].data_indices.input["2t"]
        assert era5_idx != cerra_idx

    def test_single_dataset_validates(self):
        """Single-dataset flat dict validates correctly."""
        data = _flat_inference_dict(["era5_1deg"])
        meta = InferenceMetadata.model_validate(data)
        assert len(meta.datasets) == 1
        assert "era5_1deg" in meta.datasets


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestInferenceMetadataRoundTrip:
    """model_dump() -> model_validate() preserves all data."""

    def test_round_trip_flat_input(self, sample_inference_dict):
        """Round-trip through dump/validate preserves all fields."""
        original = InferenceMetadata.model_validate(sample_inference_dict)
        dumped = original.model_dump()
        restored = InferenceMetadata.model_validate(dumped)
        assert restored.seed == original.seed
        assert restored.run_id == original.run_id
        assert restored.dataset_names == original.dataset_names
        assert set(restored.datasets.keys()) == set(original.datasets.keys())

    def test_round_trip_preserves_variable_indices(self, sample_inference_dict):
        """Variable index values survive a dump/validate cycle."""
        original = InferenceMetadata.model_validate(sample_inference_dict)
        restored = InferenceMetadata.model_validate(original.model_dump())
        orig_indices = original.datasets["era5_1deg"].data_indices.input
        rest_indices = restored.datasets["era5_1deg"].data_indices.input
        assert orig_indices == rest_indices


# ---------------------------------------------------------------------------
# DataIndices validation
# ---------------------------------------------------------------------------


class TestDataIndices:
    """DataIndices rejects wrong types and enforces extra='forbid'."""

    def test_valid_data_indices(self):
        """Valid integer-valued dicts are accepted."""
        di = DataIndices(input={"2t": 0, "msl": 1}, output={"2t": 0})
        assert di.input["2t"] == 0

    def test_string_values_rejected(self):
        """String values instead of ints raise ValidationError."""
        with pytest.raises(ValidationError):
            DataIndices(input={"2t": "zero"}, output={"2t": 0})

    def test_extra_fields_rejected(self):
        """Extra fields raise ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError):
            DataIndices(input={"2t": 0}, output={"2t": 0}, unknown_field=True)

    def test_frozen(self):
        """DataIndices is immutable (frozen=True)."""
        di = DataIndices(input={"2t": 0}, output={"2t": 0})
        with pytest.raises(Exception):
            di.input = {"2t": 99}


# ---------------------------------------------------------------------------
# VariableTypes validation
# ---------------------------------------------------------------------------


class TestVariableTypes:
    """VariableTypes defaults to empty lists for all categories."""

    def test_all_defaults_empty(self):
        """All four category lists default to empty."""
        vt = VariableTypes()
        assert vt.forcing == []
        assert vt.target == []
        assert vt.prognostic == []
        assert vt.diagnostic == []

    def test_partial_specification(self):
        """Only specified categories need to be provided."""
        vt = VariableTypes(prognostic=["2t", "msl"])
        assert vt.prognostic == ["2t", "msl"]
        assert vt.forcing == []

    def test_extra_fields_rejected(self):
        """Extra category names raise ValidationError."""
        with pytest.raises(ValidationError):
            VariableTypes(unknown_category=["2t"])

    def test_frozen(self):
        """VariableTypes is immutable."""
        vt = VariableTypes(prognostic=["2t"])
        with pytest.raises(Exception):
            vt.prognostic = ["msl"]


# ---------------------------------------------------------------------------
# TimestepConfig validation
# ---------------------------------------------------------------------------


class TestTimestepConfig:
    """TimestepConfig requires all four fields."""

    def test_valid_timestep_config(self):
        """All required fields present validates successfully."""
        ts = TimestepConfig(
            timestep="6h",
            input_relative_date_indices=[-1, 0],
            output_relative_date_indices=[1],
            relative_date_indices_training=[-1, 0, 1],
        )
        assert ts.timestep == "6h"

    def test_missing_timestep_raises(self):
        """Missing 'timestep' field raises ValidationError."""
        with pytest.raises(ValidationError):
            TimestepConfig(
                input_relative_date_indices=[-1, 0],
                output_relative_date_indices=[1],
                relative_date_indices_training=[-1, 0, 1],
            )

    def test_missing_input_indices_raises(self):
        """Missing 'input_relative_date_indices' raises ValidationError."""
        with pytest.raises(ValidationError):
            TimestepConfig(
                timestep="6h",
                output_relative_date_indices=[1],
                relative_date_indices_training=[-1, 0, 1],
            )

    def test_missing_output_indices_raises(self):
        """Missing 'output_relative_date_indices' raises ValidationError."""
        with pytest.raises(ValidationError):
            TimestepConfig(
                timestep="6h",
                input_relative_date_indices=[-1, 0],
                relative_date_indices_training=[-1, 0, 1],
            )

    def test_missing_training_indices_raises(self):
        """Missing 'relative_date_indices_training' raises ValidationError."""
        with pytest.raises(ValidationError):
            TimestepConfig(
                timestep="6h",
                input_relative_date_indices=[-1, 0],
                output_relative_date_indices=[1],
            )

    def test_extra_fields_rejected(self):
        """Extra fields raise ValidationError."""
        with pytest.raises(ValidationError):
            TimestepConfig(
                timestep="6h",
                input_relative_date_indices=[-1, 0],
                output_relative_date_indices=[1],
                relative_date_indices_training=[-1, 0, 1],
                extra_key="oops",
            )


# ---------------------------------------------------------------------------
# TensorShapes validation
# ---------------------------------------------------------------------------


class TestTensorShapes:
    """TensorShapes allows grid=None and defaults ensemble=1."""

    def test_grid_none_allowed(self):
        """grid=None is a valid value."""
        shapes = TensorShapes(variables=5, input_timesteps=2, grid=None)
        assert shapes.grid is None

    def test_ensemble_defaults_to_one(self):
        """ensemble defaults to 1 when not provided."""
        shapes = TensorShapes(variables=5, input_timesteps=2)
        assert shapes.ensemble == 1

    def test_explicit_ensemble(self):
        """Explicit ensemble value is stored correctly."""
        shapes = TensorShapes(variables=5, input_timesteps=2, ensemble=8)
        assert shapes.ensemble == 8

    def test_explicit_grid(self):
        """Explicit grid value is stored correctly."""
        shapes = TensorShapes(variables=5, input_timesteps=2, grid=40320)
        assert shapes.grid == 40320

    def test_missing_variables_raises(self):
        """Missing required 'variables' field raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorShapes(input_timesteps=2)

    def test_missing_input_timesteps_raises(self):
        """Missing required 'input_timesteps' field raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorShapes(variables=5)

    def test_extra_fields_rejected(self):
        """Extra fields raise ValidationError."""
        with pytest.raises(ValidationError):
            TensorShapes(variables=5, input_timesteps=2, unknown=True)


# ---------------------------------------------------------------------------
# DatasetInferenceConfig
# ---------------------------------------------------------------------------


class TestDatasetInferenceConfig:
    """DatasetInferenceConfig bundles all per-dataset sub-models."""

    def test_valid_config(self):
        """All sub-fields present validates successfully."""
        cfg = DatasetInferenceConfig.model_validate(_minimal_dataset_block())
        assert isinstance(cfg.data_indices, DataIndices)
        assert isinstance(cfg.variable_types, VariableTypes)
        assert isinstance(cfg.timesteps, TimestepConfig)
        assert isinstance(cfg.shapes, TensorShapes)

    def test_extra_fields_rejected(self):
        """Extra fields at the DatasetInferenceConfig level are rejected."""
        block = _minimal_dataset_block()
        block["unexpected_key"] = "value"
        with pytest.raises(ValidationError):
            DatasetInferenceConfig.model_validate(block)

    def test_missing_data_indices_raises(self):
        """Missing data_indices raises ValidationError."""
        block = _minimal_dataset_block()
        del block["data_indices"]
        with pytest.raises(ValidationError):
            DatasetInferenceConfig.model_validate(block)


# ---------------------------------------------------------------------------
# Missing dataset entry
# ---------------------------------------------------------------------------


class TestMissingDatasetEntry:
    """InferenceMetadata raises when a listed dataset has no data."""

    def test_missing_dataset_data_raises(self):
        """dataset_names lists a name but no data block exists for it."""
        data = {
            "seed": 1,
            "run_id": "run-xyz",
            "dataset_names": ["era5_1deg", "missing_dataset"],
            # Only era5_1deg data is provided; missing_dataset has no block.
            "era5_1deg": _minimal_dataset_block(),
        }
        # The model_validator raises ValueError when dataset_names references
        # a dataset that has no corresponding data block.
        with pytest.raises(ValidationError, match="missing_dataset"):
            InferenceMetadata.model_validate(data)
