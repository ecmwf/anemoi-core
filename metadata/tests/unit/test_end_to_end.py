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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anemoi.metadata.checkpoint import save_metadata
from anemoi.metadata.interface import Metadata

# ---------------------------------------------------------------------------
# Full round-trip: dict → Metadata → to_dict() → Metadata
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    """dict → Metadata → to_dict() → Metadata preserves all values, for every golden file."""

    def test_round_trip_is_stable_and_json_serialisable(self, golden_metadata: tuple[str, dict[str, Any]]) -> None:
        """to_dict() gives the dict back unchanged; a second round-trip gives the same dict."""
        _name, raw = golden_metadata
        first = Metadata.from_dict(raw).to_dict()
        second = Metadata.from_dict(first).to_dict()
        assert first == raw
        assert second == first
        json.dumps(first)

    def test_values_survive_the_round_trip(self, golden_metadata: tuple[str, dict[str, Any]]) -> None:
        """What the interface reads from the restored metadata equals what it read from the original."""
        _name, raw = golden_metadata
        original = Metadata.from_dict(raw)
        restored = Metadata.from_dict(original.to_dict())

        assert restored.version == original.version
        assert restored.component_versions == original.component_versions
        assert restored.dataset_names == original.dataset_names
        assert restored.task.task_name == original.task.task_name
        assert restored.task.timestep() == original.task.timestep()
        assert restored.task.n_step_input == original.task.n_step_input
        assert restored.task.n_step_output == original.task.n_step_output
        for name in original.dataset_names:
            assert restored.dataset(name).data.variables() == original.dataset(name).data.variables()
            assert (
                restored.dataset(name).data.variable_categories() == original.dataset(name).data.variable_categories()
            )
        assert restored.sections.timestamp == original.sections.timestamp
        assert restored.sections.config == original.sections.config
        assert restored.sections.training == original.sections.training
        assert restored.sections.provenance == original.sections.provenance


# ---------------------------------------------------------------------------
# Checkpoint round-trip: dict → save_metadata → from_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Metadata written with the package writer can be loaded back intact."""

    def test_every_golden_file_survives_save_and_load(
        self,
        golden_metadata: tuple[str, dict[str, Any]],
        empty_ckpt: Callable[..., Path],
    ) -> None:
        """The raw content read back equals the content written, and reads the same through the interface."""
        name, raw = golden_metadata
        # The golden files list array files that these checkpoints do not contain.
        raw.pop("supporting_arrays_paths", None)
        ckpt = empty_ckpt(f"{name}.ckpt")
        # Historical formats are not valid for the strict writer.
        save_metadata(ckpt, raw, validate=False)

        loaded = Metadata.from_checkpoint(ckpt)
        stored = loaded.to_dict()
        # The low-level writer sets supporting_arrays_paths; everything else is unchanged.
        stored.pop("supporting_arrays_paths", None)
        assert stored == raw

        original = Metadata.from_dict(raw)
        assert loaded.version == original.version
        assert loaded.component_versions == original.component_versions
        assert loaded.dataset_names == original.dataset_names
        assert loaded.task.timestep() == original.task.timestep()


# ---------------------------------------------------------------------------
# Pipeline: create → save → load → use
# ---------------------------------------------------------------------------


class TestPipeline:
    """Inference pipeline on the F6 file: write metadata with arrays, load, query."""

    @pytest.fixture
    def loaded(self, empty_ckpt: Callable[..., Path], golden_loader: Callable[[str], dict[str, Any]]) -> Metadata:
        """Write the F6 golden metadata and its coordinates to a checkpoint and load it."""
        raw = golden_loader("f6_main")
        raw.pop("supporting_arrays_paths")
        arrays = {
            "data": {
                "latitudes": np.linspace(-90, 90, 40320),
                "longitudes": np.linspace(0, 360, 40320, endpoint=False),
            },
        }
        ckpt = empty_ckpt("pipeline.ckpt")
        # Historical (2.0) content: it is not valid for the strict writer.
        save_metadata(ckpt, raw, supporting_arrays=arrays, validate=False)
        return Metadata.from_checkpoint(ckpt)

    def test_loaded_values(self, loaded: Metadata) -> None:
        """Checkpoint-level, task, data and section values of the F6 file after a checkpoint round-trip."""
        assert loaded.version == "2.0"
        assert loaded.component_versions == {"data": "0.2", "task": "0.2", "lineage": "1.0"}
        assert loaded.sections.seeds == {"seed": 2102982546, "base_seed": 567166687}
        assert loaded.sections.run_id == "cbe2a8e3-87a2-4bf5-83fd-6ca34970ec82"
        assert loaded.task.task_name == "forecaster"
        assert loaded.task.timestep() == "6h"
        assert (loaded.task.n_step_input, loaded.task.n_step_output) == (2, 1)
        assert loaded.dataset_names == ["data"]
        assert len(loaded.variables) == 101
        assert loaded.grid_size == 40320
        assert loaded.sections.training["global_step"] == 6

    def test_coordinates_are_loaded(self, loaded: Metadata) -> None:
        """The coordinates stored beside the metadata are read for the only dataset."""
        assert loaded.latitudes.shape == (40320,)
        assert loaded.dataset("data").arrays.longitudes.shape == (40320,)


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: no raw dumps, single-step and multi-step models."""

    def test_minimal_dict_without_raw_dumps(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """A dict with only version and metadata_inference loads: no task name reads as the forecaster, dumps as {}."""
        raw = golden_loader("f6_main")
        minimal = {"version": raw["version"], "metadata_inference": raw["metadata_inference"]}
        minimal["metadata_inference"]["task"] = None
        m = Metadata.from_dict(minimal)
        assert m.version == "2.0"
        assert m.task.task_name == "forecaster"
        assert m.dataset_names == ["data"]
        assert m.sections.config == {}
        assert m.sections.timestamp is None

    @pytest.mark.parametrize(
        ("input_indices", "output_indices", "n_input", "n_output"),
        [([0], [1], 1, 1), ([0, 1], [2, 3, 4], 2, 3)],
    )
    def test_step_counts_follow_the_relative_date_indices(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        input_indices: list[int],
        output_indices: list[int],
        n_input: int,
        n_output: int,
    ) -> None:
        """n_step_input and n_step_output are the number of input and output relative date indices."""
        raw = golden_loader("f6_main")
        raw["metadata_inference"]["data"]["timesteps"].update(
            input_relative_date_indices=input_indices,
            output_relative_date_indices=output_indices,
        )
        m = Metadata.from_dict(raw)
        assert m.task.n_step_input == n_input
        assert m.task.n_step_output == n_output
