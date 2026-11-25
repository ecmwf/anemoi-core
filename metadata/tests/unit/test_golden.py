# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Golden-fixture tests: the public interface on every checkpoint format.

The golden files in ``tests/data`` (see ``tests/data/README.md``) are trimmed
real or synthetic metadata, one per format F0 to F8.  Here:

* every file opens through ``Metadata.from_dict`` and through
  ``Metadata.from_checkpoint`` on a checkpoint written with the low-level
  writer (never the strict path), and selects the expected component adapters;
* ``EXPECTED`` pins the values that matter per format;
* the ``TestG*`` classes give one named assertion per known gap (G1 to G8) of
  the adapter plan;
* an F9 dict (F8 plus ``component_versions``) round-trips through the strict
  ``save_metadata``.  Rejections of invalid dicts, unknown stamps and F0-F8
  are covered in ``test_write_schema.py``.
"""

from __future__ import annotations

import copy
import json
import warnings
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from anemoi.metadata import CURRENT_VERSION
from anemoi.metadata import DatasetSelectionError
from anemoi.metadata import Metadata
from anemoi.metadata import save_metadata
from anemoi.utils.checkpoints import save_metadata as save_metadata_low_level

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data"


def load_golden(name: str) -> dict:
    """Return a fresh copy of ``tests/data/<name>.json`` (same files as the ``golden_loader`` fixture)."""
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
# Expectation table: one entry per golden file
# ---------------------------------------------------------------------------
#
# version / components: top-level ``version`` and the inferred component versions (which select the adapters).
# task: (task name, timestep, n_step_input, n_step_output).
# datasets: per dataset (data_type, grid_size, number of variables, first variables).


EXPECTED: dict[str, dict] = {
    "f0_ai_models": {
        "version": None,
        "components": {"data": "0.1", "task": "0.1", "lineage": "0.1"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 10, ["2t", "10u"])},
    },
    "f0_no_variables_metadata": {
        "version": None,
        "components": {"data": "0.1", "task": "0.1", "lineage": "0.1"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 10, ["2t", "10u"])},
    },
    "f1_ensemble": {
        "version": "1.0",
        "components": {"data": "0.1", "task": "0.1", "lineage": "0.2"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 542080, 134, ["10u", "10v"])},
    },
    "f1_null_diagnostic": {
        "version": "1.0",
        "components": {"data": "0.1", "task": "0.1", "lineage": "0.1"},
        "task": ("forecaster", "24h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 30, ["avg_svn", "avg_sve"])},
    },
    "f1_single_dataset": {
        "version": "1.0",
        "components": {"data": "0.1", "task": "0.1", "lineage": "0.2"},
        "task": ("forecaster", "24h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 131, ["10u", "10v"])},
    },
    "f2_dev_scalar_index": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 101, ["10u", "10v"])},
    },
    "f2_early": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("forecaster", "6h", 2, 2),
        "datasets": {"data": ("gridded", 10944, 43, ["10u", "10v"])},
    },
    "f3_multi_dataset": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {
            "era5": ("gridded", 10944, 43, ["10u", "10v"]),
            "cerra": ("gridded", 25326, 17, ["cos_julian_day", "cos_latitude"]),
        },
    },
    "f3_single_dataset": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("forecaster", "6h", 2, 2),
        "datasets": {"data": ("gridded", 10944, 43, ["10u", "10v"])},
    },
    "f3_temporal_downscaler": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("time-interpolator", "6h", 2, 2),
        "datasets": {"data": ("gridded", 10944, 43, ["10u", "10v"])},
    },
    "f4_task_top_level": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 101, ["10u", "10v"])},
    },
    "f5_offsets": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "0.2"},
        "task": ("offset-forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 1748, 40, ["10u", "10v"])},
    },
    "f6_main": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "1.0"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 101, ["10u", "10v"])},
    },
    "f6_warm_start": {
        "version": "2.0",
        "components": {"data": "0.2", "task": "0.2", "lineage": "1.0"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {"data": ("gridded", 40320, 101, ["10u", "10v"])},
    },
    "f8_richer_batch": {
        "version": "3.0",
        "components": {"data": "1.0", "task": "1.0", "lineage": "1.0"},
        "task": ("forecaster", "6h", 2, 1),
        "datasets": {
            "data": ("gridded", 40320, 101, ["10u", "10v"]),
            "obs": ("tabular", None, 6, ["2t", "2d"]),
        },
    },
}

#: Files whose checkpoint has a flat (unkeyed) ``supporting_arrays_paths``: the single-dataset legacy ones.
_FLAT_ARRAYS = {name for name, expected in EXPECTED.items() if expected["components"]["data"] == "0.1"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(empty_ckpt: Callable[..., Path], name: str, data: dict, arrays: dict | None = None) -> Path:
    """Write *data* into a minimal checkpoint with the low-level writer (no validation)."""
    path = empty_ckpt(f"{name}.ckpt")
    save_metadata_low_level(str(path), data, supporting_arrays=arrays)
    return path


def _lat_lon(grid_size: int) -> dict[str, np.ndarray]:
    return {
        "latitudes": np.linspace(-90, 90, grid_size, dtype="float32"),
        "longitudes": np.linspace(0, 360, grid_size, endpoint=False, dtype="float32"),
    }


def _hours(values: list[timedelta]) -> list[float]:
    return [value.total_seconds() / 3600 for value in values]


def _timesteps(data: dict, dataset: str) -> dict:
    return data["metadata_inference"][dataset]["timesteps"]


def _check_dataset(meta: Metadata, name: str, expected: tuple) -> None:
    data_type, grid_size, n_variables, first_variables = expected
    view = meta.dataset(name)
    assert view.name == name
    assert view.data.data_type() == data_type
    assert view.data.is_tabular() is (data_type == "tabular")
    assert view.data.grid_size() == grid_size
    variables = view.data.variables()
    assert len(variables) == n_variables
    assert variables[: len(first_variables)] == first_variables


def _check_expected(meta: Metadata, expected: dict) -> None:
    assert meta.version == expected["version"]
    assert meta.component_versions == expected["components"]

    task_name, timestep, n_input, n_output = expected["task"]
    assert meta.task.task_name == task_name
    assert meta.task.timestep() == timestep
    assert meta.task.n_step_input == n_input
    assert meta.task.n_step_output == n_output

    assert meta.dataset_names == list(expected["datasets"])
    for name, dataset in expected["datasets"].items():
        _check_dataset(meta, name, dataset)

    if len(expected["datasets"]) == 1:
        (only,) = expected["datasets"].values()
        assert meta.grid_size == only[1]
        assert len(meta.variables) == only[2]
    else:
        for shortcut in ("variables", "grid_size", "latitudes", "longitudes"):
            with pytest.raises(DatasetSelectionError):
                getattr(meta, shortcut)


# ---------------------------------------------------------------------------
# Every golden file opens, from a dict and from a checkpoint
# ---------------------------------------------------------------------------


def test_table_has_one_entry_per_golden_file(golden_metadata: tuple[str, dict[str, Any]]) -> None:
    name, _ = golden_metadata
    assert name in EXPECTED


def test_open_from_dict(golden_metadata: tuple[str, dict[str, Any]]) -> None:
    name, data = golden_metadata
    meta = Metadata.from_dict(data)
    _check_expected(meta, EXPECTED[name])
    assert meta.to_dict() == data


def test_open_from_checkpoint(golden_metadata: tuple[str, dict[str, Any]], empty_ckpt: Callable[..., Path]) -> None:
    """Written with the low-level writer, read back with its arrays (lat/lon for gridded datasets)."""
    name, data = golden_metadata
    expected = EXPECTED[name]
    gridded = {ds: spec[1] for ds, spec in expected["datasets"].items() if spec[0] == "gridded"}
    arrays = _lat_lon(gridded["data"]) if name in _FLAT_ARRAYS else {ds: _lat_lon(size) for ds, size in gridded.items()}
    path = _write_checkpoint(empty_ckpt, name, data, arrays)

    meta = Metadata.from_checkpoint(path)
    _check_expected(meta, expected)

    stored = meta.to_dict()
    assert stored.pop("supporting_arrays_paths") is not None
    assert stored == {key: value for key, value in data.items() if key != "supporting_arrays_paths"}

    for ds, size in gridded.items():
        assert meta.dataset(ds).arrays.latitudes.shape == (size,)
        assert meta.dataset(ds).arrays.longitudes.shape == (size,)
    for ds, spec in expected["datasets"].items():
        if spec[0] == "tabular":
            assert meta.dataset(ds).arrays.latitudes is None
            assert len(meta.dataset(ds).arrays) == 0


# ---------------------------------------------------------------------------
# G1 to G8
# ---------------------------------------------------------------------------


class TestG1NestedDatasetSection:
    """``dataset`` is keyed by dataset name from 2.0; a flat dump is still read."""

    def test_data_request_per_dataset(self) -> None:
        meta = Metadata.from_dict(load_golden("f3_multi_dataset"))
        assert meta.dataset("era5").data.grid() == "O48"
        assert meta.dataset("era5").data.area() == [88.573, 0.0, -88.573, 358.27]
        assert meta.dataset("cerra").data.grid() == [0.2, 0.2]
        assert meta.dataset("cerra").data.area() == [60.0, 345.0, 35.0, 25.0]

    def test_flat_dataset_section_fallback(self) -> None:
        raw = load_golden("f6_main")
        nested = Metadata.from_dict(raw).dataset("data").data
        raw["dataset"] = raw["dataset"]["data"]
        flat = Metadata.from_dict(raw).dataset("data").data
        assert flat.grid() == nested.grid() == "O96"
        assert flat.data_request() == nested.data_request()
        assert flat.open_dataset_args() == nested.open_dataset_args()
        assert flat.data_frequency() == nested.data_frequency()


class TestG2Offsets:
    """The forecaster offset keys are read, not dropped."""

    def test_f5_offsets(self) -> None:
        task = Metadata.from_dict(load_golden("f5_offsets")).task
        assert task.task_name == "offset-forecaster"
        assert _hours(task.input_offsets) == [-6, 0]
        assert _hours(task.output_offsets) == [6]
        assert task.rollout_shift == timedelta(hours=6)
        assert task.advance_map == {"inin": [(1, 0)], "outin": [(0, 1)]}
        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2]

    def test_stored_offsets_win_over_derived(self) -> None:
        raw = load_golden("f5_offsets")
        timesteps = _timesteps(raw, "data")
        timesteps.update(input_offsets=["-12h", "0h"], output_offsets=["3h"], rollout_shift="3h")
        task = Metadata.from_dict(raw).task
        assert _hours(task.input_offsets) == [-12, 0]
        assert _hours(task.output_offsets) == [3]
        assert task.rollout_shift == timedelta(hours=3)


class TestTaskTimingOnTheView:
    """The task view carries the explicit times and the downscaler's look-ahead ``lagged``."""

    def test_temporal_downscaler(self) -> None:
        task = Metadata.from_dict(load_golden("f3_temporal_downscaler")).task
        assert task.input_explicit_times == [0, 2]
        assert task.target_explicit_times == [1, 2]
        assert _hours(task.lagged) == [0, 12]
        assert _hours(task.input_offsets) == [0, 12]
        assert _hours(task.output_offsets) == [6, 12]
        assert task.rollout_shift == timedelta(hours=12)

    def test_temporal_downscaler_with_training_task_config(self) -> None:
        raw = load_golden("f3_temporal_downscaler")
        raw["config"]["task"] = {
            "input_timestep": "6h",
            "output_timestep": "1h",
            "output_left_boundary": False,
            "output_right_boundary": True,
        }
        task = Metadata.from_dict(raw).task
        assert _hours(task.input_offsets) == _hours(task.lagged) == [0, 6]
        assert _hours(task.output_offsets) == [1, 2, 3, 4, 5, 6]
        assert task.rollout_shift == timedelta(hours=6)
        assert task.advance_map is None

    def test_forecaster_falls_back_to_the_relative_indices(self) -> None:
        task = Metadata.from_dict(load_golden("f6_main")).task
        assert task.input_explicit_times == task.input_relative_date_indices() == [0, 1]
        assert task.target_explicit_times == task.output_relative_date_indices() == [2]
        assert _hours(task.lagged) == [-6, 0]
        assert _hours(task.output_offsets) == [6]
        assert task.rollout_shift == timedelta(hours=6)


class TestG3TaskName:
    """``metadata_inference.task``, else the top-level ``task``, else ``forecaster``."""

    def test_f2_early_defaults_to_forecaster(self) -> None:
        raw = load_golden("f2_early")
        assert "task" not in raw["metadata_inference"] and "task" not in raw
        assert Metadata.from_dict(raw).task.task_name == "forecaster"

    def test_f4_top_level_task(self) -> None:
        raw = load_golden("f4_task_top_level")
        assert raw["metadata_inference"].get("task") is None
        assert raw["task"] == "forecaster"
        assert Metadata.from_dict(raw).task.task_name == "forecaster"

    def test_f6_inference_task(self) -> None:
        raw = load_golden("f6_main")
        assert "task" not in raw
        assert Metadata.from_dict(raw).task.task_name == "forecaster"

    def test_inference_task_wins_over_top_level(self) -> None:
        raw = load_golden("f6_main")
        raw["task"] = "temporal-downscaler"
        assert Metadata.from_dict(raw).task.task_name == "forecaster"


class TestG4ProvenanceAndTimestamp:
    """``provenance_training`` and ``timestamp`` are read through ``sections``."""

    def test_f6_main(self) -> None:
        sections = Metadata.from_dict(load_golden("f6_main")).sections
        assert {"time", "python", "module_versions"} <= set(sections.provenance)
        assert sections.timestamp == datetime.fromisoformat("2026-11-23T16:11:13.640918+00:00")

    def test_f1_single_dataset(self) -> None:
        sections = Metadata.from_dict(load_golden("f1_single_dataset")).sections
        assert "python" in sections.provenance
        assert sections.timestamp == datetime.fromisoformat("2026-02-28T06:02:18.360537+00:00")

    def test_absent_in_f0(self) -> None:
        sections = Metadata.from_dict(load_golden("f0_ai_models")).sections
        assert sections.provenance == {}
        assert sections.timestamp is None


class TestG5GridSize:
    """``grid_size`` is found on 2.0 files that store ``shapes.grid`` and on 3.0 files that record ``grid_size``."""

    def test_shapes_grid_key_in_2_0(self) -> None:
        raw = load_golden("f6_main")
        raw["metadata_inference"]["data"]["shapes"]["grid"] = 123
        assert Metadata.from_dict(raw).dataset("data").data.grid_size() == 123

    def test_dataset_shape_when_stored_grid_is_null_in_2_0(self) -> None:
        raw = load_golden("f6_main")
        assert raw["metadata_inference"]["data"]["shapes"]["grid"] is None
        assert Metadata.from_dict(raw).dataset("data").data.grid_size() == 40320

    def test_grid_size_in_3_0(self) -> None:
        meta = Metadata.from_dict(load_golden("f8_richer_batch"))
        assert meta.dataset("data").data.grid_size() == 40320
        assert meta.dataset("obs").data.grid_size() is None


class TestG6LegacyRepairs:
    """The repairs inference applied to F0 and F1 checkpoints."""

    def test_variables_metadata_synthesised_from_names(self) -> None:
        raw = load_golden("f0_no_variables_metadata")
        assert "variables_metadata" not in raw["dataset"]
        data = Metadata.from_dict(raw).dataset("data").data
        with pytest.warns(UserWarning, match="synthesising") as caught:
            metadata = data.variables_metadata()
        assert any("synthesising" in str(warning.message) for warning in caught)
        assert sorted(metadata) == sorted(data.variables())
        assert metadata["2t"]["mars"] == {"param": "2t", "levtype": "sfc"}
        assert metadata["tp"]["process"] == "accumulation"
        assert data.computed_forcings() == ["cos_latitude", "insolation"]

    def test_accumulated_flag(self) -> None:
        raw = load_golden("f0_ai_models")
        assert raw["dataset"]["variables_metadata"]["tp"]["accumulated"] is True
        assert Metadata.from_dict(raw).dataset("data").data.accumulations() == ["tp"]

    def test_grid_size_from_grid_table_without_shape(self) -> None:
        raw = load_golden("f0_ai_models")
        assert "shape" not in raw["dataset"]
        assert Metadata.from_dict(raw).grid_size == 40320

    def test_data_request_found_deep_in_the_dataset_chain(self) -> None:
        for name, grid in (("f0_ai_models", "O96"), ("f1_single_dataset", "O96"), ("f1_ensemble", "N320")):
            raw = load_golden(name)
            assert "data_request" not in raw["dataset"]
            data = Metadata.from_dict(raw).dataset("data").data
            assert data.grid() == grid
            assert data.data_request()["grid"] == grid

    def test_null_diagnostic(self) -> None:
        raw = load_golden("f1_null_diagnostic")
        assert raw["config"]["data"]["diagnostic"] is None
        data = Metadata.from_dict(raw).dataset("data").data
        assert data.variable_types()["diagnostic"] == []
        assert data.variable_types()["prognostic"]


class TestG7NewInterfaceValues:
    """Values the interface did not expose before: ids, lineage, seeds, data kind, versions (offsets: G2)."""

    def test_uuid_and_parent_uuid_on_f6_warm_start(self) -> None:
        raw = load_golden("f6_warm_start")
        lineage = Metadata.from_dict(raw).lineage
        assert lineage.uuid == raw["uuid"]
        assert lineage.parent_uuid == raw["parent_uuid"] == "0dc0e962-102b-47dd-9f35-2e3dbd285b9a"
        assert Metadata.from_dict(load_golden("f6_main")).lineage.parent_uuid is None

    def test_checkpoint_lineage(self) -> None:
        assert Metadata.from_dict(load_golden("f6_warm_start")).lineage.entries == [
            {"loading_strategy": "warm_start", "parent_uuid": "0dc0e962-102b-47dd-9f35-2e3dbd285b9a"},
        ]
        assert Metadata.from_dict(load_golden("f6_main")).lineage.entries == []

    def test_base_seed(self) -> None:
        seeds = Metadata.from_dict(load_golden("f6_main")).sections.seeds
        assert seeds == {"seed": 2102982546, "base_seed": 567166687}
        assert "base_seed" not in Metadata.from_dict(load_golden("f1_ensemble")).sections.seeds

    def test_data_type_and_grid_size_on_f8(self) -> None:
        meta = Metadata.from_dict(load_golden("f8_richer_batch"))
        assert [meta.dataset(name).data.data_type() for name in meta.dataset_names] == ["gridded", "tabular"]
        assert meta.dataset("data").data.grid_size() == 40320
        # Before 3.0 every dataset is gridded and the grid size comes from ``dataset.shape``.
        assert Metadata.from_dict(load_golden("f6_main")).dataset("data").data.data_type() == "gridded"

    def test_component_versions(self) -> None:
        assert Metadata.from_dict(load_golden("f1_single_dataset")).component_versions == {
            "data": "0.1",
            "task": "0.1",
            "lineage": "0.2",
        }
        assert Metadata.from_dict(load_golden("f6_main")).component_versions == {
            "data": "0.2",
            "task": "0.2",
            "lineage": "1.0",
        }
        assert Metadata.from_dict(load_golden("f8_richer_batch")).component_versions == {
            "data": "1.0",
            "task": "1.0",
            "lineage": "1.0",
        }


class TestG8InputOnlyAndTabular:
    """Tabular datasets have no grid; input-only datasets have no outputs."""

    def test_tabular_obs_values(self, empty_ckpt: Callable[..., Path]) -> None:
        raw = load_golden("f8_richer_batch")
        arrays = {"data": _lat_lon(40320)}
        meta = Metadata.from_checkpoint(_write_checkpoint(empty_ckpt, "f8", raw, arrays))
        obs = meta.dataset("obs")
        assert obs.data.is_tabular() and not obs.data.is_gridded()
        assert obs.data.grid_size() is None
        assert obs.data.data_request() is None
        assert obs.data.grid() is None
        assert obs.data.area() is None
        assert obs.arrays.latitudes is None
        assert obs.arrays.longitudes is None
        # The gridded dataset of the same checkpoint keeps its values and arrays.
        assert meta.dataset("data").data.data_request()["grid"] == "O96"
        assert meta.dataset("data").arrays.latitudes.shape == (40320,)

    def test_input_only_dataset(self) -> None:
        raw = load_golden("f8_richer_batch")
        assert not Metadata.from_dict(raw).dataset("obs").data.is_input_only()
        raw["metadata_inference"]["obs"]["data_indices"]["output"] = {}
        obs = Metadata.from_dict(raw).dataset("obs").data
        assert obs.is_input_only()
        assert obs.output_variable_indices() == {}
        assert obs.output_index_to_variable() == {}
        assert obs.prognostic_output_mask() == []
        assert obs.prognostic_input_mask() == []
        assert len(obs.variable_indices()) == 5

    def test_conflicting_timings_need_for_dataset(self) -> None:
        """Datasets with different timing: global access raises, ``for_dataset`` works."""
        from anemoi.metadata.components.task import TaskTimingError

        raw = load_golden("f8_richer_batch")
        _timesteps(raw, "obs")["timestep"] = "3h"
        task = Metadata.from_dict(raw).task
        with pytest.raises(TaskTimingError):
            task.timestep()
        assert task.for_dataset("data").timestep == "6h"
        assert task.for_dataset("obs").timestep == "3h"


# ---------------------------------------------------------------------------
# F9: strict write round trip
# ---------------------------------------------------------------------------


def test_f9_round_trips_through_strict_save(empty_ckpt: Callable[..., Path]) -> None:
    """F9 is F8 plus ``component_versions`` and version 3.1; it is the only format the strict writer accepts."""
    f9 = load_golden("f8_richer_batch")
    f9["version"] = CURRENT_VERSION
    f9["component_versions"] = {"data": "1.0", "task": "1.0", "lineage": "1.0"}
    original = copy.deepcopy(f9)

    path = empty_ckpt("f9.ckpt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_metadata(path, f9, supporting_arrays={"data": _lat_lon(40320)})
    assert f9 == original

    meta = Metadata.from_checkpoint(path)
    assert meta.version == "3.1"
    assert meta.component_versions == {"data": "1.0", "task": "1.0", "lineage": "1.0"}
    assert meta.dataset_names == ["data", "obs"]
    assert meta.dataset("obs").data.grid_size() is None
    assert meta.dataset("data").arrays.latitudes.shape == (40320,)
    stored = meta.to_dict()
    stored.pop("supporting_arrays_paths")
    assert stored == {key: value for key, value in original.items() if key != "supporting_arrays_paths"}
