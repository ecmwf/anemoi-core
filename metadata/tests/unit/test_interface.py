# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the component-organised Metadata interface.

Covers the checkpoint level (``version``, ``component_versions``,
``dataset_names``, ``task``, ``lineage``, ``sections``), the per-dataset
view (``data``, ``arrays``), the single-dataset shortcuts,
``with_overrides`` and the guarantee that everything returned is a copy.
"""

import json
from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anemoi.metadata.arrays import DatasetArrays
from anemoi.metadata.components.base import requires
from anemoi.metadata.exceptions import VersionError
from anemoi.metadata.interface import DatasetData
from anemoi.metadata.interface import DatasetSelectionError
from anemoi.metadata.interface import DatasetView
from anemoi.metadata.interface import Metadata
from anemoi.metadata.interface import SectionsView
from anemoi.metadata.interface import TaskView

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arrays(n: int, offset: float = 0.0) -> dict:
    """Return latitudes and longitudes of *n* points."""
    return {
        "latitudes": np.linspace(-90, 90, n) + offset,
        "longitudes": np.linspace(0, 359, n) + offset,
    }


@pytest.fixture
def raw(golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Raw dict of ``f6_main`` (one dataset, ``data``)."""
    return golden_loader("f6_main")


@pytest.fixture
def meta(raw: dict[str, Any]) -> Metadata:
    """Single-dataset Metadata (``data``)."""
    return Metadata.from_dict(raw)


@pytest.fixture
def multi_dict(golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Raw dict of ``f3_multi_dataset`` (``era5`` and ``cerra``)."""
    return golden_loader("f3_multi_dataset")


@pytest.fixture
def multi_meta(multi_dict: dict[str, Any]) -> Metadata:
    """Two-dataset Metadata."""
    return Metadata.from_dict(multi_dict)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Metadata can be built from a dict or a checkpoint file (every golden file: ``test_golden.py``)."""

    def test_from_dict_does_not_keep_the_dict(self, raw: dict[str, Any]) -> None:
        """Changing the dict after from_dict() does not change the instance."""
        m = Metadata.from_dict(raw)
        raw["metadata_inference"]["dataset_names"].append("extra")
        assert m.dataset_names == ["data"]

    def test_repr(self, meta: Metadata) -> None:
        """repr() names the version and the datasets."""
        assert repr(meta) == "Metadata(version='2.0', datasets=['data'])"


# ---------------------------------------------------------------------------
# Checkpoint level
# ---------------------------------------------------------------------------


class TestCheckpointLevel:
    """to_dict and the names that are not on the interface.

    Version, component_versions and dataset_names are checked for every golden file in ``test_golden.py``.
    """

    def test_to_dict_is_json_serialisable_and_keeps_unknown_keys(self, raw: dict[str, Any]) -> None:
        """to_dict() is JSON-serialisable and unknown top-level keys survive."""
        raw["custom_experiment_tag"] = "ablation-v7"
        stored = Metadata.from_dict(raw).to_dict()
        json.dumps(stored)
        assert stored["custom_experiment_tag"] == "ablation-v7"

    @pytest.mark.parametrize(
        "name",
        [
            "schema_version",
            "original_schema_version",
            "grid_points",
            "is_legacy",
            "created_at",
            "raw",
            "timestep",
            "multi_step_input",
            "multi_step_output",
            "num_variables",
            "precision",
            "provenance",
            "variable_categories",
            "select_variables",
            "typed_variables",
            "variables_metadata",
            "accumulations",
            "computed_forcings",
            "data_request",
            "data_frequency",
            "sources",
            "open_dataset_args",
            "dataloader_config",
            "get",
            "get_environment_info",
        ],
    )
    def test_removed_names_are_not_on_the_interface(self, meta: Metadata, name: str) -> None:
        """Old names and per-dataset forwards are gone (no aliases)."""
        assert not hasattr(meta, name)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TestTask:
    """Metadata.task is the checkpoint-level task (task name, timing and conflicts: ``test_golden.py``)."""

    def test_task_is_a_task_view(self, meta: Metadata) -> None:
        """Task is a read-only TaskView."""
        assert isinstance(meta.task, TaskView)
        assert repr(meta.task) == "TaskView(task_name='forecaster')"

    def test_timing_values(self, meta: Metadata) -> None:
        """Indices and lagged times of the F6 checkpoint (input steps 0 and 1, output step 2, 6h apart)."""
        task = meta.task
        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2]
        assert task.lagged == [timedelta(hours=-6), timedelta(0)]

    def test_timings_by_dataset(self, multi_meta: Metadata) -> None:
        """timings() has one entry per dataset."""
        assert list(multi_meta.task.timings()) == ["era5", "cerra"]

    def test_task_has_no_dataset_forwards_on_dataset_view(self, meta: Metadata) -> None:
        """DatasetView has no task."""
        assert not hasattr(meta.dataset(), "task")
        assert not hasattr(meta.dataset(), "timestep")


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class TestLineage:
    """Metadata.lineage reads uuid, parent_uuid and the entries; a value the file does not record raises."""

    def test_fresh_run(self, meta: Metadata) -> None:
        """F6 records the lineage of a fresh run: a uuid, no parent, no entries."""
        assert meta.lineage.uuid == "11f0410c-d9d2-4255-8232-8cc6602dd1ab"
        assert meta.lineage.parent_uuid is None
        assert meta.lineage.entries == []

    def test_not_recorded_raises(self, raw: dict[str, Any]) -> None:
        """A file that records no lineage raises VersionError rather than returning None."""
        for key in ("uuid", "parent_uuid", "checkpoint_lineage"):
            raw.pop(key, None)
        lineage = Metadata.from_dict(raw).lineage
        for name in ("uuid", "parent_uuid", "entries"):
            with pytest.raises(VersionError, match="lineage"):
                getattr(lineage, name)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class TestSections:
    """Metadata.sections reads the raw sections."""

    def test_is_sections_view(self, meta: Metadata) -> None:
        """Sections is a SectionsView."""
        assert isinstance(meta.sections, SectionsView)

    def test_identity_values(self, meta: Metadata) -> None:
        """run_id and precision of the F6 checkpoint."""
        sections = meta.sections
        assert sections.run_id == "cbe2a8e3-87a2-4bf5-83fd-6ca34970ec82"
        assert sections.precision == "16-mixed"

    def test_timestamp_falls_back_to_created_at(self, raw: dict[str, Any]) -> None:
        """Timestamp is read from created_at when there is no timestamp."""
        del raw["timestamp"]
        raw["created_at"] = "2024-06-01T12:00:00"
        assert Metadata.from_dict(raw).sections.timestamp == datetime.fromisoformat("2024-06-01T12:00:00")

    def test_dumps(self, meta: Metadata, raw: dict[str, Any]) -> None:
        """config, training, tracker and provenance are dicts as stored; F6 records no environment."""
        sections = meta.sections
        assert sections.config == raw["config"]
        assert sections.config["training"]["precision"] == "16-mixed"
        assert sections.training == {"current_epoch": 1, "global_step": 6, "elapsed_time": 85.70771217346191}
        assert sections.tracker == {"null": None}
        assert sections.provenance == raw["provenance_training"]
        assert sections.environment == {}

    def test_absent_dumps_are_empty(self) -> None:
        """An absent dump reads as an empty dict, an absent scalar as None."""
        sections = Metadata.from_dict({"version": "2.0", "metadata_inference": {"dataset_names": []}}).sections
        assert sections.config == {}
        assert sections.tracker == {}
        assert sections.environment == {}

    def test_get_returns_stored_key_or_default(self, meta: Metadata) -> None:
        """get() reads any top-level key; absent gives the default."""
        assert meta.sections.get("version") == "2.0"
        assert meta.sections.get("nope", 42) == 42


class TestGetItem:
    """metadata[key] returns raw top-level keys only."""

    def test_config(self, meta: Metadata, raw: dict[str, Any]) -> None:
        """metadata['config'] is the config dump."""
        assert meta["config"] == raw["config"]
        assert meta["config"]["training"]["precision"] == "16-mixed"

    def test_version(self, meta: Metadata) -> None:
        """metadata['version'] is the stored string."""
        assert meta["version"] == "2.0"

    @pytest.mark.parametrize("name", ["nonexistent", "to_dict", "dataset_names", "with_overrides", "task", "variables"])
    def test_methods_and_attributes_are_not_reachable(self, meta: Metadata, name: str) -> None:
        """Names that are not stored keys raise KeyError, even when they are attributes."""
        with pytest.raises(KeyError):
            _ = meta[name]


# ---------------------------------------------------------------------------
# Selecting a dataset
# ---------------------------------------------------------------------------


class TestDatasetSelection:
    """dataset(name) needs a name unless there is exactly one dataset."""

    def test_named(self, multi_meta: Metadata) -> None:
        """dataset(name) returns the view of that dataset."""
        assert multi_meta.dataset("cerra").name == "cerra"

    def test_single_dataset_name_optional(self, meta: Metadata) -> None:
        """With one dataset the name may be omitted."""
        assert meta.dataset().name == "data"

    def test_multi_dataset_name_required(self, multi_meta: Metadata) -> None:
        """With several datasets omitting the name raises a clear error."""
        with pytest.raises(DatasetSelectionError, match=r"era5.*cerra"):
            multi_meta.dataset()

    def test_no_datasets(self, raw: dict[str, Any]) -> None:
        """With no dataset omitting the name raises a clear error."""
        raw["metadata_inference"]["dataset_names"] = []
        with pytest.raises(DatasetSelectionError, match="no datasets"):
            Metadata.from_dict(raw).dataset()

    def test_unknown_name(self, meta: Metadata) -> None:
        """An unknown name raises KeyError naming the available datasets."""
        with pytest.raises(KeyError, match=r"nonexistent.*data"):
            meta.dataset("nonexistent")

    def test_datasets_mapping(self, multi_meta: Metadata) -> None:
        """Datasets has one DatasetView per name, in order."""
        datasets = multi_meta.datasets
        assert list(datasets) == ["era5", "cerra"]
        assert all(isinstance(view, DatasetView) for view in datasets.values())

    def test_datasets_with_empty_dataset_names(self, raw: dict[str, Any]) -> None:
        """Datasets is empty when there are no datasets."""
        raw["metadata_inference"]["dataset_names"] = []
        assert Metadata.from_dict(raw).datasets == {}


# ---------------------------------------------------------------------------
# Single-dataset shortcuts
# ---------------------------------------------------------------------------


class TestSingleDatasetShortcuts:
    """latitudes and longitudes work for one dataset only.

    Variables, grid_size and the multi-dataset error are covered in ``test_golden.py``.
    """

    def test_latitudes_longitudes_none_without_arrays(self, meta: Metadata) -> None:
        """Latitudes and longitudes are None when no arrays were given."""
        assert meta.latitudes is None
        assert meta.longitudes is None

    def test_latitudes_longitudes(self, raw: dict[str, Any]) -> None:
        """Latitudes and longitudes come from the supporting arrays."""
        arrays = {"data": _arrays(8)}
        m = Metadata.from_dict(raw, supporting_arrays=arrays)
        np.testing.assert_array_equal(m.latitudes, arrays["data"]["latitudes"])
        np.testing.assert_array_equal(m.longitudes, arrays["data"]["longitudes"])

    @pytest.mark.parametrize("name", ["variables", "grid_size", "latitudes", "longitudes"])
    def test_raise_without_datasets(self, raw: dict[str, Any], name: str) -> None:
        """With no dataset the shortcuts raise."""
        raw["metadata_inference"]["dataset_names"] = []
        with pytest.raises(DatasetSelectionError):
            getattr(Metadata.from_dict(raw), name)


# ---------------------------------------------------------------------------
# DatasetView
# ---------------------------------------------------------------------------


class TestDatasetView:
    """DatasetView: name, data, arrays."""

    def test_members(self, meta: Metadata) -> None:
        """The view has name, data and arrays."""
        view = meta.dataset("data")
        assert view.name == "data"
        assert isinstance(view.data, DatasetData)
        assert isinstance(view.arrays, DatasetArrays)

    def test_repr(self, meta: Metadata) -> None:
        """repr() names the dataset."""
        assert repr(meta.dataset()) == "DatasetView(name='data')"

    def test_view_has_nothing_else(self, meta: Metadata) -> None:
        """No old per-dataset names on the view."""
        view = meta.dataset()
        for name in (
            "timestep",
            "variables",
            "variable_indices",
            "grid_points",
            "shapes",
            "multi_step_input",
        ):
            assert not hasattr(view, name)


class TestDatasetData:
    """DatasetView.data is the data component bound to the dataset."""

    def test_bound_values_equal_the_adapter_values(self, multi_meta: Metadata) -> None:
        """The view gives what the per-dataset adapter gives."""
        adapter = multi_meta.dataset("cerra")._raw.data("cerra")
        data = multi_meta.dataset("cerra").data

        assert data.variable_indices() == adapter.variable_indices()
        assert data.output_variable_indices() == adapter.output_variable_indices()
        assert data.input_index_to_variable() == adapter.input_index_to_variable()
        assert data.output_index_to_variable() == adapter.output_index_to_variable()
        assert data.variables() == adapter.variables()
        assert data.variable_types() == adapter.variable_types()
        assert data.prognostic_input_mask() == adapter.prognostic_input_mask()
        assert data.prognostic_output_mask() == adapter.prognostic_output_mask()
        assert data.variable_categories() == adapter.variable_categories()
        assert data.variable_categories(per_variable=True) == adapter.variable_categories(per_variable=True)
        assert data.select_variables(include=["prognostic"]) == adapter.select_variables(include=["prognostic"])
        assert data.variables_metadata() == adapter.variables_metadata()
        assert data.accumulations() == adapter.accumulations()
        assert data.computed_forcings() == adapter.computed_forcings()
        assert data.grid_size() == adapter.grid_size()
        assert data.data_request() == adapter.data_request()
        assert data.grid() == adapter.grid()
        assert data.area() == adapter.area()
        assert data.sources() == adapter.sources()
        assert data.open_dataset_args() == adapter.open_dataset_args()
        assert data.data_frequency() == adapter.data_frequency()
        assert data.dataloader_config("validation") == adapter.dataloader_config("validation")

    def test_typed_variables(self, meta: Metadata) -> None:
        """typed_variables() gives a Variable per variable (anemoi-transform is in the tests extra)."""
        pytest.importorskip("anemoi.transform")
        assert set(meta.dataset().data.typed_variables()) == set(meta.variables)

    def test_repr(self, meta: Metadata) -> None:
        """repr() names the dataset."""
        assert "data" in repr(meta.dataset().data)


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


class TestArrays:
    """DatasetView.arrays: coordinates, grid indices, output mask, named masks."""

    @pytest.fixture
    def arrays(self) -> dict[str, dict[str, Any]]:
        """Supporting arrays of the two datasets of ``multi_meta``."""
        return {
            "era5": {
                **_arrays(6),
                "grid_indices": np.arange(6),
                "output_mask": np.array([True] * 4 + [False] * 2),
                "cutout_mask": np.array([True, False] * 3),
            },
            "cerra": _arrays(4, offset=1.0),
        }

    def test_per_dataset_arrays(self, multi_dict: dict[str, Any], arrays: dict[str, dict[str, Any]]) -> None:
        """Each dataset gets its own arrays; absent ones are None."""
        m = Metadata.from_dict(multi_dict, supporting_arrays=arrays)

        era5 = m.dataset("era5").arrays
        np.testing.assert_array_equal(era5.latitudes, arrays["era5"]["latitudes"])
        np.testing.assert_array_equal(era5.longitudes, arrays["era5"]["longitudes"])
        np.testing.assert_array_equal(era5.grid_indices, np.arange(6))
        np.testing.assert_array_equal(era5.output_mask, arrays["era5"]["output_mask"])
        assert list(era5.masks) == ["cutout_mask"]
        np.testing.assert_array_equal(era5.masks["cutout_mask"], arrays["era5"]["cutout_mask"])
        np.testing.assert_array_equal(era5.get("cutout_mask"), arrays["era5"]["cutout_mask"])
        assert "cutout_mask" in era5
        assert sorted(era5.names) == sorted(arrays["era5"])

        cerra = m.dataset("cerra").arrays
        assert cerra.latitudes.shape == (4,)
        assert cerra.grid_indices is None
        assert cerra.output_mask is None
        assert cerra.masks == {}
        assert cerra.get("cutout_mask") is None

    def test_no_arrays_gives_none(self, meta: Metadata) -> None:
        """Without supporting arrays every attribute is None or empty."""
        arrays = meta.dataset().arrays
        assert arrays.latitudes is None
        assert arrays.longitudes is None
        assert arrays.grid_indices is None
        assert arrays.output_mask is None
        assert arrays.masks == {}
        assert len(arrays) == 0

    def test_flat_arrays_belong_to_the_only_dataset(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """Older single-dataset files (F1) store a flat mapping."""
        m = Metadata.from_dict(golden_loader("f1_single_dataset"), supporting_arrays=_arrays(4))
        assert m.latitudes.shape == (4,)
        assert m.dataset().arrays.longitudes.shape == (4,)

    def test_flat_arrays_with_several_datasets_are_dropped(
        self,
        multi_dict: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Flat arrays cannot be assigned to one of several datasets: dropped with a warning, not given to the first."""
        with caplog.at_level("WARNING"):
            m = Metadata.from_dict(multi_dict, supporting_arrays=_arrays(5))
        assert m.dataset("era5").arrays.latitudes is None
        assert m.dataset("cerra").arrays.latitudes is None
        assert "not keyed by dataset name" in caplog.text

    def test_from_checkpoint_loads_arrays(
        self,
        empty_ckpt: Callable[..., Path],
        multi_dict: dict[str, Any],
        arrays: dict[str, dict[str, Any]],
    ) -> None:
        """from_checkpoint() reads the supporting arrays stored beside the metadata."""
        import anemoi.utils.checkpoints as checkpoints_utils

        ckpt = empty_ckpt("with_arrays.ckpt")
        checkpoints_utils.save_metadata(str(ckpt), multi_dict, supporting_arrays=arrays)

        m = Metadata.from_checkpoint(ckpt)
        for name, stored in arrays.items():
            view = m.dataset(name).arrays
            np.testing.assert_array_equal(view.latitudes, stored["latitudes"])
            np.testing.assert_array_equal(view.longitudes, stored["longitudes"])
        np.testing.assert_array_equal(m.dataset("era5").arrays.grid_indices, np.arange(6))
        assert m.dataset("cerra").arrays.output_mask is None

    def test_from_checkpoint_without_arrays(self, empty_ckpt: Callable[..., Path], raw: dict[str, Any]) -> None:
        """A checkpoint without arrays gives None."""
        import anemoi.utils.checkpoints as checkpoints_utils

        # The golden file lists array files that this checkpoint does not contain.
        del raw["supporting_arrays_paths"]
        ckpt = empty_ckpt("no_arrays.ckpt")
        checkpoints_utils.save_metadata(str(ckpt), raw)
        assert Metadata.from_checkpoint(ckpt).latitudes is None

    def test_repr(self, meta: Metadata) -> None:
        """repr() lists the array names."""
        assert repr(meta.dataset().arrays) == "DatasetArrays([])"

    @pytest.mark.parametrize(
        ("fixture", "dataset", "version"),
        [
            ("f0_ai_models", "data", "0.1"),
            ("f1_single_dataset", "data", "0.1"),
            ("f2_dev_scalar_index", "data", "0.2"),
            ("f3_multi_dataset", None, "0.2"),
            ("f8_richer_batch", "data", "1.0"),
            ("f8_richer_batch", "obs", "1.0"),
        ],
    )
    def test_arrays_carry_the_data_component_and_version(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        fixture: str,
        dataset: str | None,
        version: str,
    ) -> None:
        """The arrays carry the component name and version of the dataset's data adapter, with or without arrays."""
        m = Metadata.from_dict(golden_loader(fixture), supporting_arrays=None)
        for name in [dataset] if dataset is not None else m.dataset_names:
            adapter = m._raw.data(name)
            arrays = m.dataset(name).arrays
            assert arrays.component == adapter.component == "data"
            assert arrays.version == adapter.version == version

    def test_arrays_with_content_carry_the_version(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """Arrays holding data carry the version too, not only the empty ones."""
        m = Metadata.from_dict(golden_loader("f8_richer_batch"), supporting_arrays={"data": _arrays(4)})
        assert len(m.dataset("data").arrays) > 0
        assert m.dataset("data").arrays.version == "1.0"

    def test_default_arrays_have_no_version(self) -> None:
        """Arrays built on their own have no version, so a gated member raises."""
        assert DatasetArrays().version is None
        assert DatasetArrays(version="0.2").version == "0.2"
        assert repr(DatasetArrays(version="0.2")) == "DatasetArrays([])"

    def test_arrays_carry_the_new_version_after_with_overrides(self, meta: Metadata) -> None:
        """Re-stamping the data component through with_overrides gives the new version, arrays kept."""
        meta = Metadata.from_dict(meta.to_dict(), supporting_arrays={"data": _arrays(4)})
        assert meta.dataset().arrays.version == "0.2"

        patched = meta.with_overrides({"component_versions": {"data": "1.0"}})
        assert patched.component_versions["data"] == "1.0"
        arrays = patched.dataset().arrays
        assert arrays.version == patched._raw.data("data").version == "1.0"
        assert arrays.latitudes.shape == (4,)
        # The original is unchanged.
        assert meta.dataset().arrays.version == "0.2"

    def test_arrays_keep_the_version_when_overriding_arrays(self, meta: Metadata) -> None:
        """Replacing arrays through with_overrides does not lose the version."""
        patched = meta.with_overrides({}, supporting_arrays={"data": _arrays(3)})
        assert patched.dataset().arrays.latitudes.shape == (3,)
        assert patched.dataset().arrays.version == "0.2"

    def test_from_checkpoint_arrays_carry_the_version(
        self,
        empty_ckpt: Callable[..., Path],
        multi_dict: dict[str, Any],
    ) -> None:
        """Arrays read back from a checkpoint are not empty and carry the data version."""
        import anemoi.utils.checkpoints as checkpoints_utils

        ckpt = empty_ckpt("version_arrays.ckpt")
        checkpoints_utils.save_metadata(
            str(ckpt),
            multi_dict,
            supporting_arrays={"era5": _arrays(6), "cerra": _arrays(4)},
        )

        m = Metadata.from_checkpoint(ckpt)
        for name in ("era5", "cerra"):
            arrays = m.dataset(name).arrays
            assert len(arrays) > 0
            assert arrays.component == "data"
            assert arrays.version == m._raw.data(name).version == m.component_versions["data"] == "0.2"

    def test_default_view_arrays_carry_the_adapter_version(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        """A DatasetView built with no arrays gets empty arrays with the data adapter's version."""
        for fixture, version in [("f0_ai_models", "0.1"), ("f6_main", "0.2"), ("f8_richer_batch", "1.0")]:
            raw_adapter = Metadata.from_dict(golden_loader(fixture))._raw
            view = DatasetView(raw_adapter, raw_adapter.dataset_names[0])
            assert len(view.arrays) == 0
            assert view.arrays.component == "data"
            assert view.arrays.version == raw_adapter.data(raw_adapter.dataset_names[0]).version == version

    @pytest.fixture
    def gated_arrays_class(self) -> type[DatasetArrays]:
        """A test-only DatasetArrays subclass with a member gated on the data version."""

        class GatedArrays(DatasetArrays):
            @property
            @requires(max_version="0.2")
            def legacy_only(self) -> str:
                return "legacy"

        return GatedArrays

    @pytest.mark.parametrize(
        ("fixture", "passes"),
        [("f0_ai_models", True), ("f2_dev_scalar_index", True), ("f8_richer_batch", False)],
    )
    def test_requires_gates_on_the_arrays_version(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        gated_arrays_class: type[DatasetArrays],
        fixture: str,
        passes: bool,
    ) -> None:
        """``@requires(max_version="0.2")`` passes for data 0.1 and 0.2 and raises for 1.0."""
        view = Metadata.from_dict(golden_loader(fixture)).dataset("data")
        gated = gated_arrays_class(version=view.arrays.version)
        if passes:
            assert gated.legacy_only == "legacy"
        else:
            with pytest.raises(VersionError, match=r"data\.legacy_only requires data version <= 0\.2.*version 1\.0"):
                gated.legacy_only

    def test_requires_raises_without_a_version(self, gated_arrays_class: type[DatasetArrays]) -> None:
        """Arrays without a version cannot satisfy any gate."""
        with pytest.raises(VersionError, match=r"data\.legacy_only"):
            gated_arrays_class().legacy_only

    def test_number_of_grid_points_precedence(self, raw: dict[str, Any]) -> None:
        """grid_indices, else latitudes, else the recorded grid size."""
        assert Metadata.from_dict(raw).dataset().number_of_grid_points == 40320
        with_lat = Metadata.from_dict(raw, supporting_arrays={"data": _arrays(6)})
        assert with_lat.dataset().number_of_grid_points == 6
        both = Metadata.from_dict(raw, supporting_arrays={"data": {**_arrays(6), "grid_indices": np.arange(4)}})
        assert both.dataset().number_of_grid_points == 4

    def test_number_of_grid_points_of_tabular_dataset(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """A tabular dataset without arrays has no grid points."""
        assert Metadata.from_dict(golden_loader("f8_richer_batch")).dataset("obs").number_of_grid_points is None


# ---------------------------------------------------------------------------
# with_overrides
# ---------------------------------------------------------------------------


class TestWithOverrides:
    """with_overrides(patch) returns a new Metadata from a deep-merged raw dict."""

    def test_returns_new_instance_and_leaves_original(self, meta: Metadata, raw: dict[str, Any]) -> None:
        """The original is unchanged and the new instance has the patched value."""
        patched = meta.with_overrides({"config": {"training": {"precision": "32"}}})

        assert patched is not meta
        assert isinstance(patched, Metadata)
        assert patched["config"]["training"]["precision"] == "32"
        assert meta["config"]["training"]["precision"] == "16-mixed"
        assert meta.to_dict() == raw

    def test_deep_merge_keeps_siblings(self, meta: Metadata) -> None:
        """Keys the patch does not mention are kept."""
        patched = meta.with_overrides({"config": {"training": {"precision": "32"}}})
        assert patched["config"]["training"]["method"] == meta["config"]["training"]["method"]
        assert patched["config"]["task"] == meta["config"]["task"]
        assert patched["training"] == meta["training"]

    def test_lists_and_scalars_are_replaced(self, meta: Metadata) -> None:
        """Non-dict values replace the stored value."""
        before = meta.dataset().data.variable_types()
        patched = meta.with_overrides({"metadata_inference": {"data": {"variable_types": {"forcing": []}}}})
        assert patched.dataset().data.variable_types()["forcing"] == []
        assert patched.dataset().data.variable_types()["prognostic"] == before["prognostic"]

    def test_new_keys_are_added(self, meta: Metadata) -> None:
        """Keys that are not stored yet are added."""
        patched = meta.with_overrides({"custom": {"a": 1}})
        assert patched["custom"] == {"a": 1}
        assert "custom" not in meta.to_dict()

    def test_patch_is_copied(self, meta: Metadata) -> None:
        """Changing the patch afterwards does not change the new instance."""
        patch = {"config": {"extra": {"values": [1, 2]}}}
        patched = meta.with_overrides(patch)
        patch["config"]["extra"]["values"].append(3)
        assert patched["config"]["extra"]["values"] == [1, 2]

    def test_chained_overrides(self, meta: Metadata) -> None:
        """Overrides can be chained; each step leaves the earlier instance unchanged."""
        first = meta.with_overrides({"config": {"a": 1}})
        second = first.with_overrides({"config": {"b": 2}})
        assert "b" not in first["config"]
        assert second["config"]["a"] == 1
        assert second["config"]["b"] == 2

    def test_overrides_change_what_adapters_read(self, meta: Metadata) -> None:
        """The new instance reads the patched raw dict (adapters are rebuilt)."""
        patched = meta.with_overrides({"metadata_inference": {"data": {"timesteps": {"timestep": "3h"}}}})
        assert patched.task.timestep() == "3h"
        assert meta.task.timestep() == "6h"

    def test_keeps_supporting_arrays(self, raw: dict[str, Any]) -> None:
        """The supporting arrays are kept, as copies independent of the original."""
        arrays = {"data": {**_arrays(6), "output_mask": np.ones(6, dtype=bool)}}
        original = Metadata.from_dict(raw, supporting_arrays=arrays)
        patched = original.with_overrides({"config": {"a": 1}})

        view = patched.dataset().arrays
        np.testing.assert_array_equal(view.latitudes, arrays["data"]["latitudes"])
        np.testing.assert_array_equal(view.longitudes, arrays["data"]["longitudes"])
        np.testing.assert_array_equal(view.output_mask, np.ones(6, dtype=bool))
        assert patched.latitudes is not original.latitudes

    def test_keeps_flat_arrays(self, raw: dict[str, Any]) -> None:
        """Flat arrays given to a single-dataset checkpoint survive too."""
        patched = Metadata.from_dict(raw, supporting_arrays=_arrays(5)).with_overrides({"a": 1})
        assert patched.latitudes.shape == (5,)

    def test_keeps_arrays_from_checkpoint(self, empty_ckpt: Callable[..., Path], raw: dict[str, Any]) -> None:
        """Arrays loaded by from_checkpoint are kept."""
        import anemoi.utils.checkpoints as checkpoints_utils

        ckpt = empty_ckpt("override.ckpt")
        checkpoints_utils.save_metadata(str(ckpt), raw, supporting_arrays={"data": _arrays(5)})
        patched = Metadata.from_checkpoint(ckpt).with_overrides({"config": {"a": 1}})
        assert patched.latitudes.shape == (5,)

    def test_supporting_arrays_replace_and_add_per_dataset(self, multi_dict: dict[str, Any]) -> None:
        """Named arrays are replaced or added for their dataset; everything else is kept, as copies."""
        original = Metadata.from_dict(
            multi_dict,
            supporting_arrays={"era5": {**_arrays(6), "cutout_mask": np.ones(6, dtype=bool)}, "cerra": _arrays(4)},
        )
        new_lat = np.zeros(3)
        patched = original.with_overrides(
            {"config": {"a": 1}},
            supporting_arrays={"era5": {"latitudes": new_lat, "grid_indices": np.arange(3)}},
        )

        era5 = patched.dataset("era5").arrays
        np.testing.assert_array_equal(era5.latitudes, new_lat)
        np.testing.assert_array_equal(era5.grid_indices, np.arange(3))
        np.testing.assert_array_equal(era5.longitudes, _arrays(6)["longitudes"])  # kept
        assert list(era5.masks) == ["cutout_mask"]  # kept
        assert patched.dataset("era5").number_of_grid_points == 3
        assert patched.dataset("cerra").arrays.latitudes.shape == (4,)  # other dataset untouched

        new_lat[:] = 9
        np.testing.assert_array_equal(patched.dataset("era5").arrays.latitudes, np.zeros(3))  # copied
        assert original.dataset("era5").arrays.latitudes.shape == (6,)  # original unchanged
        assert original.dataset("era5").arrays.grid_indices is None

    def test_supporting_arrays_can_be_added_to_a_dataset_without(self, meta: Metadata) -> None:
        """A dataset without arrays gets the given ones."""
        patched = meta.with_overrides({}, supporting_arrays={"data": _arrays(5)})
        assert patched.latitudes.shape == (5,)
        assert meta.latitudes is None

    def test_supporting_arrays_for_unknown_dataset_raise(self, meta: Metadata) -> None:
        """An array for a dataset the metadata does not have is an error, not silently dropped."""
        with pytest.raises(KeyError, match="nope"):
            meta.with_overrides({}, supporting_arrays={"nope": _arrays(5)})

    def test_tabular_dataset_stays_without_arrays(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """Arrays of the gridded dataset are kept and the tabular one still has none."""
        original = Metadata.from_dict(golden_loader("f8_richer_batch"), supporting_arrays={"data": _arrays(3)})
        patched = original.with_overrides({"config": {"a": 1}})
        assert patched.dataset("data").arrays.latitudes.shape == (3,)
        assert patched.dataset("obs").arrays.latitudes is None


# ---------------------------------------------------------------------------
# Copies
# ---------------------------------------------------------------------------


class TestReturnedValuesAreCopies:
    """Changing a returned list, dict or array never changes the instance.

    The task and data components are covered in ``test_component_task.py`` and ``test_component_data.py``.
    """

    def test_dataset_names(self, meta: Metadata) -> None:
        """dataset_names."""
        meta.dataset_names.append("x")
        assert meta.dataset_names == ["data"]

    def test_component_versions(self, meta: Metadata) -> None:
        """component_versions."""
        meta.component_versions["data"] = "9.9"
        assert meta.component_versions["data"] == "0.2"

    def test_datasets(self, meta: Metadata) -> None:
        """Datasets."""
        meta.datasets.clear()
        assert list(meta.datasets) == ["data"]

    def test_to_dict(self, meta: Metadata, raw: dict[str, Any]) -> None:
        """to_dict()."""
        meta.to_dict()["config"]["training"]["precision"] = "changed"
        assert meta.to_dict() == raw

    def test_getitem(self, meta: Metadata) -> None:
        """metadata[key]."""
        meta["config"]["training"]["precision"] = "changed"
        assert meta["config"]["training"]["precision"] == "16-mixed"

    def test_sections(self, meta: Metadata) -> None:
        """Every dict returned by sections."""
        meta.sections.config["training"]["precision"] = "changed"
        meta.sections.training["global_step"] = 0
        meta.sections.tracker["x"] = 1
        meta.sections.provenance["python"] = "0"
        meta.sections.seeds["seed"] = 0
        meta.sections.get("config")["training"]["precision"] = "changed"
        assert meta["config"]["training"]["precision"] == "16-mixed"
        assert meta["training"]["global_step"] == 6
        assert meta.sections.tracker == {"null": None}
        assert meta.sections.provenance["python"] != "0"
        assert meta.sections.seeds["seed"] == 2102982546

    def test_lineage_entries(self, raw: dict[str, Any]) -> None:
        """Mutating the returned entries does not change the instance."""
        raw["checkpoint_lineage"] = [{"loading_strategy": "warm_start", "parent_uuid": "a"}]
        m = Metadata.from_dict(raw)
        m.lineage.entries[0]["parent_uuid"] = "b"
        m.lineage.entries.append({})
        assert m.lineage.entries == [{"loading_strategy": "warm_start", "parent_uuid": "a"}]

    def test_arrays_returned_are_copies(self, raw: dict[str, Any]) -> None:
        """Mutating a returned array, or the array given in, does not change the instance."""
        given = {"data": {**_arrays(6), "cutout_mask": np.ones(6, dtype=bool)}}
        m = Metadata.from_dict(raw, supporting_arrays=given)
        expected = np.linspace(-90, 90, 6)

        m.dataset().arrays.latitudes[:] = 0
        m.latitudes[:] = 0
        m.dataset().arrays.masks["cutout_mask"][:] = False
        m.dataset().arrays.get("cutout_mask")[:] = False
        m.dataset().arrays.masks.clear()
        given["data"]["latitudes"][:] = 0

        np.testing.assert_array_equal(m.latitudes, expected)
        np.testing.assert_array_equal(m.dataset().arrays.masks["cutout_mask"], np.ones(6, dtype=bool))
        assert m.dataset().arrays.latitudes is not m.dataset().arrays.latitudes

    def test_arrays_are_writable_copies_of_read_only_input(self, raw: dict[str, Any]) -> None:
        """Arrays loaded from a file are read-only buffers; the returned copies can be changed."""
        buffer = np.frombuffer(np.arange(4, dtype="float64").tobytes(), dtype="float64")
        assert not buffer.flags.writeable
        m = Metadata.from_dict(raw, supporting_arrays={"data": {"latitudes": buffer}})
        m.latitudes[0] = 5.0
        assert m.latitudes[0] == 0.0


# ---------------------------------------------------------------------------
# Every golden file
# ---------------------------------------------------------------------------


class TestGoldenFiles:
    """Every golden file opens through the interface."""

    def test_every_golden_file_opens_through_the_interface(self, golden_metadata: tuple[str, dict[str, Any]]) -> None:
        """Every golden file gives checkpoint-level and per-dataset values; none has arrays without being given."""
        _name, raw = golden_metadata
        m = Metadata.from_dict(raw)

        assert isinstance(m.component_versions, dict)
        assert m.task.task_name
        assert m.sections.seeds is not None
        for dataset_name, view in m.datasets.items():
            assert view.name == dataset_name
            assert isinstance(view.data.variables(), list)
            assert view.arrays.latitudes is None
