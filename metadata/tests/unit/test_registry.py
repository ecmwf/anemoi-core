# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the component registry, format detection and adapter loading.

Tests that register adapters use the ``registry_state`` fixture, which
isolates the global registry with ``ComponentRegistry.snapshot`` and
``ComponentRegistry.restore``.
"""

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata import formats
from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.components.base import ComponentAdapter
from anemoi.metadata.components.data import DataReaderV1_0
from anemoi.metadata.components.data import GriddedV0_2
from anemoi.metadata.components.data import GriddedV1_0
from anemoi.metadata.components.data import TabularV1_0
from anemoi.metadata.components.data import register_data_version
from anemoi.metadata.components.legacy import ForecasterV0_1
from anemoi.metadata.components.legacy import GriddedV0_1
from anemoi.metadata.components.legacy import TaskV0_1
from anemoi.metadata.components.task import TaskV0_2
from anemoi.metadata.components.task import TaskV1_0
from anemoi.metadata.exceptions import UnknownVersionError
from anemoi.metadata.registry import ComponentRegistry


@pytest.fixture
def generic_tasks_only(registry_state: None) -> None:
    """Remove the named task adapters from the registry (restored afterwards by ``registry_state``).

    Tests of the generic fallback and of registering a named adapter need a registry
    in which no task name has an adapter yet.
    """
    del registry_state  # Requested for its teardown, which restores the registry.
    for key in [key for key in ComponentRegistry._adapters if key[0] == "task" and key[1] is not None]:
        del ComponentRegistry._adapters[key]


# ---------------------------------------------------------------------------
# Registered adapters
# ---------------------------------------------------------------------------


class TestRegisteredAdapters:
    """Importing the package registers the data and task adapters."""

    def test_data_versions(self) -> None:
        assert ComponentRegistry.versions("data", "gridded") == ["0.1", "0.2", "1.0"]
        assert ComponentRegistry.versions("data", "tabular") == ["0.2", "1.0"]

    def test_data_adapters_are_registered_for_every_data_type(self) -> None:
        assert ComponentRegistry.names("data") == ["gridded", "tabular"]

    def test_task_versions(self) -> None:
        assert ComponentRegistry.versions("task") == ["0.1", "0.2", "1.0"]

    def test_legacy_adapters_are_registered_at_0_1(self) -> None:
        assert ComponentRegistry.get("data", "0.1", "gridded") is GriddedV0_1
        assert ComponentRegistry.get("task", "0.1") is TaskV0_1
        assert ComponentRegistry.get("task", "0.1", "forecaster") is ForecasterV0_1

    def test_get_exact(self) -> None:
        assert ComponentRegistry.get("data", "0.2", "gridded") is GriddedV0_2
        assert ComponentRegistry.get("data", "1.0", "gridded") is GriddedV1_0
        assert ComponentRegistry.get("data", "1.0", "tabular") is TabularV1_0
        assert ComponentRegistry.get("task", "0.2") is TaskV0_2
        assert ComponentRegistry.get("task", "1.0") is TaskV1_0

    def test_registration_records_key_on_class(self) -> None:
        assert (GriddedV1_0.component, GriddedV1_0.version, GriddedV1_0.name) == ("data", "1.0", "gridded")
        assert (TabularV1_0.component, TabularV1_0.version, TabularV1_0.name) == ("data", "1.0", "tabular")

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(UnknownVersionError):
            ComponentRegistry.get("data", "99.0", "gridded")

    def test_newest_version(self) -> None:
        assert ComponentRegistry.newest_version("data", "gridded") == "1.0"

    def test_newest_version_unknown_component_raises(self) -> None:
        with pytest.raises(UnknownVersionError):
            ComponentRegistry.newest_version("nonexistent")

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(RuntimeError):
            ComponentRegistry()


# ---------------------------------------------------------------------------
# Registration, task names and snapshot / restore
# ---------------------------------------------------------------------------


class TestRegistration:
    """The register decorator, keyed by (component, version) and (task name, version)."""

    @pytest.mark.usefixtures("registry_state")
    def test_register_new_version_subclassing_nearest(self) -> None:
        @ComponentRegistry.register("data", "1.1", name="gridded")
        class GriddedMinor1(GriddedV1_0):
            pass

        assert ComponentRegistry.get("data", "1.1", "gridded") is GriddedMinor1
        assert ComponentRegistry.versions("data", "gridded") == ["0.1", "0.2", "1.0", "1.1"]
        assert ComponentRegistry.newest_version("data", "gridded") == "1.1"

    @pytest.mark.usefixtures("registry_state")
    def test_versions_sorted_by_semver_not_text(self) -> None:
        @ComponentRegistry.register("data", "1.10", name="gridded")
        class GriddedMinor10(GriddedV1_0):
            pass

        @ComponentRegistry.register("data", "1.9", name="gridded")
        class GriddedMinor9(GriddedV1_0):
            pass

        assert ComponentRegistry.versions("data", "gridded") == ["0.1", "0.2", "1.0", "1.9", "1.10"]

    @pytest.mark.usefixtures("registry_state")
    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @ComponentRegistry.register("data", "1.0", name="gridded")
            class Duplicate(GriddedV1_0):
                pass

    @pytest.mark.usefixtures("registry_state")
    def test_invalid_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid component version"):
            ComponentRegistry.register("data", "one")

    @pytest.mark.usefixtures("generic_tasks_only")
    def test_named_task_adapter(self) -> None:
        @ComponentRegistry.register("task", "0.2", name="forecaster")
        class Forecaster(TaskV0_2):
            pass

        assert ComponentRegistry.get("task", "0.2", "forecaster") is Forecaster
        assert ComponentRegistry.names("task") == ["forecaster"]
        assert ComponentRegistry.versions("task", "forecaster") == ["0.2"]
        assert ComponentRegistry.versions("task") == ["0.1", "0.2", "1.0"]
        assert Forecaster.name == "forecaster"

    def test_named_task_adapters_are_registered_for_every_task(self) -> None:
        assert ComponentRegistry.names("task") == [
            "autoencoder",
            "forecaster",
            "offset-forecaster",
            "temporal-downscaler",
            "timeless",
        ]

    def test_snapshot_restore_isolates_registrations(self) -> None:
        state = ComponentRegistry.snapshot()
        try:

            @ComponentRegistry.register("data", "7.0", name="gridded")
            class Data7(GriddedV1_0):
                pass

            assert "7.0" in ComponentRegistry.versions("data", "gridded")
        finally:
            ComponentRegistry.restore(state)
        assert "7.0" not in ComponentRegistry.versions("data", "gridded")

    def test_snapshot_is_a_copy(self) -> None:
        state = ComponentRegistry.snapshot()
        state.clear()
        assert ComponentRegistry.versions("data", "gridded") == ["0.1", "0.2", "1.0"]

    @pytest.mark.usefixtures("registry_state")
    def test_registry_state_fixture_restores(self) -> None:
        """Registrations (and removals) of the previous registry_state tests did not leak."""
        assert ComponentRegistry.versions("data", "gridded") == ["0.1", "0.2", "1.0"]
        assert "forecaster" in ComponentRegistry.names("task")


# ---------------------------------------------------------------------------
# resolve(): fallbacks
# ---------------------------------------------------------------------------


class TestResolve:
    """resolve() picks the nearest older adapter and warns for unknown versions."""

    def test_exact_match_is_silent(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ComponentRegistry.resolve("data", "1.0", name="gridded") is GriddedV1_0

    def test_future_version_falls_back_to_newest_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="Unknown data version '9.0'"):
            assert ComponentRegistry.resolve("data", "9.0", name="gridded") is GriddedV1_0

    @pytest.mark.usefixtures("registry_state")
    def test_between_versions_uses_nearest_older_silently(self) -> None:
        @ComponentRegistry.register("data", "2.0", name="gridded")
        class Data2(GriddedV1_0):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ComponentRegistry.resolve("data", "1.5", name="gridded") is GriddedV1_0

    def test_legacy_version_is_an_exact_match(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ComponentRegistry.resolve("data", "0.1", name="gridded") is GriddedV0_1
            assert ComponentRegistry.resolve("task", "0.1", name="forecaster") is ForecasterV0_1

    def test_older_than_any_adapter_warns_and_uses_oldest(self) -> None:
        with pytest.warns(UserWarning, match="older than any known adapter"):
            assert ComponentRegistry.resolve("data", "0.0.1", name="gridded") is GriddedV0_1

    def test_unusable_version_warns_and_uses_newest(self) -> None:
        with pytest.warns(UserWarning, match="Unusable data version"):
            assert ComponentRegistry.resolve("data", "garbage", name="gridded") is GriddedV1_0

    def test_none_version_warns_and_uses_newest(self) -> None:
        with pytest.warns(UserWarning, match="Unusable"):
            assert ComponentRegistry.resolve("task", None) is TaskV1_0

    @pytest.mark.usefixtures("generic_tasks_only")
    def test_unknown_task_name_is_silent_without_named_adapters(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ComponentRegistry.resolve("task", "0.2", name="medium-range") is TaskV0_2

    @pytest.mark.usefixtures("generic_tasks_only")
    def test_unknown_task_name_warns_once_named_adapters_exist(self) -> None:
        @ComponentRegistry.register("task", "0.2", name="forecaster")
        class Forecaster(TaskV0_2):
            pass

        with pytest.warns(UserWarning, match="No task adapter for 'other'"):
            assert ComponentRegistry.resolve("task", "0.2", name="other") is TaskV0_2

    @pytest.mark.usefixtures("generic_tasks_only")
    def test_named_task_adapter_selected_by_name_and_version(self) -> None:
        @ComponentRegistry.register("task", "0.2", name="forecaster")
        class Forecaster(TaskV0_2):
            pass

        @ComponentRegistry.register("task", "1.0", name="forecaster")
        class Forecaster1(Forecaster):
            pass

        assert ComponentRegistry.resolve("task", "0.2", name="forecaster") is Forecaster
        assert ComponentRegistry.resolve("task", "1.0", name="forecaster") is Forecaster1
        assert ComponentRegistry.resolve("task", "1.0") is TaskV1_0

    def test_resolve_unknown_component_raises(self) -> None:
        with pytest.raises(UnknownVersionError):
            ComponentRegistry.resolve("nonexistent", "1.0")


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _with_inference(version: str | None = None, **extra: Any) -> dict[str, Any]:
    raw = {"metadata_inference": {"dataset_names": ["ds"], "ds": {}}}
    if version is not None:
        raw["version"] = version
    raw.update(extra)
    return raw


class TestFormatDetection:
    """component_versions(): stamps first, otherwise inferred from version and content."""

    def test_legacy_without_metadata_inference(self) -> None:
        assert formats.component_versions({}) == {"data": "0.1", "task": "0.1", "lineage": "0.1"}
        assert formats.component_versions({"version": "1.0"}) == {"data": "0.1", "task": "0.1", "lineage": "0.1"}

    def test_is_legacy(self) -> None:
        assert formats.is_legacy({"version": "2.0"})
        assert not formats.is_legacy(_with_inference("2.0"))

    @pytest.mark.parametrize("version", [None, "1.0", "2.0"])
    def test_2_0_layout(self, version: str | None) -> None:
        assert formats.component_versions(_with_inference(version)) == {"data": "0.2", "task": "0.2", "lineage": "0.1"}

    @pytest.mark.parametrize("version", ["3.0", "3.1"])
    def test_3_x_layout(self, version: str) -> None:
        assert formats.component_versions(_with_inference(version)) == {"data": "1.0", "task": "1.0", "lineage": "0.1"}

    def test_absent_version_with_layout_3_content(self) -> None:
        raw = _with_inference()
        raw["metadata_inference"]["ds"]["grid_size"] = 10
        assert formats.component_versions(raw)["data"] == "1.0"

    @pytest.mark.parametrize("version", [None, "2.0", "3.1"])
    def test_lineage_version_follows_the_content_not_the_version(self, version: str | None) -> None:
        assert formats.component_versions(_with_inference(version, uuid="u"))["lineage"] == "0.2"
        assert formats.component_versions(_with_inference(version, parent_uuid="p"))["lineage"] == "0.2"
        assert formats.component_versions(_with_inference(version, uuid="u", checkpoint_lineage=[]))["lineage"] == "1.0"
        assert formats.component_versions(_with_inference(version, checkpoint_lineage="bad"))["lineage"] == "0.1"

    def test_lineage_of_a_legacy_file_with_a_uuid(self) -> None:
        assert formats.component_versions({"version": "1.0", "uuid": "u"}) == {
            "data": "0.1",
            "task": "0.1",
            "lineage": "0.2",
        }

    def test_invalid_version_string_treated_as_absent(self) -> None:
        assert formats.component_versions(_with_inference("abc"))["data"] == "0.2"

    def test_stamps_win_over_version(self) -> None:
        raw = _with_inference("3.1", component_versions={"data": "1.0", "task": "0.2"})
        assert formats.component_versions(raw) == {"data": "1.0", "task": "0.2", "lineage": "0.1"}

    def test_partial_stamps_infer_the_rest(self) -> None:
        raw = _with_inference("2.0", component_versions={"task": "1.0"})
        assert formats.component_versions(raw) == {"data": "0.2", "task": "1.0", "lineage": "0.1"}

    def test_unknown_components_in_stamps_ignored(self) -> None:
        raw = _with_inference("3.1", component_versions={"model": "1.0", "data": "1.0", "task": 3})
        assert formats.stamped_versions(raw) == {"data": "1.0"}
        assert formats.component_versions(raw) == {"data": "1.0", "task": "1.0", "lineage": "0.1"}

    def test_future_version_uses_newest_known_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="Unknown metadata version '4.0'"):
            result = formats.component_versions(_with_inference("4.0"))
        assert result == {"data": "1.0", "task": "1.0", "lineage": "0.1"}

    def test_future_version_without_metadata_inference_warns(self) -> None:
        with pytest.warns(UserWarning, match="Unknown metadata version '4.0' without metadata_inference"):
            result = formats.component_versions({"version": "4.0"})
        assert result == {"data": "0.1", "task": "0.1", "lineage": "0.1"}

    @pytest.mark.parametrize("version", [None, "1.0", "2.0", "3.0", "3.1", "abc"])
    def test_known_version_without_metadata_inference_does_not_warn(self, version: str | None) -> None:
        raw = {} if version is None else {"version": version}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert formats.component_versions(raw) == {"data": "0.1", "task": "0.1", "lineage": "0.1"}

    def test_future_version_with_stamps_does_not_warn(self) -> None:
        raw = _with_inference("4.0", component_versions={"data": "1.0", "task": "1.0"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert formats.component_versions(raw) == {"data": "1.0", "task": "1.0", "lineage": "0.1"}

    def test_parse_version(self) -> None:
        assert str(formats.parse_version("3.1")) == "3.1"
        assert formats.parse_version(None) is None
        assert formats.parse_version(3) is None
        assert formats.parse_version("x") is None


# ---------------------------------------------------------------------------
# CheckpointAdapter: component selection
# ---------------------------------------------------------------------------


class TestCheckpointAdapterSelection:
    """CheckpointAdapter selects the component adapters from the stamps and the layout."""

    def test_missing_version_does_not_warn(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f6_main")
        del raw["version"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            adapter = CheckpointAdapter(raw)
        assert adapter.version is None
        assert adapter.component_versions == {"data": "0.2", "task": "0.2", "lineage": "1.0"}

    def test_unknown_future_version_loads_with_warning(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f8_richer_batch")
        raw["version"] = "9.0"
        with pytest.warns(UserWarning, match="Unknown metadata version"):
            adapter = CheckpointAdapter(raw)
        assert type(adapter.data("data")) is GriddedV1_0
        assert type(adapter.data("obs")) is TabularV1_0
        assert adapter.dataset_names == ["data", "obs"]

    def test_unknown_future_stamp_loads_with_warning(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f8_richer_batch")
        raw["version"] = "3.1"
        raw["component_versions"] = {"data": "5.0", "task": "1.0"}
        with pytest.warns(UserWarning, match="Unknown data version '5.0'"):
            adapter = CheckpointAdapter(raw)
        assert type(adapter.data("data")) is GriddedV1_0
        assert type(adapter.data("obs")) is TabularV1_0
        assert adapter.component_versions["data"] == "5.0"

    @pytest.mark.usefixtures("registry_state")
    def test_new_data_version_reaches_gridded_and_tabular_datasets(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        """A test-only reader registered with ``register_data_version`` serves every data type."""

        class ShoutingReader(DataReaderV1_0):
            def variable_indices(self) -> dict[str, int]:
                return {name.upper(): index for name, index in super().variable_indices().items()}

        register_data_version("1.1", ShoutingReader)

        raw = golden_loader("f8_richer_batch")
        raw["version"] = "3.1"
        raw["component_versions"] = {"data": "1.1", "task": "1.0"}
        adapter = CheckpointAdapter(raw)
        gridded, tabular = adapter.data("data"), adapter.data("obs")

        # The version's reading change reaches both kinds of dataset ...
        for data in (gridded, tabular):
            assert data.variable_indices()
            assert all(name.isupper() for name in data.variable_indices())
        # ... while each keeps the behaviour of its data type.
        assert (gridded.data_type(), tabular.data_type()) == ("gridded", "tabular")
        assert gridded.grid_size() == 40320
        assert tabular.grid_size() is None
        assert tabular.data_request() is None

    @pytest.mark.usefixtures("registry_state")
    def test_registered_task_adapter_is_selected_by_name(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        @ComponentRegistry.register("task", "0.2", name="brand-new-task")
        class BrandNew(TaskV0_2):
            def timestep(self, dataset_name: str | None = None) -> str:
                del dataset_name
                return "overridden"

        raw = golden_loader("f6_main")
        raw["metadata_inference"]["task"] = "brand-new-task"
        adapter = CheckpointAdapter(raw)
        assert type(adapter.task) is BrandNew
        assert adapter.task.timestep() == "overridden"


class TestComponentAdapterBase:
    """The component adapter base class."""

    def test_resolve_dataset_default_is_first(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = ComponentAdapter(golden_loader("f3_multi_dataset"))
        assert adapter.resolve_dataset() == "era5"
        assert adapter.resolve_dataset("cerra") == "cerra"

    def test_resolve_dataset_unknown_raises(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        with pytest.raises(KeyError, match="Available datasets"):
            ComponentAdapter(golden_loader("f6_main")).resolve_dataset("x")
