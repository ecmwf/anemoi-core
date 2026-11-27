# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for base.py.

Covers the CheckpointAdapter: envelope
properties, composition of the component adapters, delegating getters,
lenient reading and unchanged ``to_dict()``.  The inputs are golden metadata
files (``tests/data``); per-format values are pinned in ``test_golden.py``.
"""

import copy
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from typing import Any

import pytest
from packaging.version import Version

from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.components.data import GriddedV0_2
from anemoi.metadata.components.data import GriddedV1_0
from anemoi.metadata.components.data import TabularV1_0
from anemoi.metadata.components.legacy import ForecasterV0_1
from anemoi.metadata.components.legacy import GriddedV0_1
from anemoi.metadata.components.task import ForecasterV0_2
from anemoi.metadata.components.task import ForecasterV1_0

#: Metadata with a ``metadata_inference`` block and nothing else.
_EMPTY = {"metadata_inference": {}}


@pytest.fixture
def raw(golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """The ``f6_main`` golden dict (version ``2.0``, one dataset ``data``); a fresh copy per test."""
    return golden_loader("f6_main")


@pytest.fixture
def adapter(raw: dict[str, Any]) -> CheckpointAdapter:
    """:class:`CheckpointAdapter` over ``raw``."""
    return CheckpointAdapter(raw)


# ---------------------------------------------------------------------------
# Envelope: version, parsed_version, component_versions
# ---------------------------------------------------------------------------


class TestEnvelope:
    """parsed_version, component_versions and created_at (``version`` is pinned in ``test_golden.py``)."""

    def test_parsed_version(self, adapter: CheckpointAdapter, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """parsed_version is a packaging Version that supports comparison."""
        assert adapter.parsed_version == Version("2.0")
        assert adapter.parsed_version < Version("3.0")
        assert CheckpointAdapter(golden_loader("f8_richer_batch")).parsed_version == Version("3.0")

    def test_parsed_version_none_for_invalid(self, raw: dict[str, Any]) -> None:
        """An unparseable version reads as None rather than failing."""
        raw["version"] = "not-a-version"
        assert CheckpointAdapter(raw).parsed_version is None

    def test_component_versions_returns_a_copy(self, adapter: CheckpointAdapter) -> None:
        """Mutating component_versions does not change the adapter."""
        adapter.component_versions["data"] = "9.9"
        assert adapter.component_versions == {"data": "0.2", "task": "0.2", "lineage": "1.0"}

    def test_created_at_is_the_timestamp(self, adapter: CheckpointAdapter) -> None:
        """created_at is the parsed ``timestamp`` of the file."""
        assert adapter.created_at == datetime(2026, 11, 23, 16, 11, 13, 640918, tzinfo=UTC)

    def test_created_at_falls_back_to_created_at_key(self, raw: dict[str, Any]) -> None:
        """Without ``timestamp`` the ``created_at`` key is read."""
        del raw["timestamp"]
        raw["created_at"] = "2024-06-01T12:00:00+00:00"
        assert CheckpointAdapter(raw).created_at == datetime(2024, 6, 1, 12, tzinfo=UTC)

    @pytest.mark.parametrize("name", ["version", "created_at", "component_versions"])
    def test_envelope_properties_are_read_only(self, adapter: CheckpointAdapter, name: str) -> None:
        """Envelope properties cannot be assigned."""
        with pytest.raises(AttributeError):
            setattr(adapter, name, "x")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    """The adapter composes one adapter per component plus the sections reader."""

    @pytest.mark.parametrize(
        ("name", "data_cls", "task_cls"),
        [
            ("f1_single_dataset", GriddedV0_1, ForecasterV0_1),
            ("f6_main", GriddedV0_2, ForecasterV0_2),
            ("f8_richer_batch", GriddedV1_0, ForecasterV1_0),
        ],
    )
    def test_adapters_selected_by_format(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        name: str,
        data_cls: type,
        task_cls: type,
    ) -> None:
        """The data adapter follows the layout and the data type; the task adapter the layout and the task name."""
        adapter = CheckpointAdapter(golden_loader(name))
        assert type(adapter.data("data")) is data_cls
        assert type(adapter.task) is task_cls

    def test_mixed_checkpoint_gets_one_data_adapter_per_data_type(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        """Data adapters are per dataset: a gridded and a tabular dataset get the adapter of their kind."""
        adapter = CheckpointAdapter(golden_loader("f8_richer_batch"))
        assert adapter.dataset_names == ["data", "obs"]
        assert type(adapter.data("data")) is GriddedV1_0
        assert type(adapter.data("obs")) is TabularV1_0

    def test_unknown_dataset_raises(self, adapter: CheckpointAdapter) -> None:
        with pytest.raises(KeyError, match="Available datasets"):
            adapter.data("nope")

    def test_unknown_data_type_reads_as_gridded_with_a_warning(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f8_richer_batch")
        raw["metadata_inference"]["obs"]["data_type"] = "point-cloud"
        with pytest.warns(UserWarning, match="unknown data_type 'point-cloud'"):
            adapter = CheckpointAdapter(raw)
        assert adapter.data("obs").data_type() == "gridded"

    def test_stamps_select_each_component_independently(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """A component_versions stamp selects that component's adapter only."""
        raw = golden_loader("f8_richer_batch")
        raw["version"] = "3.1"
        raw["component_versions"] = {"data": "1.0", "task": "0.2"}
        adapter = CheckpointAdapter(raw)
        assert type(adapter.data("data")) is GriddedV1_0
        assert type(adapter.task) is ForecasterV0_2

    def test_sections_reader_present(self, adapter: CheckpointAdapter) -> None:
        """The sections reader is composed and reads the raw dumps."""
        assert adapter.sections.config["task"]["multistep_input"] == 2

    def test_raw_returns_the_held_dict(self, adapter: CheckpointAdapter, raw: dict[str, Any]) -> None:
        """Raw exposes the held dictionary."""
        assert adapter.raw == raw

    @pytest.mark.parametrize(
        ("name", "legacy"),
        [("f0_ai_models", True), ("f1_single_dataset", True), ("f6_main", False)],
    )
    def test_is_legacy(self, golden_loader: Callable[[str], dict[str, Any]], name: str, legacy: bool) -> None:
        """A checkpoint is legacy exactly when it has no metadata_inference block."""
        assert CheckpointAdapter(golden_loader(name)).is_legacy is legacy

    def test_non_mapping_rejected(self) -> None:
        """Constructing from a non-mapping raises TypeError."""
        with pytest.raises(TypeError, match="mapping"):
            CheckpointAdapter([1, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Getters delegate to the components
# ---------------------------------------------------------------------------


class TestGetters:
    """The contract getters read the golden metadata through the components."""

    def test_variable_indices(self, adapter: CheckpointAdapter) -> None:
        assert adapter.data("data").variable_indices()["lsm"] == 9
        assert "lsm" not in adapter.data("data").output_variable_indices()
        assert adapter.data("data").output_variable_indices()["10u"] == 0

    def test_precision_read_from_config(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        assert CheckpointAdapter(golden_loader("f6_main")).sections.precision == "16-mixed"
        assert CheckpointAdapter(golden_loader("f1_single_dataset")).sections.precision == "bf16-mixed"

    def test_get_section(self, adapter: CheckpointAdapter, raw: dict[str, Any]) -> None:
        assert adapter.get_section("training") == {
            "current_epoch": 1,
            "global_step": 6,
            "elapsed_time": 85.70771217346191,
        }
        assert adapter.get_section("uuid") == raw["uuid"] == "11f0410c-d9d2-4255-8232-8cc6602dd1ab"
        assert adapter.get_section("environment") == {}
        assert adapter.get_section("absent") is None


# ---------------------------------------------------------------------------
# Lenient reading and unchanged to_dict()
# ---------------------------------------------------------------------------


class TestLenientRead:
    """Reading never fails on unknown or missing optional keys; to_dict() is unchanged."""

    def test_unknown_keys_preserved(self, raw: dict[str, Any]) -> None:
        raw["something_new"] = {"a": [1, 2]}
        raw["metadata_inference"]["data"]["future_key"] = 1
        assert CheckpointAdapter(raw).to_dict() == raw

    def test_to_dict_returns_a_copy(self, adapter: CheckpointAdapter) -> None:
        copied = adapter.to_dict()
        copied["config"]["task"]["multistep_input"] = 99
        assert adapter.get_section("config")["task"]["multistep_input"] == 2

    def test_input_dict_is_not_aliased(self, raw: dict[str, Any]) -> None:
        original = copy.deepcopy(raw)
        adapter = CheckpointAdapter(raw)
        raw["config"]["task"]["multistep_input"] = 99
        assert adapter.to_dict() == original

    def test_empty_dict_reads_without_error(self) -> None:
        adapter = CheckpointAdapter(_EMPTY)
        assert adapter.dataset_names == []
        assert adapter.task.task_name == "forecaster"
        assert adapter.sections.precision is None
        assert adapter.sections.provenance == {}
        assert adapter.created_at is None

    def test_empty_dataset_block_reads_from_the_other_sections(self, raw: dict[str, Any]) -> None:
        """A dataset block without data_indices, variable_types or timesteps reads as empty or from config."""
        raw["metadata_inference"]["data"] = {}
        adapter = CheckpointAdapter(raw)
        assert adapter.data("data").variable_indices() == {}
        assert adapter.data("data").output_variable_indices() == {}
        assert adapter.data("data").variable_types() == {
            "forcing": [],
            "prognostic": [],
            "diagnostic": [],
            "target": [],
        }
        # No recorded timing: the task falls back to config.task, the grid to the dataset shape.
        assert adapter.task.timestep() == "6H"
        assert adapter.task.input_relative_date_indices() == [0, 1]
        assert adapter.task.output_relative_date_indices() == [2]
        assert adapter.data("data").grid_size() == 40320

    def test_null_sections_do_not_fail(self, raw: dict[str, Any]) -> None:
        raw["config"] = None
        raw["dataset"] = None
        adapter = CheckpointAdapter(raw)
        assert adapter.sections.precision is None
        assert adapter.data("data").data_request() == {}
        assert adapter.data("data").data_frequency() is None
        assert adapter.data("data").dataloader_config() == {}
