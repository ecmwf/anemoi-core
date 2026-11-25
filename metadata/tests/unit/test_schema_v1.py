# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for reading the raw sections through the adapters.

Covers the checkpoint adapter's lenient reading of the raw sections (absent and
null sections, the dataloader layouts) and the unversioned
:class:`~anemoi.metadata.sections.SectionsReader` (provenance, timestamp,
seeds).  Per-format values are pinned in ``test_golden.py``.
"""

import copy
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from typing import Any

import pytest

from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.sections import SectionsReader


@pytest.fixture
def raw(golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """The ``f6_main`` golden dict (one dataset ``data``); a fresh copy per test."""
    return golden_loader("f6_main")


# ---------------------------------------------------------------------------
# Absent and null sections
# ---------------------------------------------------------------------------


class TestMissingKeys:
    """Missing keys never fail the read."""

    def test_raw_sections_default_to_empty(self, raw: dict[str, Any]) -> None:
        for key in ("config", "training", "dataset", "environment", "provenance", "provenance_training"):
            raw.pop(key, None)
        adapter = CheckpointAdapter(raw)
        assert adapter.sections.config == {}
        assert adapter.sections.dataset == {}
        assert adapter.sections.provenance == {}
        assert adapter.get_section("training") == {}
        assert adapter.get_section("environment") == {}
        assert adapter.get_section("not-a-section") is None


class TestNullSectionRobustness:
    """Sections set to None read as absent."""

    def test_null_config_groups_do_not_break_data_frequency(self, raw: dict[str, Any]) -> None:
        raw["config"].update({"task": None, "data": None, "dataloader": None})
        raw["dataset"]["data"]["frequency"] = "3h"
        adapter = CheckpointAdapter(raw)
        assert adapter.data("data").data_frequency() == "3h"
        assert adapter.data("data").dataloader_config() == {}

    @pytest.mark.parametrize("dataloader", [None, {"training": None}])
    def test_null_dataloader_or_partition_returns_empty_dict(
        self,
        raw: dict[str, Any],
        dataloader: dict[str, Any] | None,
    ) -> None:
        raw["config"]["dataloader"] = dataloader
        assert CheckpointAdapter(raw).data("data").dataloader_config("training") == {}


# ---------------------------------------------------------------------------
# dataset section
# ---------------------------------------------------------------------------


class TestDatasetSection:
    """Reading derived values never changes the raw ``dataset`` dump."""

    def test_constant_fields_do_not_modify_raw(self, raw: dict[str, Any]) -> None:
        assert "constant_in_time" not in raw["dataset"]["data"]["variables_metadata"]["2t"]
        raw["dataset"]["data"]["constant_fields"] = ["2t"]
        adapter = CheckpointAdapter(raw)
        assert adapter.data("data").variables_metadata()["2t"]["constant_in_time"] is True
        assert "constant_in_time" not in adapter.to_dict()["dataset"]["data"]["variables_metadata"]["2t"]


class TestDataloaderConfig:
    """dataloader_config handles multi-dataset nesting and dataset_config."""

    def test_dataset_config_unwrapped_with_start_end(self, raw: dict[str, Any]) -> None:
        """f6_main: ``dataset_config`` of the ``data`` entry, with the entry's ``end`` and its null values dropped."""
        assert CheckpointAdapter(raw).data("data").dataloader_config() == {
            "dataset": ".../aifs-ea-an-oper-0001-mars-o96-2017-2017-6h-v8-testing.zarr",
            "frequency": "6h",
            "end": "2017-01-08 12:00:00",
        }

    def test_per_dataset_entry(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        checkpoint = CheckpointAdapter(golden_loader("f3_multi_dataset"))
        assert checkpoint.data("era5").dataloader_config() == {
            "dataset": "aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1",
            "frequency": "6h",
            "drop": [],
            "start": 1985,
            "end": "2020-01-08 12:00:00",
        }
        assert checkpoint.data("cerra").dataloader_config("validation") == {
            "dataset": "cerra-rr-an-oper-0001-mars-0p2-2017-2017-6h-v1-testing",
            "frequency": "6h",
            "drop": [],
            "start": "2017-01-08 18:00:00",
            "end": 2021,
        }

    def test_plain_partition(self, raw: dict[str, Any]) -> None:
        raw["config"]["dataloader"] = {"training": {"dataset": "x", "start": 1979}}
        assert CheckpointAdapter(raw).data("data").dataloader_config() == {"dataset": "x", "start": 1979}

    def test_string_dataset_config(self, raw: dict[str, Any]) -> None:
        raw["config"]["dataloader"] = {"training": {"dataset_config": "path.zarr"}}
        assert CheckpointAdapter(raw).data("data").dataloader_config() == {"dataset": "path.zarr"}

    def test_does_not_modify_raw(self, raw: dict[str, Any]) -> None:
        before = copy.deepcopy(raw)
        adapter = CheckpointAdapter(raw)
        adapter.data("data").dataloader_config()
        assert adapter.to_dict() == before


# ---------------------------------------------------------------------------
# SectionsReader
# ---------------------------------------------------------------------------


class TestSectionsReader:
    """The unversioned reader of raw dumps."""

    def test_section_as_is(self) -> None:
        reader = SectionsReader({"tracker": {"a": 1}, "model": [1, 2]})
        assert reader.section("tracker") == {"a": 1}
        assert reader.section("model") == [1, 2]
        assert reader.section("missing") is None
        assert reader.section("missing", 5) == 5

    def test_provenance_prefers_provenance_training(self) -> None:
        reader = SectionsReader({"provenance_training": {"a": 1}, "provenance": {"b": 2}})
        assert reader.provenance == {"a": 1}

    def test_provenance_falls_back_to_provenance(self) -> None:
        assert SectionsReader({"provenance": {"b": 2}}).provenance == {"b": 2}

    def test_provenance_empty_training_falls_back(self) -> None:
        assert SectionsReader({"provenance_training": {}, "provenance": {"b": 2}}).provenance == {"b": 2}

    def test_provenance_absent(self) -> None:
        assert SectionsReader({}).provenance == {}

    def test_timestamp_parsed_from_iso_string(self) -> None:
        stamp = SectionsReader({"timestamp": "2026-10-21T10:30:00+00:00"}).timestamp
        assert stamp == datetime(2026, 10, 21, 10, 30, tzinfo=UTC)

    def test_timestamp_accepts_datetime(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        assert SectionsReader({"timestamp": now}).timestamp is now

    def test_timestamp_falls_back_to_created_at(self) -> None:
        reader = SectionsReader({"created_at": "2024-06-01T12:00:00+00:00"})
        assert reader.timestamp == datetime(2024, 6, 1, 12, tzinfo=UTC)

    def test_timestamp_prefers_timestamp(self) -> None:
        reader = SectionsReader({"timestamp": "2026-01-01T00:00:00", "created_at": "2024-01-01T00:00:00"})
        assert reader.timestamp.year == 2026

    def test_timestamp_unparseable_is_none(self) -> None:
        assert SectionsReader({"timestamp": "yesterday"}).timestamp is None
        assert SectionsReader({}).timestamp is None

    def test_timestamp_unparseable_falls_back_to_created_at(self) -> None:
        reader = SectionsReader({"timestamp": "yesterday", "created_at": "2024-06-01T12:00:00+00:00"})
        assert reader.timestamp == datetime(2024, 6, 1, 12, tzinfo=UTC)

    def test_seeds_fall_back_to_metadata_inference(self) -> None:
        reader = SectionsReader({"metadata_inference": {"seed": 3}, "base_seed": 4})
        assert reader.seeds == {"seed": 3, "base_seed": 4}

    def test_seeds_absent(self) -> None:
        assert SectionsReader({}).seeds == {}

    def test_run_id_top_level_then_inference(self) -> None:
        assert SectionsReader({"run_id": "a", "metadata_inference": {"run_id": "b"}}).run_id == "a"
        assert SectionsReader({"metadata_inference": {"run_id": "b"}}).run_id == "b"
        assert SectionsReader({}).run_id is None

    def test_precision(self) -> None:
        assert SectionsReader({"config": {"training": {"precision": "bf16-mixed"}}}).precision == "bf16-mixed"
        assert SectionsReader({"config": {"training": None}}).precision is None
        assert SectionsReader({}).precision is None

    def test_reader_does_not_modify_raw(self) -> None:
        raw = {"provenance": {"b": 2}, "timestamp": "2026-01-01T00:00:00"}
        before = copy.deepcopy(raw)
        reader = SectionsReader(raw)
        assert reader.provenance == {"b": 2}
        assert reader.timestamp == datetime.fromisoformat("2026-01-01T00:00:00")
        assert reader.seeds == {}
        assert raw == before
