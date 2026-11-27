# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the strict write schema of format 3.1.

Strict writes are checked with F9 dicts only (the F8 golden file stamped as
this package writes it).  Historical formats are read, never written.
"""

import copy
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import anemoi.metadata as anemoi_metadata
from anemoi.metadata import Metadata
from anemoi.metadata import save_metadata
from anemoi.metadata.exceptions import MetadataError
from anemoi.metadata.registry import ComponentRegistry
from anemoi.metadata.write_schema import COMPONENT_VERSIONS
from anemoi.metadata.write_schema import CURRENT_VERSION
from anemoi.metadata.write_schema import WriteValidationError
from anemoi.metadata.write_schema import register_write_schema
from anemoi.metadata.write_schema import validate_for_write
from anemoi.metadata.write_schema import write_schema_versions


@pytest.fixture
def f9(golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """An F9 dict: the F8 golden file (gridded ``data`` + tabular ``obs``) stamped as written."""
    metadata = golden_loader("f8_richer_batch")
    metadata["version"] = CURRENT_VERSION
    metadata["component_versions"] = dict(COMPONENT_VERSIONS)
    return metadata


def _rejected(metadata: dict[str, Any], match: str) -> WriteValidationError:
    with pytest.raises(WriteValidationError, match=match) as info:
        validate_for_write(metadata)
    return info.value


# ---------------------------------------------------------------------------
# Constants and exports
# ---------------------------------------------------------------------------


class TestConstants:
    def test_current_version(self) -> None:
        assert CURRENT_VERSION == "3.1"

    def test_component_versions(self) -> None:
        assert COMPONENT_VERSIONS == {"data": "1.0", "task": "1.0", "lineage": "1.0"}

    def test_each_written_version_has_a_schema_and_an_adapter(self) -> None:
        for component, version in COMPONENT_VERSIONS.items():
            assert version in write_schema_versions(component)
            assert ComponentRegistry.get(component, version, "gridded" if component == "data" else None) is not None

    def test_public_exports(self) -> None:
        assert anemoi_metadata.CURRENT_VERSION == CURRENT_VERSION
        assert anemoi_metadata.COMPONENT_VERSIONS == COMPONENT_VERSIONS
        assert anemoi_metadata.validate_for_write is validate_for_write
        assert anemoi_metadata.WriteValidationError is WriteValidationError

    def test_error_is_a_metadata_error(self) -> None:
        assert issubclass(WriteValidationError, MetadataError)

    def test_duplicate_schema_registration_rejected(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_write_schema("data", "1.0")(lambda _metadata, _errors: None)


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


class TestValid:
    def test_f9_is_valid_and_returned_as_new_dict(self, f9: dict[str, Any]) -> None:
        original = copy.deepcopy(f9)
        result = validate_for_write(f9)
        assert result == original
        assert result is not f9
        assert result["metadata_inference"] is not f9["metadata_inference"]

    def test_input_is_untouched_when_restamped(self, f9: dict[str, Any]) -> None:
        f9["version"] = "3.0"
        before = copy.deepcopy(f9)
        result = validate_for_write(f9)
        assert f9 == before
        assert result["version"] == CURRENT_VERSION

    @pytest.mark.parametrize("version", [None, "3.0", "3.1"])
    def test_version_is_set_to_current(self, f9: dict[str, Any], version: str | None) -> None:
        if version is None:
            del f9["version"]
        else:
            f9["version"] = version
        assert validate_for_write(f9)["version"] == CURRENT_VERSION

    def test_unknown_raw_sections_and_keys_are_kept(self, f9: dict[str, Any]) -> None:
        f9["future_section"] = {"anything": [1, 2]}
        f9["config"] = {"free": {"form": True}}
        assert validate_for_write(f9)["future_section"] == {"anything": [1, 2]}

    def test_raw_dumps_are_not_inspected(self, f9: dict[str, Any]) -> None:
        for key in ("provenance_training", "training", "tracker", "model", "timestamp"):
            f9.pop(key, None)
        f9["dataset"]["data"] = {"anything": "goes"}
        validate_for_write(f9)

    def test_gridded_only(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["dataset_names"] = ["data"]
        del f9["metadata_inference"]["obs"]
        del f9["dataset"]["obs"]
        validate_for_write(f9)

    def test_tabular_only(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["dataset_names"] = ["obs"]
        del f9["metadata_inference"]["data"]
        del f9["dataset"]["data"]
        validate_for_write(f9)

    def test_offset_forecaster(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["task"] = "offset-forecaster"
        for name in ("data", "obs"):
            f9["metadata_inference"][name]["timesteps"].update(
                input_offsets=["-6h", "0h"],
                output_offsets=["6h"],
                rollout_shift="6h",
                advance_map={"inin": [[1, 0]], "outin": [[0, 1]]},
            )
        validate_for_write(f9)

    def test_round_trips_through_save_and_from_checkpoint(
        self,
        empty_ckpt: Callable[..., Path],
        f9: dict[str, Any],
    ) -> None:
        ckpt = empty_ckpt("f9.ckpt")
        save_metadata(ckpt, f9)
        metadata = Metadata.from_checkpoint(ckpt)
        assert metadata.version == CURRENT_VERSION
        assert metadata.component_versions == COMPONENT_VERSIONS
        assert metadata.dataset_names == ["data", "obs"]
        assert metadata.dataset("data").data.data_type() == "gridded"
        assert metadata.dataset("data").data.grid_size() == 40320
        assert metadata.dataset("obs").data.data_type() == "tabular"
        assert metadata.dataset("obs").data.grid_size() is None
        assert metadata.task.timestep() == "6h"


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestInvalid:
    def test_not_a_dict(self) -> None:
        with pytest.raises(TypeError, match="dict"):
            validate_for_write("metadata")  # type: ignore[arg-type]

    def test_missing_dataset_names(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["dataset_names"]
        error = _rejected(f9, "dataset_names")
        assert any("metadata_inference.dataset_names" in line for line in error.errors)

    @pytest.mark.parametrize("names", [[], "data", ["data", "data"], ["data", 3]])
    def test_bad_dataset_names(self, f9: dict[str, Any], names: Any) -> None:
        f9["metadata_inference"]["dataset_names"] = names
        _rejected(f9, "dataset_names")

    def test_missing_metadata_inference(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]
        _rejected(f9, "metadata_inference")

    def test_dataset_block_missing(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["obs"]
        _rejected(f9, "metadata_inference.obs")

    def test_data_type_image(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["data"]["data_type"] = "image"
        _rejected(f9, "data_type")

    def test_data_type_missing(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["data"]["data_type"]
        _rejected(f9, "data_type")

    def test_grid_size_missing_on_gridded(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["data"]["grid_size"]
        _rejected(f9, "grid_size")

    def test_grid_size_null_on_gridded(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["data"]["grid_size"] = None
        _rejected(f9, "grid_size is required for a gridded dataset")

    @pytest.mark.parametrize("value", [0, -5, 1.5, "40320", True])
    def test_grid_size_wrong_value(self, f9: dict[str, Any], value: Any) -> None:
        f9["metadata_inference"]["data"]["grid_size"] = value
        _rejected(f9, "grid_size")

    def test_grid_size_on_tabular(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["obs"]["grid_size"] = 10
        _rejected(f9, "grid_size must be null for a tabular dataset")

    def test_data_indices_missing_output(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["data"]["data_indices"]["output"]
        _rejected(f9, "data_indices.output")

    def test_data_indices_duplicate(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["obs"]["data_indices"]["input"]["msl"] = 0
        _rejected(f9, "input indices must be unique")

    def test_data_indices_not_int(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["obs"]["data_indices"]["input"]["msl"] = "2"
        _rejected(f9, "data_indices.input")

    def test_variable_types_missing_key(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["obs"]["variable_types"]["target"]
        _rejected(f9, "variable_types.target")

    def test_variable_type_not_in_indices(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["obs"]["variable_types"]["forcing"].append("ghost")
        _rejected(f9, "ghost")

    def test_unknown_key_in_dataset_block(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["data"]["shapes"] = {"grid": 1}
        _rejected(f9, "shapes")

    def test_task_missing(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["task"]
        _rejected(f9, "metadata_inference.task")

    def test_task_not_a_string(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["task"] = 3
        _rejected(f9, "metadata_inference.task")

    def test_timesteps_missing(self, f9: dict[str, Any]) -> None:
        del f9["metadata_inference"]["obs"]["timesteps"]
        _rejected(f9, "metadata_inference.obs.timesteps")

    def test_timestep_unparsable(self, f9: dict[str, Any]) -> None:
        for name in ("data", "obs"):
            f9["metadata_inference"][name]["timesteps"]["timestep"] = "soon"
        _rejected(f9, "not a valid time step")

    def test_empty_input_indices(self, f9: dict[str, Any]) -> None:
        for name in ("data", "obs"):
            f9["metadata_inference"][name]["timesteps"]["input_relative_date_indices"] = []
        _rejected(f9, "input_relative_date_indices")

    def test_timesteps_differ_between_datasets(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["obs"]["timesteps"]["timestep"] = "12h"
        _rejected(f9, "checkpoint level")

    def test_offset_forecaster_needs_offsets(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["task"] = "offset-forecaster"
        error = _rejected(f9, "rollout_shift")
        assert any("advance_map" in line for line in error.errors)

    @pytest.mark.parametrize("kind", ["input", "output"])
    def test_indices_must_be_in_training_indices(self, f9: dict[str, Any], kind: str) -> None:
        for name in ("data", "obs"):
            f9["metadata_inference"][name]["timesteps"][f"{kind}_relative_date_indices"] = [0, 7]
        error = _rejected(f9, "not in relative_date_indices_training")
        assert any(f"{kind}_relative_date_indices" in line for line in error.errors)

    def test_repeated_training_indices_are_a_set(self, f9: dict[str, Any]) -> None:
        for name in ("data", "obs"):
            f9["metadata_inference"][name]["timesteps"]["relative_date_indices_training"] = [0, 1, 2, 2]
        validate_for_write(f9)

    @pytest.mark.parametrize("kind", ["input", "output"])
    def test_offsets_must_match_indices(self, f9: dict[str, Any], kind: str) -> None:
        f9["metadata_inference"]["task"] = "offset-forecaster"
        for name in ("data", "obs"):
            f9["metadata_inference"][name]["timesteps"].update(
                input_offsets=["-6h", "0h"],
                output_offsets=["6h"],
                rollout_shift="6h",
                advance_map={"inin": [[1, 0]], "outin": [[0, 1]]},
            )
            f9["metadata_inference"][name]["timesteps"][f"{kind}_offsets"].append("12h")
        _rejected(f9, f"{kind}_offsets: .* offsets for")

    def test_offsets_rejected_for_other_tasks(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["data"]["timesteps"]["rollout_shift"] = "6h"
        _rejected(f9, "rollout_shift")

    def test_config_missing(self, f9: dict[str, Any]) -> None:
        del f9["config"]
        _rejected(f9, "config")

    def test_dataset_dump_missing(self, f9: dict[str, Any]) -> None:
        del f9["dataset"]
        _rejected(f9, "dataset: required")

    def test_dataset_dump_entry_missing(self, f9: dict[str, Any]) -> None:
        del f9["dataset"]["obs"]
        _rejected(f9, "dataset.obs")

    def test_all_problems_are_reported_together(self, f9: dict[str, Any]) -> None:
        f9["metadata_inference"]["data"]["data_type"] = "image"
        del f9["config"]
        error = _rejected(f9, "data_type")
        assert len(error.errors) >= 2
        assert "config" in str(error)


# ---------------------------------------------------------------------------
# lineage 1.0
# ---------------------------------------------------------------------------


class TestLineage:
    def test_fresh_run_and_warm_start_are_valid(self, f9: dict[str, Any]) -> None:
        validate_for_write(f9)
        f9["checkpoint_lineage"] = [{"loading_strategy": "warm_start", "parent_uuid": "0dc0e962"}]
        f9["parent_uuid"] = "0dc0e962"
        assert validate_for_write(f9)["parent_uuid"] == "0dc0e962"

    def test_parent_uuid_may_be_none_for_a_parent_without_uuid(self, f9: dict[str, Any]) -> None:
        f9["checkpoint_lineage"] = [{"loading_strategy": "cold_start", "parent_uuid": None}]
        f9["parent_uuid"] = None
        validate_for_write(f9)

    @pytest.mark.parametrize("key", ["uuid", "checkpoint_lineage"])
    def test_required(self, f9: dict[str, Any], key: str) -> None:
        del f9[key]
        _rejected(f9, f"lineage.{key}")

    @pytest.mark.parametrize("uuid", ["", None, 3])
    def test_uuid_must_be_a_non_empty_string(self, f9: dict[str, Any], uuid: Any) -> None:
        f9["uuid"] = uuid
        _rejected(f9, "lineage.uuid")

    def test_entries_are_checked(self, f9: dict[str, Any]) -> None:
        f9["checkpoint_lineage"] = [{"loading_strategy": "warm_start"}]
        _rejected(f9, r"lineage.checkpoint_lineage.0.parent_uuid")
        f9["checkpoint_lineage"] = [{"loading_strategy": "warm_start", "parent_uuid": "a", "extra": 1}]
        _rejected(f9, r"lineage.checkpoint_lineage.0.extra")
        f9["checkpoint_lineage"] = "none"
        _rejected(f9, "lineage.checkpoint_lineage")

    def test_omitted_or_none_parent_uuid_contradicting_the_last_entry_is_rejected(self, f9: dict[str, Any]) -> None:
        f9["checkpoint_lineage"] = [{"loading_strategy": "warm_start", "parent_uuid": "parent"}]
        f9.pop("parent_uuid", None)
        _rejected(f9, "differs from the last entry")
        f9["parent_uuid"] = None
        _rejected(f9, "differs from the last entry")

    def test_omitted_parent_uuid_is_fine_when_the_last_entry_has_no_parent(self, f9: dict[str, Any]) -> None:
        f9["checkpoint_lineage"] = [{"loading_strategy": "cold_start", "parent_uuid": None}]
        f9.pop("parent_uuid", None)
        validate_for_write(f9)

    def test_parent_uuid_must_match_the_last_entry(self, f9: dict[str, Any]) -> None:
        f9["checkpoint_lineage"] = [{"loading_strategy": "warm_start", "parent_uuid": "a"}]
        f9["parent_uuid"] = "b"
        _rejected(f9, "differs from the last entry")
        f9["checkpoint_lineage"] = []
        f9["parent_uuid"] = "a"
        _rejected(f9, "differs from the last entry")


# ---------------------------------------------------------------------------
# Versions and stamps
# ---------------------------------------------------------------------------


class TestVersionsAndStamps:
    def test_component_versions_required(self, f9: dict[str, Any]) -> None:
        del f9["component_versions"]
        _rejected(f9, "component_versions")

    def test_component_versions_not_a_dict(self, f9: dict[str, Any]) -> None:
        f9["component_versions"] = ["data", "task"]
        _rejected(f9, "component_versions")

    @pytest.mark.parametrize("component", ["data", "task", "lineage"])
    def test_stamp_missing(self, f9: dict[str, Any], component: str) -> None:
        del f9["component_versions"][component]
        _rejected(f9, f"component_versions.{component}: required")

    @pytest.mark.parametrize(
        ("component", "stamp"),
        [("task", "9.9"), ("data", "0.2"), ("data", "2.0"), ("task", 1.0), ("lineage", "0.2"), ("lineage", "2.0")],
    )
    def test_unknown_stamp(self, f9: dict[str, Any], component: str, stamp: str | float) -> None:
        f9["component_versions"][component] = stamp
        _rejected(f9, f"component_versions.{component}")

    def test_unknown_component_stamp(self, f9: dict[str, Any]) -> None:
        f9["component_versions"]["model"] = "1.0"
        _rejected(f9, "component_versions.model: unknown component")

    @pytest.mark.parametrize("version", ["2.0", "1.0", "3.2", "4.0", "3", 3, 3.1])
    def test_existing_version_not_known_is_rejected_not_restamped(
        self,
        f9: dict[str, Any],
        version: str | float,
    ) -> None:
        f9["version"] = version
        _rejected(f9, "version")

    def test_historical_layout_is_never_relabelled(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        for name in ("f6_main", "f3_multi_dataset", "f0_ai_models"):
            metadata = golden_loader(name)
            with pytest.raises(WriteValidationError):
                validate_for_write(metadata)

    def test_historical_layout_with_stamps_added_is_still_rejected_by_version(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        metadata = golden_loader("f6_main")
        metadata["component_versions"] = dict(COMPONENT_VERSIONS)
        _rejected(metadata, "version")

    def test_future_schema_does_not_touch_others(self, f9: dict[str, Any]) -> None:
        """A schema registered for a new data version is used for that stamp only."""
        from anemoi.metadata import write_schema

        calls = []

        def schema(metadata: Mapping[str, Any], _errors: list[str]) -> None:
            calls.append(metadata["component_versions"]["data"])

        saved = dict(write_schema._SCHEMAS)
        try:
            register_write_schema("data", "1.1")(schema)
            f9["component_versions"]["data"] = "1.1"
            validate_for_write(f9)
            assert calls == ["1.1"]
            assert write_schema_versions("data") == ["1.0", "1.1"]
            assert write_schema_versions("task") == ["1.0"]
        finally:
            write_schema._SCHEMAS.clear()
            write_schema._SCHEMAS.update(saved)


# ---------------------------------------------------------------------------
# Reading stays lenient
# ---------------------------------------------------------------------------


class TestReadingIsLenient:
    def test_invalid_f9_still_reads(self, f9: dict[str, Any]) -> None:
        """Content the strict writer rejects is still readable."""
        f9["metadata_inference"]["data"]["data_type"] = "image"
        del f9["config"]
        metadata = Metadata.from_dict(f9)
        assert metadata.dataset_names == ["data", "obs"]
