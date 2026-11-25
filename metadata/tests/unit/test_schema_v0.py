# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the legacy formats F0 and F1 (no ``metadata_inference``).

Legacy checkpoints are read by the same checkpoint adapter as every other
format, through the legacy component adapters (component version ``0.1``).
The first part covers null-section handling on a minimal dict, the second the
golden files: the repairs ported from anemoi-inference ``legacy.py`` and the
nested ``data_request`` of real F1 checkpoints.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.components.legacy import ForecasterV0_1
from anemoi.metadata.components.legacy import GriddedV0_1

LEGACY_GOLDEN = [
    "f0_ai_models",
    "f0_no_variables_metadata",
    "f1_single_dataset",
    "f1_ensemble",
    "f1_null_diagnostic",
]


@pytest.fixture
def legacy_dict(golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """The F0 golden dict (a fresh copy), which the null-section tests mutate."""
    return golden_loader("f0_ai_models")


class TestLegacyNullSectionRobustness:
    """Legacy checkpoints with config sections set to None are read without raising."""

    def test_null_data_does_not_break_get_timestep(self, legacy_dict: dict[str, Any]) -> None:
        """config.data=None returns the default timestep without raising."""
        legacy_dict["config"]["data"] = None
        meta = CheckpointAdapter(legacy_dict)
        # Should return the default "6h"
        assert meta.task.timestep() == "6h"

    def test_null_data_does_not_break_get_data_frequency(self, legacy_dict: dict[str, Any]) -> None:
        """config.data=None does not raise in data_frequency."""
        legacy_dict["config"]["data"] = None
        legacy_dict["dataset"]["frequency"] = "12h"
        meta = CheckpointAdapter(legacy_dict)
        # Should fall back to dataset.frequency
        assert meta.data("data").data_frequency() == "12h"

    def test_null_training_does_not_break_multistep_input(self, legacy_dict: dict[str, Any]) -> None:
        """config.training=None returns default multistep_input=1."""
        legacy_dict["config"]["training"] = None
        meta = CheckpointAdapter(legacy_dict)
        # task.input_relative_date_indices should use default multistep_input=1
        assert meta.task.input_relative_date_indices() == [0]

    def test_null_dataloader_returns_empty_dict(self, legacy_dict: dict[str, Any]) -> None:
        """config.dataloader=None causes dataloader_config to return {}."""
        legacy_dict["config"]["dataloader"] = None
        meta = CheckpointAdapter(legacy_dict)
        result = meta.data("data").dataloader_config("training")
        assert result == {}

    def test_null_partition_value_returns_empty_dict(self, legacy_dict: dict[str, Any]) -> None:
        """config.dataloader.training=None returns {} without raising."""
        legacy_dict["config"]["dataloader"] = {"training": None}
        meta = CheckpointAdapter(legacy_dict)
        result = meta.data("data").dataloader_config("training")
        assert result == {}

    def test_all_null_config_sections(self, legacy_dict: dict[str, Any]) -> None:
        """Multiple null config sections do not break accessors."""
        legacy_dict["config"] = {
            "data": None,
            "training": None,
            "dataloader": None,
        }
        legacy_dict["dataset"]["frequency"] = "3h"
        meta = CheckpointAdapter(legacy_dict)
        # timestep returns default
        assert meta.task.timestep() == "6h"
        # data_frequency falls back to dataset.frequency
        assert meta.data("data").data_frequency() == "3h"
        # dataloader_config returns empty dict
        assert meta.data("data").dataloader_config() == {}
        # multistep-related methods use defaults
        assert meta.task.input_relative_date_indices() == [0]

    def test_null_data_does_not_break_get_variable_types(self, legacy_dict: dict[str, Any]) -> None:
        """config.data=None in variable_types returns empty lists."""
        legacy_dict["config"]["data"] = None
        meta = CheckpointAdapter(legacy_dict)
        var_types = meta.data("data").variable_types()
        # forcing and diagnostic should be empty lists (default from .get())
        assert var_types["forcing"] == []
        assert var_types["diagnostic"] == []


class TestLegacyRouting:
    """Every legacy format goes through CheckpointAdapter and the 0.1 component adapters."""

    @pytest.mark.parametrize("name", LEGACY_GOLDEN)
    def test_legacy_golden_files_use_legacy_components(
        self,
        name: str,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        adapter = CheckpointAdapter(golden_loader(name))
        assert type(adapter) is CheckpointAdapter
        assert adapter.is_legacy
        assert adapter.component_versions["data"] == adapter.component_versions["task"] == "0.1"
        assert type(adapter.data("data")) is GriddedV0_1
        assert type(adapter.task) is ForecasterV0_1

    @pytest.mark.parametrize("name", LEGACY_GOLDEN)
    def test_legacy_is_one_dataset_called_data(self, name: str, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader(name))
        assert adapter.dataset_names == ["data"]
        with pytest.raises(KeyError):
            adapter.data("other")

    @pytest.mark.parametrize("name", LEGACY_GOLDEN)
    def test_legacy_task_is_the_forecaster(self, name: str, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader(name))
        assert adapter.task.task_name == "forecaster"
        assert adapter.task.n_step_input == 2
        assert adapter.task.n_step_output == 1
        assert adapter.task.input_relative_date_indices() == [0, 1]
        assert adapter.task.output_relative_date_indices() == [2]

    @pytest.mark.parametrize("name", LEGACY_GOLDEN)
    def test_legacy_indices_are_consistent(self, name: str, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        data = CheckpointAdapter(golden_loader(name)).data("data")
        inputs, outputs = data.variable_indices(), data.output_variable_indices()
        assert sorted(inputs.values()) == list(range(len(inputs)))
        assert set(outputs) <= set(inputs) | set(data.variables())
        types = data.variable_types()
        assert set(types["prognostic"]) <= set(outputs)
        assert types["target"] == [name for name, _ in sorted(outputs.items(), key=lambda item: item[1])]

    def test_accessors_are_read_only(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f0_ai_models"))
        for name in ("dataset_names", "task", "sections"):
            with pytest.raises(AttributeError):
                setattr(adapter, name, None)

    def test_dict_is_unchanged_by_repairs(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """The repairs apply to what is read; the raw metadata is returned as stored."""
        raw = golden_loader("f0_no_variables_metadata")
        adapter = CheckpointAdapter(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter.data("data").variables_metadata()
        assert adapter.to_dict() == raw
        assert "variables_metadata" not in adapter.to_dict()["dataset"]


class TestLegacyF0:
    """F0 (ai-models era): no version, no ``dataset.shape``, ``accumulated`` flags, ``param`` + ``mars``."""

    def test_no_version(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f0_ai_models"))
        assert adapter.version is None
        assert adapter.task.task_name == "forecaster"

    def test_grid_size_from_the_grid_point_table(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """``dataset.shape`` is absent: the O96 grid of the data request gives the point count (legacy.py:181)."""
        raw = golden_loader("f0_ai_models")
        assert "shape" not in raw["dataset"]
        adapter = CheckpointAdapter(raw)
        assert adapter.data("data").grid() == "O96"
        assert adapter.data("data").grid_size() == 40_320
        assert adapter.data("data").grid_size() == 40_320

    @pytest.mark.parametrize(("grid", "points"), [("N320", 542_080), ("n320", 542_080), ("O96", 40_320)])
    def test_grid_point_table(self, golden_loader: Callable[[str], dict[str, Any]], grid: str, points: int) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["specific"]["attrs"]["data_request"]["grid"] = grid
        assert CheckpointAdapter(raw).data("data").grid_size() == points

    @pytest.mark.parametrize("grid", ["O1280", [0.25, 0.25], None])
    def test_unknown_grid_is_not_recorded(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        grid: str | list[float] | None,
    ) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["specific"]["attrs"]["data_request"]["grid"] = grid
        assert CheckpointAdapter(raw).data("data").grid_size() is None

    def test_recorded_shape_wins_over_the_table(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["shape"] = [10, 10, 1, 77]
        assert CheckpointAdapter(raw).data("data").grid_size() == 77

    def test_accumulated_flag_counts_as_accumulation(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """``tp`` carries ``accumulated: true`` and no ``process`` (legacy checkpoints)."""
        raw = golden_loader("f0_ai_models")
        assert raw["dataset"]["variables_metadata"]["tp"] == {
            "accumulated": True,
            "mars": {"levtype": "sfc", "param": "tp"},
        }
        data = CheckpointAdapter(raw).data("data")
        assert data.accumulations() == ["tp"]
        assert data.variable_categories()["accumulation"] == ["tp"]
        assert data.select_variables(include=["accumulation"]) == ["tp"]

    def test_accumulated_flag_does_not_override_process(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["variables_metadata"]["tp"]["process"] = "average"
        assert CheckpointAdapter(raw).data("data").accumulations() == []

    def test_computed_forcings_and_constants(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        data = CheckpointAdapter(golden_loader("f0_ai_models")).data("data")
        assert data.computed_forcings() == ["cos_latitude", "insolation"]
        assert sorted(data.variable_categories()["constant"]) == ["cos_latitude", "lsm", "z"]

    def test_data_request_found_in_the_nested_dataset_description(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        assert CheckpointAdapter(golden_loader("f0_ai_models")).data("data").data_request() == {"grid": "O96"}

    def test_timestep_and_precision(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f0_ai_models"))
        assert adapter.task.timestep() == "6h"
        assert adapter.sections.precision == "16-mixed"

    def test_typed_variables(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        pytest.importorskip("anemoi.transform.variables")
        typed = CheckpointAdapter(golden_loader("f0_ai_models")).data("data").typed_variables()
        assert typed["tp"].is_accumulation
        assert typed["insolation"].is_computed_forcing


@pytest.mark.filterwarnings("ignore:Unknown variables")
class TestLegacyVariablesMetadataRepairs:
    """Port of anemoi-inference ``legacy.py:50-170``: synthesise, backfill ``mars``, repair computed forcings."""

    def test_synthesised_from_the_variable_names(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f0_no_variables_metadata")
        assert "variables_metadata" not in raw["dataset"]
        adapter = CheckpointAdapter(raw)
        with pytest.warns(UserWarning, match="synthesising"):
            metadata = adapter.data("data").variables_metadata()
        assert list(metadata) == adapter.data("data").variables()
        assert metadata["2t"] == {"mars": {"param": "2t", "levtype": "sfc"}}
        assert metadata["lsm"] == {"mars": {"param": "lsm", "levtype": "sfc"}, "constant_in_time": True}
        assert metadata["insolation"] == {"computed_forcing": True, "constant_in_time": False}
        assert metadata["cos_latitude"] == {"computed_forcing": True, "constant_in_time": True}
        assert metadata["tp"]["process"] == "accumulation"
        assert metadata["tp"]["mars"] == {"param": "tp", "levtype": "sfc", "type": "fc", "stream": "oper"}

    def test_synthesised_values_feed_the_derived_getters(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f0_no_variables_metadata"))
        with pytest.warns(UserWarning, match="synthesising"):
            assert adapter.data("data").accumulations() == ["tp"]
        assert adapter.data("data").computed_forcings() == ["cos_latitude", "insolation"]

    def test_synthesis_warns_once_per_adapter(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f0_no_variables_metadata"))
        with pytest.warns(UserWarning, match="synthesising"):
            adapter.data("data").variables_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            adapter.data("data").variables_metadata()
            adapter.data("data").accumulations()

    def test_synthesised_pressure_level_and_unknown_variables(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f0_no_variables_metadata")
        raw["dataset"]["variables"] = ["q_850", "2t", "mystery"]
        raw["data_indices"]["data"]["input"]["full"] = [0, 1, 2]
        raw["data_indices"]["data"]["output"]["full"] = [0, 1, 2]
        with pytest.warns(UserWarning, match="Unknown variables") as record:
            metadata = CheckpointAdapter(raw).data("data").variables_metadata()
        assert metadata["q_850"] == {"mars": {"param": "q", "levelist": 850, "levtype": "pl"}}
        assert metadata["mystery"] == {"mars": {"param": "mystery", "levtype": "sfc"}}
        assert any("Unknown variables: ['mystery']" in str(w.message) for w in record)

    def test_returned_metadata_is_a_copy(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f0_ai_models"))
        adapter.data("data").variables_metadata()["2t"]["mars"]["param"] = "changed"
        assert adapter.data("data").variables_metadata()["2t"]["mars"]["param"] == "2t"

    def test_param_without_mars_gets_mars(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["variables_metadata"]["2t"] = {"param": "2t", "levtype": "sfc"}
        with caplog.at_level(logging.WARNING):
            metadata = CheckpointAdapter(raw).data("data").variables_metadata()
        assert metadata["2t"]["mars"] == {"param": "2t", "levtype": "sfc"}
        assert "Old metadata format detected" in caplog.text

    def test_computed_forcing_flags_are_repaired(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["variables_metadata"]["insolation"] = {"computed_forcing": True, "constant_in_time": True}
        raw["dataset"]["variables_metadata"]["cos_latitude"] = {}
        metadata = CheckpointAdapter(raw).data("data").variables_metadata()
        assert metadata["insolation"] == {"computed_forcing": True, "constant_in_time": False}
        assert metadata["cos_latitude"] == {"computed_forcing": True, "constant_in_time": True}

    def test_good_metadata_is_left_alone(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        raw = golden_loader("f1_single_dataset")
        with caplog.at_level(logging.WARNING):
            metadata = CheckpointAdapter(raw).data("data").variables_metadata()
        assert metadata == raw["dataset"]["variables_metadata"] | {
            name: {**metadata[name]} for name in raw["dataset"].get("constant_fields", [])
        }
        assert "Old metadata format detected" not in caplog.text

    def test_constant_fields_are_marked(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f0_ai_models")
        raw["dataset"]["constant_fields"] = ["msl"]
        assert CheckpointAdapter(raw).data("data").variables_metadata()["msl"]["constant_in_time"] is True


class TestLegacyF1:
    """F1 (first anemoi-training layout, ``version`` "1.0")."""

    def test_version_and_task(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f1_single_dataset"))
        assert adapter.version == "1.0"
        assert adapter.task.task_name == "forecaster"

    def test_timestep_comes_from_the_config(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        assert CheckpointAdapter(golden_loader("f1_single_dataset")).task.timestep() == "24h"
        assert CheckpointAdapter(golden_loader("f1_ensemble")).task.timestep() == "6h"

    def test_data_request_is_found_deep_in_the_dataset_chain(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f1_single_dataset")
        assert "data_request" not in raw["dataset"]
        request = CheckpointAdapter(raw).data("data").data_request()
        assert request["grid"] == "O96"
        assert request["area"] == [89.285, 0.0, -89.285, 359.1]
        assert "param_level" in request

    def test_grid_and_area_come_from_the_nested_data_request(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        data = CheckpointAdapter(golden_loader("f1_single_dataset")).data("data")
        assert data.grid() == "O96"
        assert data.area() == [89.285, 0.0, -89.285, 359.1]

    def test_grid_size_from_dataset_shape(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f1_single_dataset")
        assert CheckpointAdapter(raw).data("data").grid_size() == raw["dataset"]["shape"][-1] == 40_320

    def test_data_frequency_and_precision(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f1_single_dataset"))
        assert adapter.data("data").data_frequency() == "6h"
        assert adapter.sections.precision == "bf16-mixed"

    def test_provenance_and_sections(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f1_single_dataset"))
        assert adapter.sections.provenance
        assert adapter.lineage.uuid
        assert adapter.sections.config["data"]["timestep"] == "24h"

    def test_variables_follow_dataset_order(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f1_single_dataset")
        data = CheckpointAdapter(raw).data("data")
        names = data.variables()
        # Inputs first, in input order; output-only variables follow.
        assert len(data.variable_indices()) == len(raw["data_indices"]["data"]["input"]["full"])
        assert names[: len(data.variable_indices())] == list(data.variable_indices())
        assert set(names) == set(raw["dataset"]["variables"])

    def test_derived_categories(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        data = CheckpointAdapter(golden_loader("f1_single_dataset")).data("data")
        categories = data.variable_categories()
        assert {"tp", "cp"} <= set(categories["accumulation"])
        assert {"cos_latitude", "insolation"} <= set(categories["computed"])


class TestLegacyF1Ensemble:
    """The ensemble-trained public checkpoint opens and reads the same way."""

    def test_opens(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f1_ensemble"))
        assert adapter.dataset_names == ["data"]
        assert len(adapter.data("data").variables()) == 134
        assert adapter.task.task_name == "forecaster"

    def test_grid_is_n320(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f1_ensemble")
        adapter = CheckpointAdapter(raw)
        assert adapter.data("data").grid() == "N320"
        assert adapter.data("data").grid_size() == raw["dataset"]["shape"][-1] == 542_080

    def test_grid_size_without_a_shape_uses_the_table(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f1_ensemble")
        del raw["dataset"]["shape"]
        assert CheckpointAdapter(raw).data("data").grid_size() == 542_080

    def test_variable_types(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        types = CheckpointAdapter(golden_loader("f1_ensemble")).data("data").variable_types()
        assert len(types["forcing"]) == 14
        assert len(types["prognostic"]) == 107
        assert len(types["diagnostic"]) == 13


class TestLegacyNullDiagnostic:
    """``config.data.diagnostic: null`` reads as an empty diagnostic list."""

    def test_diagnostic_is_an_empty_list(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f1_null_diagnostic")
        assert raw["config"]["data"]["diagnostic"] is None
        adapter = CheckpointAdapter(raw)
        types = adapter.data("data").variable_types()
        assert types["diagnostic"] == []
        assert isinstance(types["forcing"], list)
        assert adapter.data("data").select_variables(include=["diagnostic"]) == []

    def test_other_values_are_read(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        adapter = CheckpointAdapter(golden_loader("f1_null_diagnostic"))
        assert adapter.task.timestep() == "24h"
        assert adapter.data("data").grid_size() == 40_320
        assert adapter.data("data").data_request() == {}
        assert len(adapter.data("data").variables()) == 30

    def test_missing_diagnostic_key_is_empty_too(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f1_null_diagnostic")
        del raw["config"]["data"]["diagnostic"]
        assert CheckpointAdapter(raw).data("data").variable_types()["diagnostic"] == []
