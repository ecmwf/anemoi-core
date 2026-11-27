# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the data component adapters on the golden metadata files."""

import logging
from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.components.data import VARIABLE_CATEGORIES
from anemoi.metadata.components.data import DataAdapter

# ``tests/unit/conftest.py`` provides the loader as a fixture; it is bound here once per test so the
# helpers below can load golden files without a fixture argument on every test.
_LOADER = {}


@pytest.fixture(autouse=True)
def _bind_golden_loader(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    _LOADER["load"] = golden_loader


def load_golden(name: str) -> dict:
    """Load the golden file *name* (a fresh copy on every call)."""
    return _LOADER["load"](name)


def data_of(raw: dict, dataset: str | None = None) -> DataAdapter:
    """Return the data adapter of *dataset* (the first one when omitted) for the metadata dict *raw*."""
    checkpoint = CheckpointAdapter(raw)
    return checkpoint.data(dataset or checkpoint.dataset_names[0])


def adapter(name: str, dataset: str | None = None) -> DataAdapter:
    """Return the data adapter of *dataset* over the golden file *name*."""
    return data_of(load_golden(name), dataset)


# ---------------------------------------------------------------------------
# Dataset kind
# ---------------------------------------------------------------------------


class TestDataType:
    @pytest.mark.parametrize("name", ["f3_single_dataset", "f3_multi_dataset", "f4_task_top_level", "f6_main"])
    def test_2_0_layout_is_gridded(self, name: str) -> None:
        checkpoint = CheckpointAdapter(load_golden(name))
        for dataset in checkpoint.dataset_names:
            data = checkpoint.data(dataset)
            assert data.data_type() == "gridded"
            assert data.is_gridded()
            assert not data.is_tabular()

    def test_3_0_stored_data_type(self) -> None:
        gridded, tabular = adapter("f8_richer_batch", "data"), adapter("f8_richer_batch", "obs")
        assert (gridded.data_type(), tabular.data_type()) == ("gridded", "tabular")
        assert gridded.is_gridded() and not gridded.is_tabular()
        assert tabular.is_tabular() and not tabular.is_gridded()

    def test_3_0_missing_data_type_defaults_to_gridded(self) -> None:
        raw = load_golden("f8_richer_batch")
        del raw["metadata_inference"]["data"]["data_type"]
        assert data_of(raw, "data").data_type() == "gridded"

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(KeyError):
            adapter("f6_main", "nope")


# ---------------------------------------------------------------------------
# Variables, index maps, masks
# ---------------------------------------------------------------------------


class TestIndices:
    def test_f6_index_maps(self) -> None:
        data = adapter("f6_main")
        assert len(data.variable_indices()) == 99
        assert len(data.output_variable_indices()) == 88
        assert data.variable_indices()["2t"] == data.raw["metadata_inference"]["data"]["data_indices"]["input"]["2t"]

    def test_inverse_maps(self) -> None:
        data = adapter("f6_main")
        inputs = data.variable_indices()
        assert data.input_index_to_variable() == {index: name for name, index in inputs.items()}
        outputs = data.output_variable_indices()
        assert data.output_index_to_variable() == {index: name for name, index in outputs.items()}
        assert data.input_index_to_variable()[inputs["msl"]] == "msl"

    def test_variables_are_union_of_inputs_and_outputs(self) -> None:
        data = adapter("f6_main")
        variables = data.variables()
        assert len(variables) == 101
        assert set(variables) == set(data.variable_indices()) | set(data.output_variable_indices())
        # Output-only variables follow the inputs.
        assert variables[-2:] == ["cp", "tp"]
        assert variables[: len(data.variable_indices())] == list(data.input_index_to_variable().values())

    @pytest.mark.parametrize("name", ["f3_single_dataset", "f4_task_top_level", "f6_main", "f8_richer_batch"])
    def test_prognostic_masks_match_stored_model_indices(self, name: str) -> None:
        """The masks equal the full index collection written by training (``model.*.prognostic``)."""
        data = adapter(name, "data")
        stored = data.raw["data_indices"]["data"]["model"]
        assert data.prognostic_input_mask() == sorted(stored["input"]["prognostic"])
        assert data.prognostic_output_mask() == sorted(stored["output"]["prognostic"])

    def test_prognostic_masks_follow_the_index_maps(self) -> None:
        data = adapter("f6_main")
        prognostic = set(data.variable_types()["prognostic"])
        assert data.prognostic_input_mask() == sorted(data.variable_indices()[v] for v in prognostic)
        assert data.prognostic_output_mask() == sorted(data.output_variable_indices()[v] for v in prognostic)

    def test_multi_dataset_blocks_are_independent(self) -> None:
        era5, cerra = adapter("f3_multi_dataset", "era5"), adapter("f3_multi_dataset", "cerra")
        assert era5.variable_indices() != cerra.variable_indices()
        assert set(cerra.variables()) >= set(cerra.variable_types()["prognostic"])

    def test_input_only_dataset_has_empty_outputs(self) -> None:
        raw = load_golden("f8_richer_batch")
        raw["metadata_inference"]["obs"]["data_indices"]["output"] = {}
        data = data_of(raw, "obs")
        assert data.is_input_only()
        assert data.output_variable_indices() == {}
        assert data.output_index_to_variable() == {}
        assert data.prognostic_output_mask() == []
        assert data.prognostic_input_mask() == []
        assert data.variables() == list(data.variable_indices())
        assert not adapter("f8_richer_batch", "obs").is_input_only()

    def test_missing_data_indices_gives_empty_values(self) -> None:
        raw = load_golden("f6_main")
        del raw["metadata_inference"]["data"]["data_indices"]
        data = data_of(raw)
        assert data.variable_indices() == {}
        assert data.variables() == []
        assert data.prognostic_input_mask() == []
        assert not data.is_input_only()


# ---------------------------------------------------------------------------
# grid_size
# ---------------------------------------------------------------------------


class TestGridSize:
    def test_stored_grid_size_3_0(self) -> None:
        data = adapter("f8_richer_batch", "data")
        assert data.grid_size() == 40320

    def test_stored_wins_over_derived(self) -> None:
        raw = load_golden("f8_richer_batch")
        raw["metadata_inference"]["data"]["grid_size"] = 7
        assert data_of(raw, "data").grid_size() == 7

    def test_from_dataset_shape_when_shapes_grid_is_null(self) -> None:
        # f6_main stores shapes.grid = null; the dataset shape has the grid on the last axis.
        data = adapter("f6_main")
        assert data.grid_size() == 40320

    def test_from_shapes_grid_key(self) -> None:
        raw = load_golden("f6_main")
        raw["metadata_inference"]["data"]["shapes"]["grid"] = 123
        assert data_of(raw).grid_size() == 123

    def test_multi_dataset_shapes(self) -> None:
        assert adapter("f3_multi_dataset", "era5").grid_size() == 10944
        assert adapter("f3_multi_dataset", "cerra").grid_size() == 25326

    def test_not_recorded_is_none(self) -> None:
        raw = load_golden("f6_main")
        del raw["metadata_inference"]["data"]["shapes"]
        del raw["dataset"]["data"]["shape"]
        assert data_of(raw).grid_size() is None

    def test_3_0_null_grid_size_falls_back_for_gridded(self) -> None:
        raw = load_golden("f8_richer_batch")
        raw["metadata_inference"]["data"]["grid_size"] = None
        assert data_of(raw, "data").grid_size() == 40320

    def test_tabular_is_none(self) -> None:
        assert adapter("f8_richer_batch", "obs").grid_size() is None

    def test_tabular_ignores_stray_shape_values(self) -> None:
        raw = load_golden("f8_richer_batch")
        raw["metadata_inference"]["obs"]["shapes"] = {"grid": 5}
        assert data_of(raw, "obs").grid_size() is None


# ---------------------------------------------------------------------------
# data request, grid, area
# ---------------------------------------------------------------------------


class TestDataRequest:
    def test_nested_data_request_is_searched_on_f6(self) -> None:
        data = adapter("f6_main")
        # The request sits at dataset.data.specific.forward.attrs.data_request.
        assert "data_request" not in data.raw["dataset"]["data"]
        request = data.data_request()
        assert request["grid"] == "O96"
        assert request["area"] == [89.285, 0.0, -89.285, 359.1]
        assert "param_level" in request
        assert data.grid() == "O96"
        assert data.area() == [89.285, 0.0, -89.285, 359.1]

    @pytest.mark.parametrize("name", ["f3_single_dataset", "f4_task_top_level"])
    def test_deeper_nesting(self, name: str) -> None:
        data = adapter(name)
        assert data.grid() in ("O48", "O96")
        assert len(data.area()) == 4

    def test_multi_dataset_requests_are_per_dataset(self) -> None:
        assert adapter("f3_multi_dataset", "era5").grid() == "O48"
        cerra = adapter("f3_multi_dataset", "cerra")
        assert cerra.grid() == [0.2, 0.2]
        assert cerra.area() == [60.0, 345.0, 35.0, 25.0]

    def test_flat_dataset_dump_fallback(self) -> None:
        raw = load_golden("f6_main")
        raw["dataset"] = raw["dataset"]["data"]
        assert data_of(raw).grid() == "O96"
        assert data_of(raw).variables_metadata()["2t"]["mars"]["param"] == "2t"

    def test_data_request_inside_a_list(self) -> None:
        raw = {
            "metadata_inference": {"dataset_names": ["data"], "data": {}},
            "dataset": {"data": {"sources": [{"forward": {"attrs": {"data_request": {"grid": "N320"}}}}]}},
        }
        assert data_of(raw).grid() == "N320"

    def test_ambiguous_requests_pick_the_first_with_a_grid(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = {
            "metadata_inference": {"dataset_names": ["data"], "data": {}},
            "dataset": {
                "data": {
                    "specific": {
                        "datasets": [
                            {"attrs": {"data_request": {"grid": None, "area": "a"}}},
                            {"attrs": {"data_request": {"grid": "O96", "area": "b"}}},
                            {"attrs": {"data_request": {"grid": "N320", "area": "b"}}},
                        ],
                    },
                },
            },
        }
        with caplog.at_level(logging.WARNING, logger="anemoi.metadata.components.data"):
            assert data_of(raw).grid() == "O96"
        assert "ambiguous" in caplog.text

    def test_gridded_without_request_is_empty_not_none(self) -> None:
        raw = load_golden("f6_main")
        del raw["dataset"]["data"]["specific"]
        data = data_of(raw)
        assert data.data_request() == {}
        assert data.grid() is None
        assert data.area() is None
        assert data.is_gridded()

    def test_tabular_is_not_applicable(self) -> None:
        obs = adapter("f8_richer_batch", "obs")
        assert obs.data_request() is None
        assert obs.grid() is None
        assert obs.area() is None
        assert obs.grid_size() is None
        # Gridded dataset in the same checkpoint is unaffected.
        assert adapter("f8_richer_batch", "data").grid() == "O96"

    def test_not_applicable_is_distinct_from_not_recorded(self) -> None:
        tabular = adapter("f8_richer_batch", "obs")
        raw = load_golden("f8_richer_batch")
        del raw["dataset"]["data"]["specific"]
        recorded_nothing = data_of(raw, "data")
        assert tabular.data_request() is None
        assert recorded_nothing.data_request() == {}
        assert recorded_nothing.is_gridded() and tabular.is_tabular()


# ---------------------------------------------------------------------------
# Raw dump values
# ---------------------------------------------------------------------------


class TestRawValues:
    def test_variables_metadata_and_flags(self) -> None:
        data = adapter("f6_main")
        metadata = data.variables_metadata()
        assert set(metadata) >= {"2t", "tp"}
        assert "tp" in data.accumulations()
        assert "cos_latitude" in data.computed_forcings()
        assert all(name in metadata for name in data.accumulations())

    def test_constant_fields_are_marked(self) -> None:
        raw = load_golden("f6_main")
        raw["dataset"]["data"]["constant_fields"] = ["lsm", "not-a-variable"]
        metadata = data_of(raw).variables_metadata()
        assert metadata["lsm"]["constant_in_time"] is True
        assert "not-a-variable" not in metadata

    def test_tabular_has_no_variables_metadata(self) -> None:
        data = adapter("f8_richer_batch", "obs")
        assert data.variables_metadata() == {}
        assert data.accumulations() == []
        assert data.computed_forcings() == []
        assert data.sources() == []
        assert data.open_dataset_args() == {}

    def test_open_dataset_args_and_sources(self) -> None:
        data = adapter("f6_main")
        assert data.open_dataset_args() == data.raw["dataset"]["data"]["arguments"]
        assert data.sources() == []

    def test_sources_returned_when_recorded(self) -> None:
        raw = load_golden("f6_main")
        raw["dataset"]["data"]["sources"] = [{"dataset": "a"}, {"dataset": "b"}]
        assert data_of(raw).sources() == [{"dataset": "a"}, {"dataset": "b"}]

    def test_data_frequency_from_dataset(self) -> None:
        assert adapter("f6_main").data_frequency() == "6h"

    def test_data_frequency_null_falls_through(self) -> None:
        raw = load_golden("f6_main")
        raw["dataset"]["data"]["frequency"] = None
        raw["config"].setdefault("data", {})["frequency"] = "12h"
        raw["config"].setdefault("task", {}).pop("output_timestep", None)
        assert data_of(raw).data_frequency() == "12h"

    def test_data_frequency_all_null_is_none(self) -> None:
        raw = load_golden("f6_main")
        raw["dataset"]["data"]["frequency"] = None
        raw["config"].setdefault("data", {})["frequency"] = None
        raw["config"].setdefault("task", {}).pop("output_timestep", None)
        assert data_of(raw).data_frequency() is None

    def test_data_frequency_output_timestep_wins(self) -> None:
        raw = load_golden("f6_main")
        raw["config"].setdefault("task", {})["output_timestep"] = "3h"
        assert data_of(raw).data_frequency() == "3h"

    def test_data_frequency_handles_missing_config(self) -> None:
        raw = load_golden("f8_richer_batch")
        del raw["config"]
        assert data_of(raw, "data").data_frequency() == "6h"
        assert data_of(raw, "obs").data_frequency() is None

    def test_dataloader_config_single_and_multi(self) -> None:
        single = adapter("f6_main").dataloader_config()
        assert isinstance(single, dict)
        era5, cerra = adapter("f3_multi_dataset", "era5"), adapter("f3_multi_dataset", "cerra")
        assert era5.dataloader_config() != cerra.dataloader_config()

    def test_dataloader_config_missing_partition(self) -> None:
        assert adapter("f6_main").dataloader_config("nope") == {}


# ---------------------------------------------------------------------------
# Categories, selection, typed variables
# ---------------------------------------------------------------------------


class TestCategories:
    def test_variable_types_always_have_four_keys(self) -> None:
        for name in ("f6_main", "f3_single_dataset"):
            assert set(adapter(name).variable_types()) == {"forcing", "prognostic", "diagnostic", "target"}

    def test_category_keyed(self) -> None:
        data = adapter("f6_main")
        categories = data.variable_categories()
        assert set(categories) >= {"forcing", "prognostic", "diagnostic", "target"}
        assert categories["diagnostic"] == ["cp", "tp"]
        assert "tp" in categories["accumulation"]
        assert "cos_latitude" in categories["computed"]
        assert categories["target"] == []

    def test_empty_derived_categories_are_dropped(self) -> None:
        categories = adapter("f8_richer_batch", "obs").variable_categories()
        assert set(categories) == {"forcing", "prognostic", "diagnostic", "target"}

    def test_per_variable(self) -> None:
        data = adapter("f6_main")
        per_variable = data.variable_categories(per_variable=True)
        assert "prognostic" in per_variable["2t"]
        assert per_variable["tp"] == sorted(per_variable["tp"])
        assert {"diagnostic", "accumulation"} <= set(per_variable["tp"])
        assert "computed" in per_variable["cos_latitude"]

    def test_variable_types_do_not_include_unknown_extras(self) -> None:
        raw = load_golden("f6_main")
        raw["metadata_inference"]["data"]["variable_types"]["extra"] = ["x"]
        assert "extra" not in data_of(raw).variable_types()


class TestSelectVariables:
    def test_include_category(self) -> None:
        data = adapter("f6_main")
        assert data.select_variables(include=["diagnostic"]) == ["cp", "tp"]
        assert set(data.select_variables(include=["prognostic"])) == set(data.variable_types()["prognostic"])

    def test_include_several_is_a_union(self) -> None:
        data = adapter("f6_main")
        selected = data.select_variables(include=["prognostic", "forcing"])
        expected = set(data.variable_types()["prognostic"]) | set(data.variable_types()["forcing"])
        assert set(selected) == expected

    def test_compound_is_an_intersection(self) -> None:
        data = adapter("f6_main")
        assert data.select_variables(include=["diagnostic+accumulation"]) == ["cp", "tp"]
        assert data.select_variables(include=["prognostic+diagnostic"]) == []

    def test_exclude(self) -> None:
        data = adapter("f6_main")
        selected = data.select_variables(exclude=["forcing", "diagnostic"])
        assert set(selected) == set(data.variable_types()["prognostic"])

    def test_include_and_exclude(self) -> None:
        data = adapter("f6_main")
        selected = data.select_variables(include=["prognostic", "diagnostic"], exclude=["diagnostic"])
        assert set(selected) == set(data.variable_types()["prognostic"])

    def test_no_arguments_selects_everything_in_variables_order(self) -> None:
        data = adapter("f6_main")
        assert data.select_variables() == data.variables()

    def test_empty_include_selects_nothing(self) -> None:
        assert adapter("f6_main").select_variables(include=[]) == []

    def test_known_but_empty_category_is_fine(self) -> None:
        assert adapter("f6_main").select_variables(include=["target"]) == []
        assert adapter("f8_richer_batch", "obs").select_variables(include=["constant"]) == []

    @pytest.mark.parametrize("bad", ["progn", "2t", "prognostic+typo", "prognostic+", ""])
    def test_unknown_names_raise_in_include(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Unknown variable category"):
            adapter("f6_main").select_variables(include=[bad])

    def test_unknown_names_raise_in_exclude(self) -> None:
        with pytest.raises(ValueError, match="Unknown variable category"):
            adapter("f6_main").select_variables(exclude=["progostic"])

    def test_no_substring_fallback(self) -> None:
        # '2t' used to match by substring; it must now be rejected.
        with pytest.raises(ValueError, match="Unknown variable category"):
            adapter("f6_main").select_variables(include=["2t"])

    def test_error_lists_known_categories(self) -> None:
        with pytest.raises(ValueError, match="Unknown variable category") as error:
            adapter("f6_main").select_variables(include=["typo"])
        assert all(category in str(error.value) for category in VARIABLE_CATEGORIES)

    def test_per_dataset(self) -> None:
        assert adapter("f8_richer_batch", "obs").select_variables(include=["diagnostic"]) == ["tp"]
        assert adapter("f8_richer_batch", "data").select_variables(include=["diagnostic"]) == ["cp", "tp"]


class TestTypedVariables:
    def test_typed_variables(self) -> None:
        pytest.importorskip("anemoi.transform.variables")
        data = adapter("f6_main")
        typed = data.typed_variables()
        assert list(typed) == data.variables()
        assert typed["cos_latitude"].is_computed_forcing
        assert typed["tp"].is_accumulation

    def test_missing_transform_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "anemoi.transform.variables", None)
        with pytest.raises(ImportError, match="anemoi-transform"):
            adapter("f6_main").typed_variables()


# ---------------------------------------------------------------------------
# Copies
# ---------------------------------------------------------------------------


class TestReturnedValuesAreCopies:
    """Mutating a returned value must not change the adapter or the raw metadata."""

    @pytest.mark.parametrize(
        "method",
        [
            "variable_indices",
            "output_variable_indices",
            "input_index_to_variable",
            "output_index_to_variable",
            "variables",
            "variable_types",
            "prognostic_input_mask",
            "prognostic_output_mask",
            "data_request",
            "variables_metadata",
            "accumulations",
            "computed_forcings",
            "open_dataset_args",
            "dataloader_config",
            "variable_categories",
        ],
    )
    def test_mutating_a_result_changes_nothing(self, method: str) -> None:
        data = adapter("f6_main")
        first = getattr(data, method)()
        reference = getattr(adapter("f6_main"), method)()
        if isinstance(first, dict):
            for key in list(first):
                if isinstance(first[key], (list, dict)):
                    first[key].clear()
            first.clear()
        else:
            first.clear()
        assert getattr(data, method)() == reference
        assert data.raw == load_golden("f6_main")

    def test_nested_data_request_is_a_deep_copy(self) -> None:
        data = adapter("f6_main")
        data.data_request()["area"][0] = -1
        data.data_request()["param_level"]["sfc"].clear()
        assert data.area()[0] == 89.285
        assert data.raw == load_golden("f6_main")

    def test_grid_and_area_are_copies(self) -> None:
        data = adapter("f6_main")
        data.area().append(1)
        assert len(data.area()) == 4

    def test_sources_are_copies(self) -> None:
        raw = load_golden("f6_main")
        raw["dataset"]["data"]["sources"] = [{"dataset": "a"}]
        data = data_of(raw)
        data.sources()[0]["dataset"] = "changed"
        data.sources().append({})
        assert data.sources() == [{"dataset": "a"}]

    def test_variables_metadata_inner_dicts_are_copies(self) -> None:
        data = adapter("f6_main")
        data.variables_metadata()["2t"]["mars"]["param"] = "changed"
        assert data.variables_metadata()["2t"]["mars"]["param"] == "2t"

    def test_select_variables_returns_a_new_list(self) -> None:
        data = adapter("f6_main")
        data.select_variables().clear()
        assert len(data.select_variables()) == 101


# ---------------------------------------------------------------------------
# Whole golden set
# ---------------------------------------------------------------------------


def test_2_0_and_3_0_goldens_give_consistent_values(golden_metadata: tuple[str, dict[str, Any]]) -> None:
    """Every golden file in a data-adapter layout gives self-consistent values."""
    _, raw = golden_metadata
    if "dataset_names" not in raw.get("metadata_inference", {}):
        pytest.skip("legacy layout without metadata_inference: handled by the legacy adapters")
    checkpoint = CheckpointAdapter(raw)
    for dataset in checkpoint.dataset_names:
        data = checkpoint.data(dataset)
        variables = data.variables()
        assert len(variables) == len(set(variables))
        assert set(data.variable_indices()) <= set(variables)
        if data.is_tabular():
            assert data.grid_size() is None
            assert data.data_request() is None
        else:
            assert isinstance(data.data_request(), dict)
            assert data.grid_size() is None or data.grid_size() > 0
        assert data.select_variables() == variables
