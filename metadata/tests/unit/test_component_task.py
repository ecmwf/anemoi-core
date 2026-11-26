# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests of the checkpoint-level task component adapters, on the golden fixtures."""

import copy
import datetime
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata.components.task import AutoencoderV1_0
from anemoi.metadata.components.task import ForecasterV0_2
from anemoi.metadata.components.task import ForecasterV1_0
from anemoi.metadata.components.task import OffsetForecasterV0_2
from anemoi.metadata.components.task import TaskAdapter
from anemoi.metadata.components.task import TaskReaderV1_0
from anemoi.metadata.components.task import TaskTiming
from anemoi.metadata.components.task import TaskTimingError
from anemoi.metadata.components.task import TaskV0_2
from anemoi.metadata.components.task import TemporalDownscalerV0_2
from anemoi.metadata.components.task import TemporalDownscalerV1_0
from anemoi.metadata.components.task import build_task_adapter
from anemoi.metadata.components.task import register_task_version
from anemoi.metadata.formats import component_versions
from anemoi.metadata.registry import ComponentRegistry

H = datetime.timedelta(hours=1)
ZERO = datetime.timedelta(0)


def make_task(raw: dict[str, Any]) -> TaskAdapter:
    """Build the task adapter chosen for *raw* the way the registry selects it: by task name and version."""
    version = component_versions(raw)["task"]
    return build_task_adapter(raw, version)


def timesteps_of(raw: dict[str, Any], dataset: str = "data") -> dict[str, Any]:
    return raw["metadata_inference"][dataset]["timesteps"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.parametrize("version", ["0.2", "1.0"])
    @pytest.mark.parametrize(
        "name",
        ["forecaster", "offset-forecaster", "temporal-downscaler", "timeless", "autoencoder"],
    )
    def test_named_adapters_exist_for_both_versions(
        self,
        name: str,
        version: str,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        adapter = ComponentRegistry.get("task", version, name)
        assert adapter.name == name
        assert adapter.version == version
        assert adapter(golden_loader("f6_main")).timestep() == "6h"

    @pytest.mark.usefixtures("registry_state")
    def test_a_version_level_change_reaches_every_named_task(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        """A reader registered for a new component version is used by every task name."""

        class NextTaskReader(TaskReaderV1_0):
            def _timesteps(self, dataset_name: str) -> dict[str, Any]:
                return {**super()._timesteps(dataset_name), "timestep": "3h"}

        register_task_version("1.1", NextTaskReader)

        for fixture, name in [
            ("f6_main", "forecaster"),
            ("f5_offsets", "offset-forecaster"),
            ("f3_temporal_downscaler", "temporal-downscaler"),
            ("f3_temporal_downscaler", "time-interpolator"),
            ("f6_main", "timeless"),
            ("f6_main", "autoencoder"),
        ]:
            raw = golden_loader(fixture)
            raw["metadata_inference"]["task"] = name
            raw["component_versions"] = {"task": "1.1"}
            task = make_task(raw)
            assert task.version == "1.1"
            assert task.timestep() == "3h", name
            assert task.task_name == name

        # The task names still behave as before on top of the new reader.
        raw = golden_loader("f6_main")
        raw["component_versions"] = {"task": "1.1"}
        task = make_task(raw)
        assert task.output_offsets == [3 * H]
        raw["metadata_inference"]["task"] = "temporal-downscaler"
        task = make_task(raw)
        # Explicit times [0, 1] in steps of the 6h data frequency.
        assert task.input_offsets == [ZERO, 6 * H]
        assert task.timestep() == "3h"

    def test_unknown_task_name_falls_back_to_base_adapter_with_warning(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f6_main")
        raw["metadata_inference"]["task"] = "brand-new-task"
        with pytest.warns(UserWarning, match="brand-new-task"):
            task = make_task(raw)
        assert type(task) is TaskV0_2
        # The base adapter still reads the generic timing.
        assert task.n_step_input == 2
        assert task.timestep() == "6h"
        assert task.output_offsets is None
        assert task.task_name == "brand-new-task"

    @pytest.mark.parametrize(
        ("fixture", "cls"),
        [
            ("f3_single_dataset", "ForecasterV0_2"),
            ("f5_offsets", "OffsetForecasterV0_2"),
            ("f3_temporal_downscaler", "TemporalDownscalerV0_2"),
            ("f8_richer_batch", "ForecasterV1_0"),
        ],
    )
    def test_golden_fixture_selects_adapter(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        fixture: str,
        cls: str,
    ) -> None:
        assert type(make_task(golden_loader(fixture))).__name__ == cls

    @pytest.mark.parametrize(
        ("version", "cls"),
        [("0.2", TemporalDownscalerV0_2), ("1.0", TemporalDownscalerV1_0)],
    )
    def test_time_interpolator_is_read_as_temporal_downscaler(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        version: str,
        cls: type[TaskAdapter],
    ) -> None:
        """``time-interpolator`` is an alias: same class, no warning, recorded name kept."""
        raw = golden_loader("f3_temporal_downscaler")
        raw["metadata_inference"]["task"] = "time-interpolator"
        raw["component_versions"] = {"task": version}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            task = make_task(raw)
        assert type(task) is cls
        assert task.task_name == "time-interpolator"


# ---------------------------------------------------------------------------
# Task name (G3)
# ---------------------------------------------------------------------------


class TestTaskName:
    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            ("f2_early", "forecaster"),  # nothing recorded: default
            ("f2_dev_scalar_index", "forecaster"),
            ("f3_single_dataset", "forecaster"),
            ("f3_temporal_downscaler", "time-interpolator"),
            ("f4_task_top_level", "forecaster"),  # top-level task, metadata_inference.task is null
            ("f5_offsets", "offset-forecaster"),
            ("f6_main", "forecaster"),
            ("f8_richer_batch", "forecaster"),
        ],
    )
    def test_name_on_golden(self, golden_loader: Callable[[str], dict[str, Any]], fixture: str, expected: str) -> None:
        assert make_task(golden_loader(fixture)).task_name == expected

    def test_metadata_inference_wins_over_top_level(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f4_task_top_level")
        raw["metadata_inference"]["task"] = "offset-forecaster"
        raw["task"] = "forecaster"
        assert TaskV0_2(raw).task_name == "offset-forecaster"

    def test_top_level_used_when_inference_task_is_null(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f4_task_top_level")
        assert raw["metadata_inference"]["task"] is None
        assert raw["task"] == "forecaster"
        raw["task"] = "autoencoder"
        assert TaskV0_2(raw).task_name == "autoencoder"

    def test_default_is_forecaster(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f2_early")
        assert "task" not in raw
        assert make_task(raw).task_name == "forecaster"
        assert type(make_task(raw)) is ForecasterV0_2


# ---------------------------------------------------------------------------
# Timing on the golden fixtures
# ---------------------------------------------------------------------------


class TestForecasterTiming:
    def test_f3_single_dataset(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f3_single_dataset"))
        assert task.timestep() == "6h"
        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2, 3]
        assert task.timing.relative_date_indices_training == [0, 1, 2, 3]
        assert task.n_step_input == 2
        assert task.n_step_output == 2
        assert task.lagged == [-6 * H, ZERO]
        assert task.input_offsets == [-6 * H, ZERO]
        assert task.output_offsets == [6 * H, 12 * H]
        assert task.rollout_shift == 12 * H
        assert task.advance_map == {"outin": [(1, 1), (0, 0)], "inin": []}

    def test_f6_main(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f6_main"))
        assert task.timestep() == "6h"
        assert task.n_step_input == 2
        assert task.n_step_output == 1
        assert task.lagged == [-6 * H, ZERO]
        assert task.output_offsets == [6 * H]
        assert task.rollout_shift == 6 * H
        assert task.advance_map == {"outin": [(0, 1)], "inin": [(1, 0)]}

    def test_f4_task_top_level(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f4_task_top_level"))
        assert task.timestep() == "6h"
        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2]
        assert task.n_step_input == 2
        assert task.n_step_output == 1
        assert task.lagged == [-6 * H, ZERO]
        assert task.rollout_shift == 6 * H
        assert task.advance_map == {"outin": [(0, 1)], "inin": [(1, 0)]}

    def test_f8_richer_batch(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f8_richer_batch")
        task = make_task(raw)
        assert type(task) is ForecasterV1_0
        assert task.dataset_names() == ["data", "obs"]
        assert task.n_step_input == 2
        assert task.n_step_output == 1
        assert task.for_dataset("obs") == task.for_dataset("data")

    def test_explicit_times_are_the_relative_indices(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f3_single_dataset"))
        assert task.input_explicit_times == [0, 1]
        assert task.target_explicit_times == [2, 3]

    def test_output_offsets_do_not_need_the_input_steps(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """The output side is derived from the timestep and the output steps alone."""
        raw = golden_loader("f3_single_dataset")
        del raw["metadata_inference"]["data"]["timesteps"]["input_relative_date_indices"]
        del raw["config"]["training"]["multistep_input"]
        task = make_task(raw)
        assert task.n_step_input is None
        assert task.output_offsets == [6 * H, 12 * H]
        assert task.rollout_shift == 12 * H

    def test_dev_scalar_output_index(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """A bare int output index (seen in real files) reads as a one item list."""
        raw = golden_loader("f2_dev_scalar_index")
        assert timesteps_of(raw)["output_relative_date_indices"] == 2
        task = make_task(raw)
        assert task.output_relative_date_indices() == [2]
        assert task.n_step_output == 1
        assert task.n_step_input == 2
        assert task.rollout_shift == 6 * H


class TestOffsetForecasterTiming:
    def test_f5_offsets(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f5_offsets"))
        assert task.timestep() == "6h"
        # Indices are positions in the offsets lists.
        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2]
        assert task.n_step_input == 2
        assert task.n_step_output == 1
        assert task.input_offsets == [-6 * H, ZERO]
        assert task.lagged == [-6 * H, ZERO]
        assert task.output_offsets == [6 * H]
        assert task.rollout_shift == 6 * H
        assert task.advance_map == {"inin": [(1, 0)], "outin": [(0, 1)]}

    def test_stored_values_win_over_derivation(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f5_offsets")
        ts = timesteps_of(raw)
        ts.update(input_offsets=["-24h", "-12h", "0h"], output_offsets=["6h", "12h"], rollout_shift="12h")
        ts["advance_map"] = {"inin": [[2, 1]], "outin": [[1, 2]]}
        ts["input_relative_date_indices"] = [0, 1, 2]
        ts["output_relative_date_indices"] = [3, 4]
        task = make_task(raw)
        assert task.input_offsets == [-24 * H, -12 * H, ZERO]
        assert task.lagged == [-24 * H, -12 * H, ZERO]
        assert task.output_offsets == [6 * H, 12 * H]
        assert task.rollout_shift == 12 * H
        assert task.advance_map == {"inin": [(2, 1)], "outin": [(1, 2)]}
        assert task.n_step_input == 3

    def test_rollout_shift_defaults_to_last_stored_output_offset(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f5_offsets")
        ts = timesteps_of(raw)
        ts.update(output_offsets=["6h", "12h"], output_relative_date_indices=[2, 3])
        del ts["rollout_shift"]
        task = make_task(raw)
        assert task.output_offsets == [6 * H, 12 * H]
        assert task.rollout_shift == 12 * H

    def test_missing_offset_keys_fall_back_to_forecaster_derivation(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f5_offsets")
        for key in ("input_offsets", "output_offsets", "rollout_shift", "advance_map"):
            del timesteps_of(raw)[key]
        task = make_task(raw)
        assert task.output_offsets == [6 * H]
        assert task.rollout_shift == 6 * H

    def test_offsets_are_not_dropped_on_generic_adapter(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """G2: the generic adapter ignores them; only the offset forecaster adapter reads them."""
        raw = golden_loader("f5_offsets")
        assert TaskV0_2(raw).input_offsets is None
        assert OffsetForecasterV0_2(raw).input_offsets == [-6 * H, ZERO]


class TestTemporalDownscalerTiming:
    def test_f3_temporal_downscaler(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        task = make_task(raw)
        assert type(task) is TemporalDownscalerV0_2
        assert task.timestep() == "6h"
        assert task.input_relative_date_indices() == [0, 2]
        assert task.output_relative_date_indices() == [1, 2]
        assert task.n_step_input == 2
        assert task.n_step_output == 2
        # Older file without input_timestep/output_timestep: explicit times in data frequency.
        assert task.input_offsets == task.lagged == [ZERO, 12 * H]
        assert task.output_offsets == [6 * H, 12 * H]
        assert task.rollout_shift == 12 * H
        assert task.advance_map is None

    def test_explicit_times_and_lagged_look_ahead(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """Lagged is the input times in data frequency (6h here), so it looks ahead (temporal downscaler)."""
        task = make_task(golden_loader("f3_temporal_downscaler"))
        assert task.input_explicit_times == [0, 2]
        assert task.target_explicit_times == [1, 2]
        assert task.lagged == [ZERO, 12 * H]

    @staticmethod
    def single_dataset(raw: dict[str, Any]) -> TemporalDownscalerV0_2:
        """Without ``dataset_names`` the explicit times come from the config, not the relative indices."""
        del raw["metadata_inference"]["dataset_names"]
        return TemporalDownscalerV0_2(raw)

    def test_real_layout_with_task_config(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """Inputs 6h apart, outputs hourly: explicit times in ``config.task`` are in ``output_timestep`` units."""
        raw = golden_loader("f3_temporal_downscaler")
        del raw["config"]["training"]["explicit_times"]
        raw["config"]["task"] = {
            "explicit_times": {"input": [0, 6], "target": [1, 2, 3, 4, 5, 6]},
            "input_timestep": "6h",
            "output_timestep": "1h",
        }
        task = self.single_dataset(raw)
        assert task.input_explicit_times == [0, 6]
        assert task.target_explicit_times == [1, 2, 3, 4, 5, 6]
        assert task.lagged == [ZERO, 6 * H]
        assert task.timestep() == "6h"

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (False, False, [1, 2, 3, 4, 5]),
            (False, True, [1, 2, 3, 4, 5, 6]),
            (True, False, [0, 1, 2, 3, 4, 5]),
            (True, True, [0, 1, 2, 3, 4, 5, 6]),
        ],
    )
    def test_offsets_as_training_defines_them(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
        left: bool,
        right: bool,
        expected: list[int],
    ) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        raw["config"]["task"] = {
            "input_timestep": "6h",
            "output_timestep": "1h",
            "output_left_boundary": left,
            "output_right_boundary": right,
        }
        task = make_task(raw)
        assert task.input_offsets == task.lagged == [ZERO, 6 * H]
        assert task.output_offsets == [h * H for h in expected]
        assert task.rollout_shift == 6 * H
        assert task.advance_map is None
        assert task.timestep() == "6h"

    def test_non_multiple_timesteps_fall_back_to_explicit_times(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        raw["config"]["task"] = {"input_timestep": "6h", "output_timestep": "4h"}
        # Explicit times [1, 2] in output_timestep (4h) units.
        assert make_task(raw).output_offsets == [4 * H, 8 * H]

    def test_training_explicit_times_win_over_task(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        raw["config"]["task"] = {"explicit_times": {"input": [0, 4], "target": [1]}}
        task = self.single_dataset(raw)
        assert task.input_explicit_times == [0, 2]
        assert task.target_explicit_times == [1, 2]

    def test_relative_indices_when_no_explicit_times(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        del raw["config"]["training"]["explicit_times"]
        task = make_task(raw)
        assert task.input_explicit_times == task.input_relative_date_indices() == [0, 2]
        assert task.target_explicit_times == task.output_relative_date_indices() == [1, 2]
        assert task.lagged == [ZERO, 12 * H]

    def test_several_datasets_use_relative_indices(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_multi_dataset")
        raw["metadata_inference"]["task"] = "temporal-downscaler"
        raw["config"]["training"]["explicit_times"] = {"input": [0, 9], "target": [3]}
        task = make_task(raw)
        assert task.input_explicit_times == task.input_relative_date_indices("era5") == [0, 1]
        assert task.for_dataset("cerra").target_explicit_times == [2]

    def test_lagged_is_kept_without_a_data_frequency(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        del raw["config"]["data"]["frequency"]
        del raw["dataset"]["data"]["frequency"]
        assert make_task(raw).lagged == [-6 * H, ZERO]

    def test_repeated_training_index_is_kept(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        timesteps_of(raw)["relative_date_indices_training"] = [0, 1, 2, 3, 4, 5, 6, 6]
        task = make_task(raw)
        assert task.timing.relative_date_indices_training == [0, 1, 2, 3, 4, 5, 6, 6]
        assert task.n_step_input == 2
        assert task.n_step_output == 2

    def test_temporal_downscaler_name(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_temporal_downscaler")
        raw["metadata_inference"]["task"] = "temporal-downscaler"
        assert type(make_task(raw)) is TemporalDownscalerV0_2


class TestTimelessTiming:
    @pytest.fixture
    def raw(self, golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        raw = golden_loader("f3_single_dataset")
        raw["metadata_inference"]["task"] = "autoencoder"
        raw["metadata_inference"]["data"]["timesteps"] = {
            "relative_date_indices_training": [0],
            "input_relative_date_indices": [0],
            "output_relative_date_indices": [0],
            "timestep": "0H",
        }
        raw["version"] = "3.0"
        return raw

    def test_timeless(self, raw: dict[str, Any]) -> None:
        task = make_task(raw)
        assert type(task) is AutoencoderV1_0
        assert task.timestep() == "0H"
        assert task.n_step_input == 1
        assert task.n_step_output == 1
        assert task.lagged == [ZERO]
        assert task.input_offsets == [ZERO]
        assert task.output_offsets == [ZERO]

    def test_timeless_defaults_without_timesteps(self, raw: dict[str, Any]) -> None:
        del raw["metadata_inference"]["data"]["timesteps"]
        task = make_task(raw)
        assert type(task) is AutoencoderV1_0
        assert task.timestep() == "0H"
        assert task.n_step_input == 1
        assert task.n_step_output == 1
        assert task.input_relative_date_indices() == [0]
        assert task.output_relative_date_indices() == [1]
        assert task.lagged == [ZERO]


# ---------------------------------------------------------------------------
# Derived defaults on historical formats
# ---------------------------------------------------------------------------


class TestDerivedDefaults:
    def test_f2_early_indices_come_from_config(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        """F2-early records neither task nor input/output indices: multistep comes from config.training."""
        raw = golden_loader("f2_early")
        assert "input_relative_date_indices" not in timesteps_of(raw)
        task = make_task(raw)
        assert task.n_step_input == 2
        assert task.n_step_output == 2
        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2, 3]
        assert task.timing.relative_date_indices_training == [0, 1, 2, 3]
        assert task.lagged == [-6 * H, ZERO]

    def test_f2_early_matches_f3_single_dataset(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        early = make_task(golden_loader("f2_early")).timing
        later = make_task(golden_loader("f3_single_dataset")).timing
        assert early == later

    def test_f3_derived_equals_recorded(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_single_dataset")
        recorded = make_task(copy.deepcopy(raw)).timing
        for key in ("input_relative_date_indices", "output_relative_date_indices"):
            del timesteps_of(raw)[key]
        assert make_task(raw).timing == recorded

    def test_f3_missing_timestep_comes_from_config_data(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_single_dataset")
        del timesteps_of(raw)["timestep"]
        assert raw["config"]["data"]["timestep"] == "6h"
        assert make_task(raw).timestep() == "6h"

    def test_f4_defaults_from_config_task(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f4_task_top_level")
        recorded = make_task(copy.deepcopy(raw)).timing
        timesteps = timesteps_of(raw)
        for key in ("input_relative_date_indices", "output_relative_date_indices", "relative_date_indices_training"):
            del timesteps[key]
        del timesteps["timestep"]
        assert raw["config"]["task"]["multistep_input"] == 2
        assert raw["config"]["task"]["timestep"] == "6H"
        derived = make_task(raw).timing
        assert derived.input_relative_date_indices == [0, 1]
        assert derived.output_relative_date_indices == [2]
        assert derived.n_step_input == recorded.n_step_input == 2
        assert derived.n_step_output == recorded.n_step_output == 1
        assert derived.lagged == recorded.lagged
        assert derived.timestep == "6H"

    def test_single_output_step_when_config_has_no_multistep_output(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f3_single_dataset")
        timesteps = timesteps_of(raw)
        del timesteps["input_relative_date_indices"]
        del timesteps["output_relative_date_indices"]
        del raw["config"]["training"]["multistep_output"]
        task = make_task(raw)
        assert task.n_step_input == 2
        assert task.n_step_output == 1

    def test_nothing_recorded_reads_empty(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_single_dataset")
        del raw["metadata_inference"]["data"]["timesteps"]
        del raw["config"]["training"]["multistep_input"]
        del raw["config"]["training"]["multistep_output"]
        task = make_task(raw)
        assert task.input_relative_date_indices() == []
        assert task.output_relative_date_indices() == []
        assert task.n_step_input is None
        assert task.n_step_output is None
        assert task.lagged is None
        assert task.output_offsets is None
        assert task.timestep() == "6h"

    def test_unreadable_timestep_gives_no_lagged(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f6_main")
        timesteps_of(raw)["timestep"] = "soon"
        task = make_task(raw)
        assert task.timestep() == "soon"
        assert task.lagged is None
        assert task.n_step_input == 2


# ---------------------------------------------------------------------------
# Several datasets
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_agreeing_timings_give_one_global_value(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f3_multi_dataset"))
        assert task.dataset_names() == ["era5", "cerra"]
        assert task.timestep() == "6h"
        assert task.n_step_input == 2
        assert task.n_step_output == 1
        assert task.for_dataset("era5") == task.for_dataset("cerra") == task.timing
        assert set(task.timings()) == {"era5", "cerra"}

    @pytest.fixture
    def conflicting(self, golden_loader: Callable[[str], dict[str, Any]]) -> TaskAdapter:
        raw = golden_loader("f3_multi_dataset")
        timesteps_of(raw, "cerra").update(
            timestep="1h",
            relative_date_indices_training=[0, 1, 2, 3],
            input_relative_date_indices=[0, 1, 2],
            output_relative_date_indices=[3],
        )
        return make_task(raw)

    def test_conflicting_timings_global_access_raises(self, conflicting: TaskAdapter) -> None:
        for read in (
            lambda t: t.timing,
            lambda t: t.timestep(),
            lambda t: t.input_relative_date_indices(),
            lambda t: t.n_step_input,
            lambda t: t.n_step_output,
            lambda t: t.lagged,
            lambda t: t.rollout_shift,
            lambda t: t.advance_map,
        ):
            with pytest.raises(TaskTimingError, match="for_dataset"):
                read(conflicting)

    def test_conflicting_timings_for_dataset_works(self, conflicting: TaskAdapter) -> None:
        era5 = conflicting.for_dataset("era5")
        cerra = conflicting.for_dataset("cerra")
        assert era5.timestep == "6h"
        assert era5.n_step_input == 2
        assert cerra.timestep == "1h"
        assert cerra.n_step_input == 3
        assert cerra.lagged == [-2 * H, -H, ZERO]
        assert cerra.output_offsets == [H]
        assert conflicting.timestep("cerra") == "1h"
        assert conflicting.input_relative_date_indices("era5") == [0, 1]

    def test_conflict_in_a_single_field_is_detected(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_multi_dataset")
        timesteps_of(raw, "cerra")["output_relative_date_indices"] = [3]
        task = make_task(raw)
        with pytest.raises(TaskTimingError):
            _ = task.n_step_input
        assert task.for_dataset("era5").output_relative_date_indices == [2]
        assert task.for_dataset("cerra").output_relative_date_indices == [3]

    def test_unknown_dataset_raises(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        task = make_task(golden_loader("f3_multi_dataset"))
        with pytest.raises(KeyError, match="nope"):
            task.for_dataset("nope")
        with pytest.raises(KeyError):
            task.timestep("nope")

    def test_no_datasets(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f6_main")
        raw["metadata_inference"]["dataset_names"] = []
        task = make_task(raw)
        assert task.dataset_names() == []
        assert task.timings() == {}
        # Falls back to what the config records.
        assert task.n_step_input == 2
        assert task.timestep() == "6H"


# ---------------------------------------------------------------------------
# Copies and leniency
# ---------------------------------------------------------------------------


class TestCopies:
    def test_returned_lists_and_dicts_are_copies(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f5_offsets")
        before = copy.deepcopy(raw)
        task = make_task(raw)
        timing = task.timing

        task.input_relative_date_indices().append(99)
        task.output_relative_date_indices().append(99)
        timing.relative_date_indices_training.append(99)
        task.lagged.append(H)
        task.input_offsets.append(H)
        task.output_offsets.append(H)
        task.advance_map["inin"].append((9, 9))
        task.advance_map["new"] = []
        task.for_dataset("data").advance_map.clear()
        task.timings().clear()

        assert task.input_relative_date_indices() == [0, 1]
        assert task.output_relative_date_indices() == [2]
        assert timing.relative_date_indices_training == [0, 1, 2]
        assert task.lagged == [-6 * H, ZERO]
        assert task.input_offsets == [-6 * H, ZERO]
        assert task.output_offsets == [6 * H]
        assert task.advance_map == {"inin": [(1, 0)], "outin": [(0, 1)]}
        assert raw == before

    def test_timing_does_not_alias_the_raw_dict(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f3_single_dataset")
        task = make_task(raw)
        indices = task.input_relative_date_indices()
        indices.append(5)
        assert timesteps_of(raw)["input_relative_date_indices"] == [0, 1]

    def test_timing_repr_and_equality(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        timing = make_task(golden_loader("f6_main")).timing
        assert isinstance(timing, TaskTiming)
        assert "n_step_input=2" in repr(timing)
        assert timing != "x"


class TestLenientRead:
    def test_legacy_without_inference_block_reads_from_config(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        task = TaskV0_2(golden_loader("f1_single_dataset"))
        assert task.dataset_names() == []
        assert task.task_name == "forecaster"
        assert task.timestep() == "24h"
        assert task.n_step_input == 2
        assert task.n_step_output == 1
        assert task.input_relative_date_indices() == [0, 1]

    def test_legacy_without_any_timing_reads_empty(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        raw = golden_loader("f0_ai_models")
        raw["config"].pop("training", None)
        raw["config"].pop("task", None)
        task = TaskV0_2(raw)
        assert task.n_step_input is None
        assert task.input_relative_date_indices() == []

    def test_no_warning_for_known_names(self, golden_loader: Callable[[str], dict[str, Any]]) -> None:
        for fixture in ("f3_single_dataset", "f5_offsets", "f6_main", "f8_richer_batch"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                make_task(golden_loader(fixture))
