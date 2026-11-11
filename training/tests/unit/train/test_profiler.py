# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.profiler import AnemoiProfiler

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), name_to_index)


def test_profiler_example_input_uses_task_num_input_timesteps() -> None:
    """Profiler example inputs slice with the instantiated task, not forecaster-only config keys."""
    profiler = AnemoiProfiler.__new__(AnemoiProfiler)
    profiler.task = TemporalDownscaler(input_timestep="18h", output_timestep="6h")
    profiler.config = DictConfig({"task": {}, "dataloader": {"read_group_size": 1}})
    profiler.data_indices = {"data": _make_minimal_index_collection({"A": 0, "B": 1})}

    batch = {"data": torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2)}

    class _DataModule:
        def train_dataloader(self) -> list[dict[str, torch.Tensor]]:
            return [batch]

    profiler.datamodule = _DataModule()

    example_input_array = profiler.get_example_input_array()

    torch.testing.assert_close(
        example_input_array["data"],
        batch["data"][:, : profiler.task.num_input_timesteps, ..., profiler.data_indices["data"].data.input.full],
    )


def test_model_summary_disabled_does_not_access_inputs_or_model(mocker: MockerFixture) -> None:
    """A disabled summary must not fetch data, construct the model or access the summary provider."""
    profiler = AnemoiProfiler.__new__(AnemoiProfiler)
    profiler.config = DictConfig({"diagnostics": {"benchmark_profiler": {"model_summary": {"enabled": False}}}})
    example_inputs = mocker.patch.object(
        profiler,
        "get_example_input_array",
        side_effect=AssertionError("Disabled summary fetched example inputs"),
    )
    model_access = mocker.patch.object(
        AnemoiProfiler,
        "model",
        new_callable=mocker.PropertyMock,
        side_effect=AssertionError("Disabled summary accessed the model"),
    )
    provider_access = mocker.patch.object(
        AnemoiProfiler,
        "profiler",
        new_callable=mocker.PropertyMock,
        side_effect=AssertionError("Disabled summary accessed the summary provider"),
    )

    assert profiler.model_summary is None

    example_inputs.assert_not_called()
    model_access.assert_not_called()
    provider_access.assert_not_called()


def test_model_summary_enabled_passes_inputs_to_summary_provider(mocker: MockerFixture) -> None:
    """An enabled summary forwards the model and example inputs to the existing provider exactly once."""
    profiler = AnemoiProfiler.__new__(AnemoiProfiler)
    profiler.config = DictConfig({"diagnostics": {"benchmark_profiler": {"model_summary": {"enabled": True}}}})
    example_input_array = {"data": mocker.sentinel.example_input}
    example_inputs = mocker.patch.object(profiler, "get_example_input_array", return_value=example_input_array)
    model_access = mocker.patch.object(
        AnemoiProfiler,
        "model",
        new_callable=mocker.PropertyMock,
        return_value=mocker.sentinel.model,
    )
    summary_provider = mocker.Mock(spec=["get_model_summary"])
    summary_provider.get_model_summary.return_value = "model summary"
    profiler.profiler = summary_provider

    assert profiler.model_summary == "model summary"

    example_inputs.assert_called_once_with()
    model_access.assert_called_once_with()
    summary_provider.get_model_summary.assert_called_once_with(
        model=mocker.sentinel.model,
        example_input_array=example_input_array,
    )
