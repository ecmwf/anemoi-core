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

import pytest
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


@pytest.fixture
def profile_owner(mocker: MockerFixture) -> AnemoiProfiler:
    """Exercise the real entry point without constructing a training model or loggers."""
    owner = AnemoiProfiler.__new__(AnemoiProfiler)
    owner.profiler = mocker.Mock(spec=["teardown"])
    mocker.patch.object(owner, "train")
    mocker.patch.object(owner, "report")
    mocker.patch.object(owner, "export_to_logger")
    return owner


def test_profile_finalizes_before_reporting(profile_owner: AnemoiProfiler, mocker: MockerFixture) -> None:
    """Success reports cannot precede capture finalization."""
    order = mocker.Mock()
    order.attach_mock(profile_owner.train, "train")
    order.attach_mock(profile_owner.profiler.teardown, "teardown")
    order.attach_mock(profile_owner.report, "report")
    order.attach_mock(profile_owner.export_to_logger, "export")

    profile_owner.profile()

    assert order.mock_calls == [
        mocker.call.train(),
        mocker.call.teardown(stage="fit"),
        mocker.call.report(),
        mocker.call.export(),
    ]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_profile_finalizes_on_training_failure(
    profile_owner: AnemoiProfiler,
    error_type: type[BaseException],
) -> None:
    """Training failures and interruptions close the owner without reporting success."""
    failure = error_type("training failed before capture activation")
    profile_owner.train.side_effect = failure

    with pytest.raises(error_type, match="training failed before capture activation") as raised:
        profile_owner.profile()

    assert raised.value is failure
    profile_owner.profiler.teardown.assert_called_once_with(stage="fit")
    profile_owner.report.assert_not_called()
    profile_owner.export_to_logger.assert_not_called()


def test_profile_propagates_cleanup_only_failure(profile_owner: AnemoiProfiler) -> None:
    """A failed finalization must not produce a successful report after training returns."""
    failure = RuntimeError("capture close failed")
    profile_owner.profiler.teardown.side_effect = failure

    with pytest.raises(RuntimeError, match="capture close failed") as raised:
        profile_owner.profile()

    assert raised.value is failure
    profile_owner.train.assert_called_once_with()
    profile_owner.report.assert_not_called()
    profile_owner.export_to_logger.assert_not_called()


def test_profile_preserves_training_and_cleanup_failures(
    profile_owner: AnemoiProfiler,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The training error remains primary, with the secondary cleanup diagnostic retained."""
    training_failure = ValueError("training failed")
    cleanup_failure = RuntimeError("capture close failed")
    profile_owner.train.side_effect = training_failure
    profile_owner.profiler.teardown.side_effect = cleanup_failure

    with pytest.raises(ValueError, match="training failed") as raised:
        profile_owner.profile()

    assert raised.value is training_failure
    assert any("capture close failed" in note for note in training_failure.__notes__)
    assert "Failed to finalize the profiler after a training failure" in caplog.text
    assert "RuntimeError: capture close failed" in caplog.text
    profile_owner.profiler.teardown.assert_called_once_with(stage="fit")
    profile_owner.report.assert_not_called()
    profile_owner.export_to_logger.assert_not_called()
