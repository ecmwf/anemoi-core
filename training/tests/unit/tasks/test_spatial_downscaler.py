# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest
import torch

from anemoi.training.tasks.spatial_downscaler import SpatialDownscaler

# ---------------------------------------------------------------------------
# Offset configuration
# ---------------------------------------------------------------------------


def test_default_offsets_are_single_zero() -> None:
    """Without explicit offsets, both input and output default to [timedelta(0)]."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
    )
    assert task._input_offsets == [datetime.timedelta(0)]
    assert task._output_offsets == [datetime.timedelta(0)]
    assert task._offsets == [datetime.timedelta(0)]


def test_multiple_offsets_parsed_correctly() -> None:
    """Three offsets are parsed, sorted, and shared between input and output."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        offsets=["0H", "6H", "12H"],
    )
    expected = [
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]
    assert task._input_offsets == expected
    assert task._output_offsets == expected
    assert task._offsets == expected


def test_input_and_output_offsets_are_always_equal() -> None:
    """The invariant ``input_offsets == output_offsets`` must hold for any offset list."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        offsets=["3H", "0H", "6H"],  # unsorted on purpose
    )
    assert task._input_offsets == task._output_offsets


# ---------------------------------------------------------------------------
# Batch index helpers
# ---------------------------------------------------------------------------


def test_batch_input_indices_match_batch_output_indices() -> None:
    """For multiple offsets, input and output batch positions are identical."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        offsets=["0H", "6H"],
    )
    assert task.get_batch_input_indices() == task.get_batch_output_indices()


def test_batch_indices_are_zero_indexed_positions() -> None:
    """Batch indices map each offset to its position in the sorted offset list."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        offsets=["0H", "6H", "12H"],
    )
    # All offsets go into a single shared _offsets list, so positions are 0, 1, 2.
    assert task.get_batch_input_indices() == [0, 1, 2]


# ---------------------------------------------------------------------------
# get_inputs / get_targets behaviour
# ---------------------------------------------------------------------------


def _make_batch(dataset_names: list[str], num_times: int, grid: int, nvar: int) -> dict[str, torch.Tensor]:
    """Create a minimal fake batch with shape (bs=1, num_times, ensemble=1, grid, nvar)."""
    return {name: torch.randn(1, num_times, 1, grid, nvar) for name in dataset_names}


class _FakeIndices:
    """Minimal stand-in for IndexCollection that exposes data.input.full."""

    def __init__(self, nvar: int) -> None:
        self.data = _FakeData(nvar)


class _FakeData:
    def __init__(self, nvar: int) -> None:
        self.input = _FakeInput(nvar)


class _FakeInput:
    def __init__(self, nvar: int) -> None:
        self.full = torch.arange(nvar)


def test_get_inputs_filters_to_input_datasets_only() -> None:
    """get_inputs returns only input_datasets and skips target_datasets."""
    task = SpatialDownscaler(
        input_datasets=["in_lres", "in_hres"],
        target_datasets=["out_hres"],
    )
    batch = _make_batch(["in_lres", "in_hres", "out_hres"], num_times=1, grid=10, nvar=4)
    data_indices = {name: _FakeIndices(4) for name in batch}
    x = task.get_inputs(batch, data_indices=data_indices)
    assert set(x.keys()) == {"in_lres", "in_hres"}
    assert "out_hres" not in x


def test_get_targets_filters_to_target_datasets_only() -> None:
    """get_targets returns only target_datasets and skips input-only datasets."""
    task = SpatialDownscaler(
        input_datasets=["in_lres", "in_hres"],
        target_datasets=["out_hres"],
    )
    batch = _make_batch(["in_lres", "in_hres", "out_hres"], num_times=1, grid=10, nvar=4)
    y = task.get_targets(batch)
    assert set(y.keys()) == {"out_hres"}
    assert "in_lres" not in y


def test_get_inputs_multi_offset_time_dimension() -> None:
    """With N offsets, the time dimension of extracted inputs has length N."""
    n_offsets = 3
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        offsets=["0H", "6H", "12H"],
    )
    batch = _make_batch(["in_lres", "out_hres"], num_times=n_offsets, grid=8, nvar=5)
    data_indices = {"in_lres": _FakeIndices(5), "out_hres": _FakeIndices(5)}
    x = task.get_inputs(batch, data_indices=data_indices)
    assert x["in_lres"].shape[1] == n_offsets


def test_get_targets_multi_offset_time_dimension() -> None:
    """With N offsets, the time dimension of extracted targets has length N."""
    n_offsets = 3
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        offsets=["0H", "6H", "12H"],
    )
    batch = _make_batch(["in_lres", "out_hres"], num_times=n_offsets, grid=8, nvar=5)
    y = task.get_targets(batch)
    assert y["out_hres"].shape[1] == n_offsets


def test_get_inputs_missing_dataset_is_skipped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A missing input dataset logs a warning and is absent from the output."""
    import logging

    task = SpatialDownscaler(
        input_datasets=["in_lres", "in_missing"],
        target_datasets=["out_hres"],
    )
    batch = _make_batch(["in_lres", "out_hres"], num_times=1, grid=4, nvar=3)
    data_indices = {"in_lres": _FakeIndices(3), "out_hres": _FakeIndices(3)}
    with caplog.at_level(logging.WARNING):
        x = task.get_inputs(batch, data_indices=data_indices)
    assert "in_missing" not in x
    assert any("in_missing" in record.message for record in caplog.records)
