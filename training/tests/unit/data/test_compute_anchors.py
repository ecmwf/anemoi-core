# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for reader anchors and their time-aligned intersection across readers."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.data_reader import TrajectoryDataReader
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.usable_indices import Anchors
from anemoi.training.data.usable_indices import compute_valid_anchors

T0 = np.datetime64("2020-01-01T00:00:00", "s")
HOUR = datetime.timedelta(hours=1)


def _gridded(
    length: int,
    frequency: datetime.timedelta = 6 * HOUR,
    start: np.datetime64 = T0,
    missing: set[int] | None = None,
) -> GriddedDataReader:
    """A gridded reader over ``length`` dates every ``frequency`` from ``start``."""
    reader = GriddedDataReader.__new__(GriddedDataReader)
    reader.data = SimpleNamespace(
        dates=start + np.arange(length) * np.timedelta64(frequency),
        missing=missing or set(),
        frequency=frequency,
        name_to_index={},
        resolution="fake",
    )
    reader.sampling = None
    return reader


def _trajectory(
    num_base_dates: int,
    num_steps: int,
    base_frequency: datetime.timedelta = 24 * HOUR,
    step_frequency: datetime.timedelta = 6 * HOUR,
    first_step: datetime.timedelta = datetime.timedelta(0),
    missing: set[int] | None = None,
    sampling: dict | None = None,
) -> TrajectoryDataReader:
    """A trajectory reader with ``num_base_dates`` forecasts of ``num_steps`` steps each."""
    reader = TrajectoryDataReader.__new__(TrajectoryDataReader)
    reader.data = SimpleNamespace(
        shape=(num_base_dates, 1, 1, num_steps, 1),
        missing=missing or set(),
        base_dates=T0 + np.arange(num_base_dates) * np.timedelta64(base_frequency),
        steps=np.timedelta64(first_step) + np.arange(num_steps) * np.timedelta64(step_frequency),
        step_frequency=step_frequency,
    )
    reader.sampling = sampling if sampling is not None else {"stride": None}
    return reader


def _anchors(readers: dict, offsets: dict, stride: int | None = 1) -> Anchors:
    return compute_valid_anchors(readers, offsets, stride=stride)


class TestReaderValidAnchors:

    def test_gridded_anchor_times_are_its_dates(self) -> None:
        reader = _gridded(length=10)
        table = reader.valid_anchors([0, 1, 2])
        np.testing.assert_array_equal(table.positions, np.arange(8))
        np.testing.assert_array_equal(table.sequences, 0)
        np.testing.assert_array_equal(table.times, reader.data.dates[:8])
        assert table.base_dates is None

    def test_gridded_skips_windows_reading_missing_dates(self) -> None:
        table = _gridded(length=10, missing={4}).valid_anchors([0, 1, 2])
        assert not {2, 3, 4} & set(table.positions.tolist())

    def test_trajectory_anchor_times_are_base_plus_step(self) -> None:
        reader = _trajectory(num_base_dates=2, num_steps=4, first_step=12 * HOUR)
        table = reader.valid_anchors([0, 1])
        np.testing.assert_array_equal(table.sequences, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(table.positions, [0, 1, 2, 0, 1, 2])
        expected = table.base_dates + reader.data.steps[table.positions].astype("timedelta64[s]")
        np.testing.assert_array_equal(table.times, expected)
        assert table.times[0] == T0 + np.timedelta64(12, "h")

    def test_trajectory_skips_missing_base_dates(self) -> None:
        table = _trajectory(num_base_dates=4, num_steps=3, missing={1, 3}).valid_anchors([0])
        assert set(table.sequences.tolist()) == {0, 2}


class TestSingleReaderStride:
    """With one reader the stride counts that reader's positions, as before."""

    @pytest.mark.parametrize(
        ("length", "offsets", "stride", "expected"),
        [
            (10, [0, 1, 2], 1, list(range(8))),
            (10, [0, 1, 2], None, [0, 3, 6]),
            (10, [0, 1], 2, [0, 2, 4, 6, 8]),
            (20, [0, 6], None, [0, 7]),
            (10, [0, 1], 100, [0]),
        ],
    )
    def test_gridded_stride(self, length: int, offsets: list[int], stride: int | None, expected: list[int]) -> None:
        anchors = _anchors({"a": _gridded(length)}, {"a": offsets}, stride=stride)
        np.testing.assert_array_equal(anchors.rows["a"][:, 1], expected)

    def test_trajectory_stride_restarts_for_each_base_date(self) -> None:
        anchors = _anchors({"f": _trajectory(num_base_dates=2, num_steps=18)}, {"f": list(range(7))}, stride=6)
        rows = anchors.rows["f"]
        np.testing.assert_array_equal(rows[:, 0], [0, 0, 1, 1])
        np.testing.assert_array_equal(rows[:, 1], [0, 6, 0, 6])

    def test_trajectory_non_overlapping_gives_one_anchor_per_forecast(self) -> None:
        anchors = _anchors({"f": _trajectory(num_base_dates=3, num_steps=10)}, {"f": list(range(7))}, stride=None)
        np.testing.assert_array_equal(anchors.rows["f"], [[0, 0], [1, 0], [2, 0]])

    def test_invalid_stride_raises(self) -> None:
        with pytest.raises(ValueError, match="stride must be >= 1"):
            _anchors({"a": _gridded(10)}, {"a": [0, 1]}, stride=0)

    def test_window_longer_than_series_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid anchors found for data reader 'a'"):
            _anchors({"a": _gridded(2)}, {"a": [0, 1, 2]})


class TestAlignmentAcrossReaders:
    """Readers are aligned on time, so they may differ in frequency and date range."""

    def test_different_frequencies_align_on_common_times(self) -> None:
        # offsets [0, 6h]: the 6-hourly reader reads [p, p+1], the hourly reader [p, p+6]
        readers = {"coarse": _gridded(5, frequency=6 * HOUR), "fine": _gridded(30, frequency=HOUR)}
        anchors = _anchors(readers, {"coarse": [0, 1], "fine": [0, 6]})

        np.testing.assert_array_equal(anchors.times, T0 + np.arange(4) * np.timedelta64(6, "h"))
        np.testing.assert_array_equal(anchors.rows["coarse"][:, 1], [0, 1, 2, 3])
        np.testing.assert_array_equal(anchors.rows["fine"][:, 1], [0, 6, 12, 18])

    def test_different_start_dates_shift_positions(self) -> None:
        later = T0 + np.timedelta64(12, "h")
        readers = {"a": _gridded(10), "b": _gridded(10, start=later)}
        anchors = _anchors(readers, {"a": [0], "b": [0]})

        assert anchors.times[0] == later
        np.testing.assert_array_equal(anchors.rows["a"][:, 1], np.arange(2, 10))
        np.testing.assert_array_equal(anchors.rows["b"][:, 1], np.arange(0, 8))

    def test_missing_dates_of_any_reader_remove_the_anchor(self) -> None:
        readers = {"a": _gridded(10), "b": _gridded(10, missing={3})}
        anchors = _anchors(readers, {"a": [0], "b": [0]})
        assert 3 not in anchors.rows["a"][:, 1]

    def test_disjoint_time_axes_raise(self) -> None:
        readers = {"a": _gridded(10), "b": _gridded(10, start=T0 + np.timedelta64(3, "h"))}
        with pytest.raises(ValueError, match="No valid anchors found after intersection"):
            _anchors(readers, {"a": [0], "b": [0]})

    def test_trajectory_and_analysis_align_on_valid_time(self) -> None:
        # forecasts every 24h with 6h steps; analysis every 6h
        readers = {"fc": _trajectory(num_base_dates=2, num_steps=4), "an": _gridded(12)}
        anchors = _anchors(readers, {"fc": [0, 1], "an": [0, 1]})

        np.testing.assert_array_equal(anchors.rows["fc"], [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        # analysis position = valid time of the forecast anchor, in 6h steps from T0
        np.testing.assert_array_equal(anchors.rows["an"][:, 1], [0, 1, 2, 4, 5, 6])
        np.testing.assert_array_equal(anchors.base_dates, np.repeat(T0 + np.arange(2) * np.timedelta64(24, "h"), 3))

    def test_stride_counts_steps_of_the_shared_grid(self) -> None:
        # shared grid is 6h (lcm of 6h and 1h); stride 2 keeps every 12h
        readers = {"coarse": _gridded(9, frequency=6 * HOUR), "fine": _gridded(60, frequency=HOUR)}
        anchors = _anchors(readers, {"coarse": [0], "fine": [0]}, stride=2)
        np.testing.assert_array_equal(anchors.rows["coarse"][:, 1], [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(anchors.rows["fine"][:, 1], [0, 12, 24, 36, 48])

    def test_non_overlapping_uses_the_widest_window(self) -> None:
        # the coarse window spans 6h and the fine one 11h → anchors at least 12h apart
        readers = {"coarse": _gridded(9, frequency=6 * HOUR), "fine": _gridded(60, frequency=HOUR)}
        anchors = _anchors(readers, {"coarse": [0, 1], "fine": list(range(12))}, stride=None)
        np.testing.assert_array_equal(anchors.rows["coarse"][:, 1], [0, 2, 4, 6])


class TestMultiDatasetSamplingStride:

    def test_unconfigured_readers_use_stride_1(self) -> None:
        assert MultiDataset._sampling_stride({"a": _gridded(5), "b": _gridded(5)}) == 1

    def test_configured_reader_sets_the_stride(self) -> None:
        readers = {"an": _gridded(5), "fc": _trajectory(2, 4, sampling={"stride": 3})}
        assert MultiDataset._sampling_stride(readers) == 3

    def test_default_trajectory_sampling_is_non_overlapping(self) -> None:
        assert MultiDataset._sampling_stride({"fc": _trajectory(2, 4)}) is None

    def test_disagreeing_readers_raise(self) -> None:
        readers = {"f1": _trajectory(2, 4, sampling={"stride": 1}), "f2": _trajectory(2, 4)}
        with pytest.raises(ValueError, match="disagree on the sampling stride"):
            MultiDataset._sampling_stride(readers)
