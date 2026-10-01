# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``compute_valid_data_indices``: the valid sample indices computed from data readers.

``MultiDataset`` samples a date index ``t`` only if every reader can provide the dates
``t + i`` for all of its relative offsets ``i``. These tests drive
``compute_valid_data_indices`` through gridded and trajectory readers.
"""

import datetime

import numpy as np
import pytest

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.data_reader import TrajectoryDataset
from anemoi.training.data.usable_indices import compute_valid_data_indices

FREQUENCY = datetime.timedelta(hours=6)
START = np.datetime64("2020-01-01T00:00:00", "s")

# ---------------------------------------------------------------------------
# Helpers: lightweight stubs that bypass open_dataset
# ---------------------------------------------------------------------------


class _FakeData:
    """Minimal stand-in for an anemoi-datasets dataset: dates, missing dates and frequency.

    ``name_to_index`` and ``resolution`` are only read by the reader's ``repr`` in error messages.
    """

    def __init__(self, length: int, missing: set[int]) -> None:
        self.dates = START + np.arange(length) * np.timedelta64(FREQUENCY)
        self.missing = missing
        self.frequency = FREQUENCY
        self.name_to_index: dict[str, int] = {}
        self.resolution = "fake"


def _make_gridded_reader(length: int, missing: set[int] | None = None) -> GriddedDataReader:
    """Return a GriddedDataReader backed by a fake single-sequence dataset."""
    reader = GriddedDataReader.__new__(GriddedDataReader)
    reader.data = _FakeData(length, missing or set())
    return reader


def _make_trajectory_reader(
    num_trajectories: int,
    trajectory_length: int,
    trajectory_start: np.datetime64 = START,
    missing: set[int] | None = None,
) -> TrajectoryDataset:
    """Return a TrajectoryDataset over consecutive forecast runs of ``trajectory_length`` steps."""
    reader = TrajectoryDataset.__new__(TrajectoryDataset)
    reader.data = _FakeData(num_trajectories * trajectory_length, missing or set())
    reader.trajectory_start = trajectory_start.astype(datetime.datetime)
    reader.trajectory_length = trajectory_length
    return reader


def _valid_indices(reader: BaseAnemoiReader, offsets: list[int]) -> np.ndarray:
    return compute_valid_data_indices({"data": reader}, {"data": offsets})


# ---------------------------------------------------------------------------
# Tests: GriddedDataReader (single sequence)
# ---------------------------------------------------------------------------


class TestGriddedValidIndices:
    """Valid sample indices for a single-sequence (analysis) reader."""

    def test_all_positions_with_a_full_window_are_valid(self) -> None:
        """Every index whose whole window lies inside the series is valid."""
        reader = _make_gridded_reader(length=10)
        # window offsets [0, 1, 2] → valid positions 0..7
        np.testing.assert_array_equal(_valid_indices(reader, [0, 1, 2]), np.arange(8))

    def test_missing_dates_are_excluded(self) -> None:
        """Indices whose window covers a missing date must not be returned."""
        reader = _make_gridded_reader(length=10, missing={4})
        # window [0, 1, 2]: positions 2, 3 and 4 would read date 4
        np.testing.assert_array_equal(_valid_indices(reader, [0, 1, 2]), [0, 1, 5, 6, 7])

    def test_non_contiguous_offsets(self) -> None:
        """Offsets need not be contiguous; the window spans max - min + 1 dates."""
        reader = _make_gridded_reader(length=20)
        # offsets [0, 6] → valid 0..13
        np.testing.assert_array_equal(_valid_indices(reader, [0, 6]), np.arange(14))

    def test_missing_date_between_non_contiguous_offsets_is_jumped(self) -> None:
        """A missing date that no offset reads does not invalidate the index."""
        reader = _make_gridded_reader(length=20, missing={3})
        indices = _valid_indices(reader, [0, 6])
        assert 0 in indices  # reads dates 0 and 6, skips 3
        assert 3 not in indices  # reads date 3

    def test_negative_offsets(self) -> None:
        """Offsets before the anchor (e.g. input steps) shift the first valid index."""
        reader = _make_gridded_reader(length=10)
        # offsets [-2, -1, 0] → valid 2..9
        np.testing.assert_array_equal(_valid_indices(reader, [-2, -1, 0]), np.arange(2, 10))

    def test_series_shorter_than_window_raises(self) -> None:
        """A series too short for the window has no valid index."""
        reader = _make_gridded_reader(length=2)
        with pytest.raises(ValueError, match="No valid date indices found for data reader 'data'"):
            _valid_indices(reader, [0, 1, 2])


# ---------------------------------------------------------------------------
# Tests: TrajectoryDataset (forecast runs)
# ---------------------------------------------------------------------------


class TestTrajectoryValidIndices:
    """Valid sample indices for a reader of consecutive forecast runs."""

    def test_windows_do_not_cross_trajectories(self) -> None:
        """A window must lie within a single forecast run."""
        # 3 runs of 4 steps: dates 0-3, 4-7, 8-11
        reader = _make_trajectory_reader(num_trajectories=3, trajectory_length=4)
        # window [0, 1]: positions 3 and 7 would span two runs
        np.testing.assert_array_equal(_valid_indices(reader, [0, 1]), [0, 1, 2, 4, 5, 6, 8, 9, 10])

    def test_window_as_long_as_a_trajectory_gives_one_index_per_run(self) -> None:
        """A window covering a whole run can only start at the run's first step."""
        reader = _make_trajectory_reader(num_trajectories=3, trajectory_length=4)
        np.testing.assert_array_equal(_valid_indices(reader, [0, 1, 2, 3]), [0, 4, 8])

    def test_trajectory_start_offsets_run_boundaries(self) -> None:
        """Runs are counted from ``trajectory_start``, not from the first date."""
        # Runs start one step after the first date: run boundaries fall at dates 1, 5, 9
        reader = _make_trajectory_reader(
            num_trajectories=3,
            trajectory_length=4,
            trajectory_start=START + np.timedelta64(FREQUENCY),
        )
        np.testing.assert_array_equal(_valid_indices(reader, [0, 1, 2, 3]), [1, 5])

    def test_missing_dates_are_excluded_within_runs(self) -> None:
        """Missing dates exclude windows in addition to the run boundaries."""
        reader = _make_trajectory_reader(num_trajectories=2, trajectory_length=4, missing={5})
        # run 0: 0, 1, 2 valid; run 1: windows [4, 5] and [5, 6] read date 5 → only 6
        np.testing.assert_array_equal(_valid_indices(reader, [0, 1]), [0, 1, 2, 6])


# ---------------------------------------------------------------------------
# Tests: several readers
# ---------------------------------------------------------------------------


class TestMultipleReaders:
    """The valid indices are the intersection over all readers."""

    def test_intersection_across_readers(self) -> None:
        """An index is valid only if it is valid for every reader and its own offsets."""
        readers = {
            "a": _make_gridded_reader(length=10),
            "b": _make_gridded_reader(length=10, missing={6}),
        }
        offsets = {"a": [0, 1, 2], "b": [0, 1]}
        # a: 0..7; b: 0..8 without 5, 6 → [0, 1, 2, 3, 4, 7]
        np.testing.assert_array_equal(compute_valid_data_indices(readers, offsets), [0, 1, 2, 3, 4, 7])

    def test_reader_without_valid_index_raises(self) -> None:
        """A reader with no valid index fails loudly, naming the reader."""
        readers = {"a": _make_gridded_reader(length=10), "short": _make_gridded_reader(length=2)}
        offsets = {"a": [0, 1], "short": [0, 1, 2]}
        with pytest.raises(ValueError, match="'short'"):
            compute_valid_data_indices(readers, offsets)

    def test_empty_intersection_raises(self) -> None:
        """Readers whose valid indices do not overlap leave nothing to sample."""
        readers = {
            "a": _make_gridded_reader(length=10, missing={5, 6, 7, 8, 9}),
            "b": _make_gridded_reader(length=10, missing={0, 1, 2, 3, 4}),
        }
        offsets = {"a": [0], "b": [0]}
        with pytest.raises(ValueError, match="after intersection"):
            compute_valid_data_indices(readers, offsets)


# ---------------------------------------------------------------------------
# Tests: output contract
# ---------------------------------------------------------------------------


class TestValidIndicesContract:
    """Invariants that must hold for all valid inputs."""

    @pytest.mark.parametrize(
        ("length", "offsets"),
        [
            (20, [0, 1, 2]),
            (20, [0, 6]),
            (20, [-3, 0, 3]),
            (6, [0, 1, 2, 3, 4, 5]),
        ],
    )
    def test_output_is_sorted_unique_1d_integers(self, length: int, offsets: list[int]) -> None:
        indices = _valid_indices(_make_gridded_reader(length), offsets)
        assert indices.ndim == 1
        assert np.issubdtype(indices.dtype, np.integer)
        np.testing.assert_array_equal(indices, np.unique(indices))

    @pytest.mark.parametrize("offsets", [[0, 1, 2], [-2, 0, 2], [0, 6]])
    def test_every_index_reads_dates_inside_the_series(self, offsets: list[int]) -> None:
        length = 15
        indices = _valid_indices(_make_gridded_reader(length), offsets)
        read = indices[:, None] + np.asarray(offsets)[None, :]
        assert read.min() >= 0
        assert read.max() < length
