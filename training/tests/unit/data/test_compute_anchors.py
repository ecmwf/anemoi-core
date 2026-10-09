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


# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``BaseAnemoiReader.compute_anchors`` and ``compute_valid_anchors``.

``MultiDataset`` samples a ``(sequence, position)`` anchor only if every reader can
provide the positions ``position + i`` of that sequence for all of its relative offsets ``i``.
"""

import datetime

import numpy as np
import pytest

from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.data_reader import TrajectoryDataReader

# ---------------------------------------------------------------------------
# Helpers: lightweight stubs that bypass open_dataset
# ---------------------------------------------------------------------------


class _FakeGriddedData:
    """Minimal stand-in for a single-sequence anemoi-datasets dataset.

    ``frequency``, ``name_to_index`` and ``resolution`` are only read by the reader's ``repr`` in error messages.
    """

    def __init__(self, length: int, missing: set[int]) -> None:
        self.dates = np.arange(length)
        self.missing = missing
        self.frequency = datetime.timedelta(hours=6)
        self.name_to_index: dict[str, int] = {}
        self.resolution = "fake"


class _FakeTrajectoryData:
    """Minimal stand-in for a 5-D ``(base_dates, variables, ensembles, steps, cells)`` trajectory dataset."""

    def __init__(self, num_sequences: int, steps_per_sequence: int, missing_sequences: set[int]) -> None:
        self.shape = (num_sequences, 1, 1, steps_per_sequence, 1)
        self.missing = missing_sequences


def _make_gridded_reader(length: int, missing: set[int] | None = None) -> GriddedDataReader:
    """Return a GriddedDataReader backed by a fake single-sequence dataset."""
    reader = GriddedDataReader.__new__(GriddedDataReader)
    reader.data = _FakeGriddedData(length, missing or set())
    reader.default_sampling = {"stride": 1}
    return reader


def _make_trajectory_reader(
    num_sequences: int,
    steps_per_sequence: int,
    missing_sequences: set[int] | None = None,
    sampling: dict | None = None,
) -> TrajectoryDataReader:
    """Return a TrajectoryDataReader backed by a fake multi-sequence dataset."""
    reader = TrajectoryDataReader.__new__(TrajectoryDataReader)
    reader.data = _FakeTrajectoryData(num_sequences, steps_per_sequence, missing_sequences or set())
    reader.default_sampling = sampling if sampling is not None else {"stride": None}
    return reader


# ---------------------------------------------------------------------------
# Tests: GriddedDataReader (single sequence)
# ---------------------------------------------------------------------------


class TestGriddedComputeAnchors:
    """compute_anchors on a single-sequence (analysis) reader."""

    def test_stride_1_returns_all_valid_positions(self) -> None:
        """stride=1 should produce one anchor per valid position."""
        reader = _make_gridded_reader(length=10)
        # window offsets [0, 1, 2] → window=3 → valid positions 0..7
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert anchors.shape == (8, 2)
        assert np.all(anchors[:, 0] == 0)  # single sequence id = 0
        np.testing.assert_array_equal(anchors[:, 1], np.arange(8))

    def test_stride_none_equals_window_size(self) -> None:
        """stride=None should use window size → non-overlapping anchors."""
        reader = _make_gridded_reader(length=10)
        # window offsets [0, 1, 2] → window=3 → valid 0..7 → stride=3 → [0, 3, 6]
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": None})
        np.testing.assert_array_equal(anchors[:, 1], [0, 3, 6])

    def test_stride_n_selects_every_nth_anchor(self) -> None:
        """stride=2 should keep every 2nd valid position."""
        reader = _make_gridded_reader(length=10)
        # window [0, 1] → window=2 → valid 0..8 → stride=2 → [0, 2, 4, 6, 8]
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 2})
        np.testing.assert_array_equal(anchors[:, 1], [0, 2, 4, 6, 8])

    def test_stride_larger_than_series_gives_single_anchor(self) -> None:
        """A stride larger than the series should yield at most one anchor."""
        reader = _make_gridded_reader(length=5)
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 100})
        assert len(anchors) == 1
        assert anchors[0, 1] == 0

    def test_default_sampling_is_stride_1(self) -> None:
        """compute_anchors without explicit sampling should use the reader's default (stride 1)."""
        reader = _make_gridded_reader(length=6)
        anchors_default = reader.compute_anchors([0, 1])
        anchors_stride1 = reader.compute_anchors([0, 1], sampling={"stride": 1})
        np.testing.assert_array_equal(anchors_default, anchors_stride1)

    def test_missing_positions_are_excluded(self) -> None:
        """Positions whose window covers a missing date must not be returned."""
        reader = _make_gridded_reader(length=10, missing={4})
        # window [0, 1, 2]: positions 2, 3 and 4 would read date 4
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        np.testing.assert_array_equal(anchors[:, 1], [0, 1, 5, 6, 7])

    def test_missing_date_between_non_contiguous_offsets_is_jumped(self) -> None:
        """A missing date that no offset reads does not invalidate the anchor."""
        reader = _make_gridded_reader(length=20, missing={3})
        positions = reader.compute_anchors([0, 6], sampling={"stride": 1})[:, 1]
        assert 0 in positions  # reads dates 0 and 6, skips 3
        assert 3 not in positions  # reads date 3

    def test_negative_offsets(self) -> None:
        """Offsets before the anchor (e.g. input steps) shift the first valid position."""
        reader = _make_gridded_reader(length=10)
        # offsets [-2, -1, 0] → valid 2..9
        anchors = reader.compute_anchors([-2, -1, 0], sampling={"stride": 1})
        np.testing.assert_array_equal(anchors[:, 1], np.arange(2, 10))

    def test_invalid_stride_raises(self) -> None:
        """Stride < 1 must raise ValueError."""
        reader = _make_gridded_reader(length=10)
        with pytest.raises(ValueError, match="stride must be >= 1"):
            reader.compute_anchors([0, 1], sampling={"stride": 0})

    def test_non_contiguous_offsets(self) -> None:
        """Offsets don't have to be contiguous; window covers max-min+1."""
        reader = _make_gridded_reader(length=20)
        # offsets [0, 6] → window=7 → valid 0..13 → stride=None → [0, 7]
        anchors = reader.compute_anchors([0, 6], sampling={"stride": None})
        np.testing.assert_array_equal(anchors[:, 1], [0, 7])

    def test_empty_series_returns_empty(self) -> None:
        """A series too short for the window should return an empty array."""
        reader = _make_gridded_reader(length=2)
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert anchors.shape == (0, 2)


# ---------------------------------------------------------------------------
# Tests: TrajectoryDataReader (multi sequence)
# ---------------------------------------------------------------------------


class TestTrajectoryComputeAnchors:
    """compute_anchors on a multi-sequence (forecast trajectory) reader."""

    def test_default_sampling_is_non_overlapping(self) -> None:
        """TrajectoryDataReader default should be stride=None → non-overlapping."""
        reader = _make_trajectory_reader(num_sequences=3, steps_per_sequence=10)
        # window offsets [0..6] → window=7; stride=None=7 → one anchor per sequence
        anchors = reader.compute_anchors(list(range(7)))
        assert len(anchors) == 3
        np.testing.assert_array_equal(anchors[:, 0], [0, 1, 2])  # sequence ids
        assert np.all(anchors[:, 1] == 0)  # each at first valid position

    def test_stride_1_samples_all_positions_across_all_sequences(self) -> None:
        """stride=1 should return every valid position in every sequence."""
        # 4 sequences, 10 steps, window=3 → 8 valid positions per sequence → 32 total
        reader = _make_trajectory_reader(num_sequences=4, steps_per_sequence=10)
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert len(anchors) == 4 * 8
        np.testing.assert_array_equal(anchors[:, 0], np.repeat(np.arange(4), 8))

    def test_windows_never_cross_sequences(self) -> None:
        """Every anchor's window lies inside its own sequence."""
        reader = _make_trajectory_reader(num_sequences=3, steps_per_sequence=4)
        # window [0, 1]: positions 0..2 in each sequence, never position 3
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 1})
        np.testing.assert_array_equal(anchors[:, 1], np.tile([0, 1, 2], 3))

    def test_stride_6_across_multiple_sequences(self) -> None:
        """stride=6 should step anchors by 6 within each sequence."""
        # 2 sequences, 18 steps, window=7 → valid 0..11; stride=6 → [0, 6]
        reader = _make_trajectory_reader(num_sequences=2, steps_per_sequence=18)
        anchors = reader.compute_anchors(list(range(7)), sampling={"stride": 6})
        np.testing.assert_array_equal(anchors[anchors[:, 0] == 0, 1], [0, 6])
        np.testing.assert_array_equal(anchors[anchors[:, 0] == 1, 1], [0, 6])

    def test_missing_sequence_is_skipped(self) -> None:
        """Sequences listed in missing_sequences should produce no anchors."""
        reader = _make_trajectory_reader(num_sequences=4, steps_per_sequence=10, missing_sequences={1, 3})
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 1})
        assert set(anchors[:, 0].tolist()) == {0, 2}

    def test_custom_sampling_overrides_default(self) -> None:
        """Passing sampling to compute_anchors should override default_sampling."""
        reader = _make_trajectory_reader(num_sequences=1, steps_per_sequence=10)
        anchors_override = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        anchors_default = reader.compute_anchors([0, 1, 2])  # uses stride=None
        assert len(anchors_override) > len(anchors_default)

    def test_explicit_stride_stored_as_default(self) -> None:
        """A configured sampling stride becomes the reader's default_sampling."""
        reader = _make_trajectory_reader(num_sequences=2, steps_per_sequence=12, sampling={"stride": 3})
        assert reader.default_sampling == {"stride": 3}
        anchors = reader.compute_anchors(list(range(7)))
        np.testing.assert_array_equal(anchors[anchors[:, 0] == 0, 1], [0, 3])


# ---------------------------------------------------------------------------
# Tests: compute_valid_anchors over several readers
# ---------------------------------------------------------------------------


class TestComputeValidAnchors:
    """The valid anchors are the intersection over all readers."""

    def test_intersection_across_readers(self) -> None:
        """An anchor is valid only if it is valid for every reader and its own offsets."""
        readers = {
            "a": _make_gridded_reader(length=10),
            "b": _make_gridded_reader(length=10, missing={6}),
        }
        offsets = {"a": [0, 1, 2], "b": [0, 1]}
        # a: 0..7; b: 0..8 without 5, 6 → [0, 1, 2, 3, 4, 7]
        anchors = compute_valid_anchors(readers, offsets)
        np.testing.assert_array_equal(anchors[:, 0], 0)
        np.testing.assert_array_equal(anchors[:, 1], [0, 1, 2, 3, 4, 7])

    def test_intersection_keeps_sequence_identity(self) -> None:
        """Anchors from different sequences with the same position are not merged."""
        readers = {
            "a": _make_trajectory_reader(num_sequences=3, steps_per_sequence=4, sampling={"stride": 1}),
            "b": _make_trajectory_reader(
                num_sequences=3,
                steps_per_sequence=4,
                missing_sequences={1},
                sampling={"stride": 1},
            ),
        }
        offsets = {"a": [0, 1], "b": [0, 1]}
        anchors = compute_valid_anchors(readers, offsets)
        expected = np.array([[0, 0], [0, 1], [0, 2], [2, 0], [2, 1], [2, 2]])
        np.testing.assert_array_equal(anchors, expected)

    def test_reader_without_valid_anchor_raises(self) -> None:
        """A reader with no valid anchor fails loudly, naming the reader."""
        readers = {"a": _make_gridded_reader(length=10), "short": _make_gridded_reader(length=2)}
        offsets = {"a": [0, 1], "short": [0, 1, 2]}
        with pytest.raises(ValueError, match="No valid anchors found for data reader 'short'"):
            compute_valid_anchors(readers, offsets)

    def test_empty_intersection_raises(self) -> None:
        """Readers whose valid anchors do not overlap leave nothing to sample."""
        readers = {
            "a": _make_gridded_reader(length=10, missing={5, 6, 7, 8, 9}),
            "b": _make_gridded_reader(length=10, missing={0, 1, 2, 3, 4}),
        }
        offsets = {"a": [0], "b": [0]}
        with pytest.raises(ValueError, match="after intersection"):
            compute_valid_anchors(readers, offsets)


# ---------------------------------------------------------------------------
# Tests: anchor array shape contract
# ---------------------------------------------------------------------------


class TestAnchorArrayShape:
    """Invariants that must hold for all valid inputs."""

    @pytest.mark.parametrize(
        ("length", "offsets", "sampling"),
        [
            (20, [0, 1, 2], {"stride": 1}),
            (20, [0, 1, 2], {"stride": None}),
            (20, [0, 1, 2], {"stride": 4}),
            (20, [0, 6], {"stride": 1}),
            (5, [0, 1, 2, 3, 4, 5], {"stride": 1}),  # window > length → empty
        ],
    )
    def test_output_is_2d_with_two_columns(self, length: int, offsets: list, sampling: dict) -> None:
        reader = _make_gridded_reader(length)
        anchors = reader.compute_anchors(offsets, sampling=sampling)
        assert anchors.ndim == 2
        assert anchors.shape[1] == 2

    def test_output_dtype_is_int64(self) -> None:
        reader = _make_gridded_reader(10)
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 1})
        assert anchors.dtype == np.int64

    @pytest.mark.parametrize("offsets", [[0, 1, 2], [-2, 0, 2], [0, 6]])
    def test_every_anchor_reads_positions_inside_the_series(self, offsets: list[int]) -> None:
        length = 15
        positions = _make_gridded_reader(length).compute_anchors(offsets, sampling={"stride": 1})[:, 1]
        read = positions[:, None] + np.asarray(offsets)[None, :]
        assert read.min() >= 0
        assert read.max() < length
