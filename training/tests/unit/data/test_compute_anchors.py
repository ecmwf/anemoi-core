# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for BaseAnemoiReader.compute_anchors and the dict-based sampling API."""

import numpy as np
import pytest

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.data_reader import TrajectoryDataset

# ---------------------------------------------------------------------------
# Helpers: lightweight stubs that bypass open_dataset
# ---------------------------------------------------------------------------


def _make_native_reader(length: int, missing: set[int] | None = None) -> BaseAnemoiReader:
    """Return a NativeGridDataset-like reader backed by a fake single-sequence dataset."""
    reader = NativeGridDataset.__new__(NativeGridDataset)

    class _FakeData:
        def __init__(self, n: int, m: set[int]) -> None:
            self._n = n
            self._missing: set[int] = m

        @property
        def dates(self) -> np.ndarray:
            return np.arange(self._n)

        @property
        def missing(self) -> set[int]:
            return self._missing

    reader.data = _FakeData(length, missing or set())
    return reader


def _make_trajectory_reader(
    num_sequences: int,
    steps_per_sequence: int,
    missing_sequences: set[int] | None = None,
    sampling: dict | None = None,
) -> TrajectoryDataset:
    """Return a TrajectoryDataset-like reader backed by a fake multi-sequence dataset."""
    reader = TrajectoryDataset.__new__(TrajectoryDataset)

    class _FakeData:
        def __init__(self, n_seq: int, n_steps: int, m_seq: set[int]) -> None:
            self._shape = (n_seq, 1, 1, n_steps, 1)
            self._missing: set[int] = m_seq

        @property
        def shape(self) -> tuple:
            return self._shape

        @property
        def missing(self) -> set[int]:
            return self._missing

    reader.data = _FakeData(num_sequences, steps_per_sequence, missing_sequences or set())
    reader.default_sampling = sampling if sampling is not None else {"stride": None}
    return reader


# ---------------------------------------------------------------------------
# Tests: NativeGridDataset (single sequence)
# ---------------------------------------------------------------------------


class TestNativeGridComputeAnchors:
    """compute_anchors on a single-sequence (analysis) reader."""

    def test_stride_1_returns_all_valid_positions(self) -> None:
        """stride=1 should produce one anchor per valid position."""
        reader = _make_native_reader(length=10)
        # window offsets [0, 1, 2] → window=3 → valid positions 0..7
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert anchors.shape == (8, 2)
        assert np.all(anchors[:, 0] == 0)  # single sequence id = 0
        np.testing.assert_array_equal(anchors[:, 1], np.arange(8))

    def test_stride_none_equals_window_size(self) -> None:
        """stride=None should use window size → non-overlapping anchors."""
        reader = _make_native_reader(length=10)
        # window offsets [0, 1, 2] → window=3 → valid 0..7 → stride=3 → [0, 3, 6]
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": None})
        np.testing.assert_array_equal(anchors[:, 1], [0, 3, 6])

    def test_stride_n_selects_every_nth_anchor(self) -> None:
        """stride=2 should keep every 2nd valid position."""
        reader = _make_native_reader(length=10)
        # window [0,1] → window=2 → valid 0..8 → stride=2 → [0,2,4,6,8]
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 2})
        np.testing.assert_array_equal(anchors[:, 1], [0, 2, 4, 6, 8])

    def test_stride_larger_than_series_gives_single_anchor(self) -> None:
        """A stride larger than the series should yield at most one anchor."""
        reader = _make_native_reader(length=5)
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 100})
        assert len(anchors) == 1
        assert anchors[0, 1] == 0

    def test_default_sampling_is_stride_1(self) -> None:
        """NativeGridDataset default_sampling must be {"stride": 1}."""
        reader = _make_native_reader(length=6)
        # compute_anchors without explicit sampling should use default
        anchors_default = reader.compute_anchors([0, 1])
        anchors_stride1 = reader.compute_anchors([0, 1], sampling={"stride": 1})
        np.testing.assert_array_equal(anchors_default, anchors_stride1)

    def test_missing_positions_are_excluded(self) -> None:
        """Positions that cover a missing index must not be returned."""
        reader = _make_native_reader(length=10, missing={4})
        # window [0,1,2]: position 4 (and 2,3) are excluded because they'd cover missing=4
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        positions = anchors[:, 1]
        # None of these positions should require index 4
        assert 2 not in positions
        assert 3 not in positions
        assert 4 not in positions

    def test_invalid_stride_raises(self) -> None:
        """Stride < 1 must raise ValueError."""
        reader = _make_native_reader(length=10)
        with pytest.raises(ValueError, match="stride must be >= 1"):
            reader.compute_anchors([0, 1], sampling={"stride": 0})

    def test_non_contiguous_offsets(self) -> None:
        """Offsets don't have to be contiguous; window covers max-min+1."""
        reader = _make_native_reader(length=20)
        # offsets [0, 6] → window=7 → valid 0..13 → stride=None → [0, 7]
        anchors = reader.compute_anchors([0, 6], sampling={"stride": None})
        np.testing.assert_array_equal(anchors[:, 1], [0, 7])

    def test_empty_series_returns_empty(self) -> None:
        """A series too short for the window should return an empty array."""
        reader = _make_native_reader(length=2)
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert anchors.shape == (0, 2)


# ---------------------------------------------------------------------------
# Tests: TrajectoryDataset (multi-sequence)
# ---------------------------------------------------------------------------


class TestTrajectoryComputeAnchors:
    """compute_anchors on a multi-sequence (forecast trajectory) reader."""

    def test_default_sampling_is_non_overlapping(self) -> None:
        """TrajectoryDataset default should be stride=None → non-overlapping."""
        reader = _make_trajectory_reader(num_sequences=3, steps_per_sequence=10)
        # window offsets [0..6] → window=7; stride=None=7 → one anchor per sequence
        anchors = reader.compute_anchors(list(range(7)))
        # 3 sequences × 1 anchor each
        assert len(anchors) == 3
        np.testing.assert_array_equal(anchors[:, 0], [0, 1, 2])  # sequence ids
        assert np.all(anchors[:, 1] == 0)  # each at first valid position

    def test_stride_1_samples_all_positions_across_all_sequences(self) -> None:
        """stride=1 should return every valid position in every sequence."""
        # 4 sequences, 10 steps, window=3 → 8 valid positions per sequence → 32 total
        reader = _make_trajectory_reader(num_sequences=4, steps_per_sequence=10)
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert len(anchors) == 4 * 8
        # sequence ids should appear in sorted order, 8 times each
        expected_seqs = np.repeat(np.arange(4), 8)
        np.testing.assert_array_equal(anchors[:, 0], expected_seqs)

    def test_stride_6_across_multiple_sequences(self) -> None:
        """stride=6 should step anchors by 6 within each sequence."""
        # 2 sequences, 18 steps, window=7 → valid 0..11; stride=6 → [0, 6]
        reader = _make_trajectory_reader(num_sequences=2, steps_per_sequence=18)
        anchors = reader.compute_anchors(list(range(7)), sampling={"stride": 6})
        positions_seq0 = anchors[anchors[:, 0] == 0, 1]
        positions_seq1 = anchors[anchors[:, 0] == 1, 1]
        np.testing.assert_array_equal(positions_seq0, [0, 6])
        np.testing.assert_array_equal(positions_seq1, [0, 6])

    def test_missing_sequence_is_skipped(self) -> None:
        """Sequences listed in missing_sequences should produce no anchors."""
        reader = _make_trajectory_reader(num_sequences=4, steps_per_sequence=10, missing_sequences={1, 3})
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 1})
        seq_ids = set(anchors[:, 0].tolist())
        assert 1 not in seq_ids
        assert 3 not in seq_ids
        assert {0, 2} == seq_ids

    def test_custom_sampling_overrides_default(self) -> None:
        """Passing sampling to compute_anchors should override default_sampling."""
        # Default is stride=None (window), but we pass stride=1 explicitly
        reader = _make_trajectory_reader(num_sequences=1, steps_per_sequence=10)
        anchors_override = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        anchors_default = reader.compute_anchors([0, 1, 2])  # uses stride=None
        assert len(anchors_override) > len(anchors_default)

    def test_explicit_stride_stored_as_default(self) -> None:
        """Sampling kwarg to TrajectoryDataset.__new__ should set default_sampling."""
        reader = _make_trajectory_reader(
            num_sequences=2,
            steps_per_sequence=12,
            sampling={"stride": 3},
        )
        assert reader.default_sampling == {"stride": 3}
        # compute without explicit sampling → uses stored default
        anchors = reader.compute_anchors(list(range(7)))
        positions_seq0 = anchors[anchors[:, 0] == 0, 1]
        np.testing.assert_array_equal(positions_seq0, [0, 3])


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
        reader = _make_native_reader(length)
        anchors = reader.compute_anchors(offsets, sampling=sampling)
        assert anchors.ndim == 2
        assert anchors.shape[1] == 2

    def test_output_dtype_is_int64(self) -> None:
        reader = _make_native_reader(10)
        anchors = reader.compute_anchors([0, 1], sampling={"stride": 1})
        assert anchors.dtype == np.int64

    def test_positions_are_non_negative(self) -> None:
        reader = _make_native_reader(15)
        anchors = reader.compute_anchors([0, 1, 2], sampling={"stride": 1})
        assert np.all(anchors[:, 1] >= 0)
