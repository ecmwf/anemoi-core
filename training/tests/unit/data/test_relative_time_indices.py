# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.relative_time_indices import normalize_time_indices
from anemoi.training.data.relative_time_indices import offset_time_indices


def test_normalize_time_indices_collapses_contiguous_ranges() -> None:
    """Test that normalize_time_indices collapses contiguous integer sequences into slices."""
    normalized = normalize_time_indices([2, 3, 4])

    assert isinstance(normalized, slice)
    assert (normalized.start, normalized.stop, normalized.step) == (2, 5, 1)


def test_normalize_time_indices_preserves_sparse_ranges() -> None:
    """Test that normalize_time_indices preserves non-contiguous integer sequences as lists."""
    normalized = normalize_time_indices([2, 4, 7])

    assert normalized == [2, 4, 7]


def test_normalize_time_indices_preserves_evenly_spaced_ranges() -> None:
    """Test that normalize_time_indices collapses evenly spaced integer sequences into slices."""
    normalized = normalize_time_indices([2, 4, 6, 8])

    assert isinstance(normalized, slice)
    assert (normalized.start, normalized.stop, normalized.step) == (2, 10, 2)


def test_offset_time_indices_shifts_indices() -> None:
    """Test that offset_time_indices correctly shifts relative time indices by a reference index."""
    offset1 = offset_time_indices(10, [0, 2, 4])
    offset2 = offset_time_indices(10, [-2, 0, 2, 4])
    offset3 = offset_time_indices(10, slice(-2, 5, 2))

    assert offset1 == [10, 12, 14]
    assert offset2 == [8, 10, 12, 14]
    assert isinstance(offset3, slice)
    assert (offset3.start, offset3.stop, offset3.step) == (8, 15, 2)


def test_get_sample_normalizes_time_indices_before_dataset_access() -> None:
    """Test that NativeGridDataset.get_sample normalizes time indices to slices before accessing the dataset."""

    class FakeDataset:
        def __init__(self) -> None:
            self.last_index = None

        def __getitem__(self, item: int) -> np.ndarray:
            self.last_index = item
            return np.zeros((3, 2, 4, 5), dtype=np.float32)

    dataset = NativeGridDataset.__new__(NativeGridDataset)
    dataset.data = FakeDataset()

    dataset.get_sample(time_indices=[4, 5, 6], grid_shard_indices=slice(0, 5))

    time_index = dataset.data.last_index[0]
    assert isinstance(time_index, list)

    dataset.get_sample(time_indices=slice(4, 7, 1), grid_shard_indices=slice(0, 5))

    time_index = dataset.data.last_index[0]
    assert isinstance(time_index, slice)
    assert (time_index.start, time_index.stop, time_index.step) == (4, 7, 1)
