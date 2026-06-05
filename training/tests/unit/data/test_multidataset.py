# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import re

import numpy as np
import pytest

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.multidataset import MultiDataset


class _FakeReader:
    """Minimal analysis-style reader exercising the real anchor model.

    A single sequence whose length and missing positions are configurable.
    Reuses :meth:`BaseAnemoiReader.compute_anchors` so the anchor logic under
    test is the production code path.
    """

    default_sampling = "all"
    num_sequences = 1
    missing_sequences: set[int] = set()
    compute_anchors = BaseAnemoiReader.compute_anchors

    def __init__(self, length: int, missing: set[int]) -> None:
        self._length = length
        self._missing = set(missing)

    def sequence_length(self, sequence: int = 0) -> int:
        return self._length

    def missing_positions(self, sequence: int = 0) -> set[int]:
        return self._missing


class TestMultiDataset:
    """Test MultiDataset anchor computation and properties."""

    @pytest.fixture
    def multi_dataset(self) -> MultiDataset:
        """A MultiDataset over two analysis-style readers."""
        data_readers = {
            "dataset_a": _FakeReader(length=30, missing=set()),
            "dataset_b": _FakeReader(length=30, missing={7, 8, 9, 10}),
        }
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}
        return MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_anchors(self, multi_dataset: MultiDataset) -> None:
        """Anchors are the intersection of valid positions across readers (seq=0)."""
        # dataset_a: positions [0 .. 23] (23 = 29 - max offset 6)
        # dataset_b: same bounds minus those blocked by missing {7, 8, 9, 10}
        # -> shared positions [0, 11, 12, ..., 23]
        expected_positions = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

        assert np.array_equal(multi_dataset.anchors[:, 0], np.zeros_like(expected_positions))
        assert np.array_equal(multi_dataset.anchors[:, 1], expected_positions)
        # The flat index used by the shuffle/shard logic enumerates the anchors.
        assert np.array_equal(multi_dataset.valid_date_indices, np.arange(len(expected_positions)))

    def test_no_valid_anchors_for_reader(self, multi_dataset: MultiDataset, mocker) -> None:
        """ValueError is raised when a reader has no valid anchors."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        mocker.patch.object(data_readers["dataset_a"], "compute_anchors", return_value=np.array([[0, 1]]))
        mocker.patch.object(data_readers["dataset_b"], "compute_anchors", return_value=np.empty((0, 2), dtype=np.int64))

        err_msg = f"No valid anchors found for data reader 'dataset_b': {data_readers['dataset_b']}"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_empty_intersection(self, multi_dataset: MultiDataset, mocker) -> None:
        """ValueError is raised when the anchor intersection is empty."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        mocker.patch.object(
            data_readers["dataset_a"], "compute_anchors", return_value=np.array([[0, 0], [0, 1], [0, 2]]),
        )
        mocker.patch.object(
            data_readers["dataset_b"], "compute_anchors", return_value=np.array([[0, 5], [0, 6], [0, 7]]),
        )

        with pytest.raises(ValueError, match="No valid anchors found after intersection across all datasets"):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)
