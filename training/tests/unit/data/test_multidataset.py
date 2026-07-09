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
from pytest_mock import MockFixture

from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.utils.seeding import derive_seed


class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @pytest.fixture
    def multi_dataset(self, mocker: MockFixture) -> MultiDataset:
        """Fixture to provide a MultiDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = list(range(30))  # 15 reference dates
        mock_dataset_a.frequency = "3h"
        mock_dataset_a.num_sequences = 1
        # relative_date_indices=[0,2,6], window=7, valid positions [0..23] at sequence 0
        anchors_a = np.column_stack([np.zeros(24, dtype=np.int64), np.arange(24, dtype=np.int64)])
        mock_dataset_a.compute_anchors.return_value = anchors_a

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = {7, 8, 9, 10}
        mock_dataset_b.dates = list(range(30))  # 15 reference dates
        mock_dataset_b.frequency = "3h"
        mock_dataset_b.num_sequences = 1
        # missing {7..10}: exclude positions {1..10}, valid positions [0, 11..23] at sequence 0
        pos_b = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64)
        anchors_b = np.column_stack([np.zeros(14, dtype=np.int64), pos_b])
        mock_dataset_b.compute_anchors.return_value = anchors_b

        data_readers = {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that valid_date_indices returns a flat range over the valid (sequence, position) anchors."""
        # relative_date_indices are: [0, 2, 6]
        # dataset_a has no missing → valid positions [0..23] at sequence 0
        # dataset_b has missing {7,8,9,10} → valid positions [0, 11..23] at sequence 0
        # intersection: [0, 11..23] → 14 anchors
        expected_positions = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

        # valid_date_indices is a flat index range over the anchors array
        valid_indices = multi_dataset.valid_date_indices
        assert np.array_equal(valid_indices, np.arange(len(expected_positions)))

        # the anchors themselves encode the expected positions at sequence 0
        assert np.array_equal(multi_dataset.anchors[:, 1], expected_positions)

    def test_set_epoch_updates_contiguous_relative_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that set_epoch can update the loaded rollout to contiguous relative date indices."""
        multi_dataset.set_epoch(
            2,
            rollout=3,
            relative_date_indices={"dataset_a": [0, 1, 2], "dataset_b": [0, 1, 2]},
        )

        assert multi_dataset.epoch == 2
        assert multi_dataset.rollout == 3
        assert multi_dataset.relative_date_indices == {
            "dataset_a": slice(0, 3, 1),
            "dataset_b": slice(0, 3, 1),
        }
        assert len(multi_dataset.valid_date_indices) > 0

    def test_worker_seed_includes_epoch(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that worker RNG seed changes with epoch while staying shared across worker partitions."""
        mocker.patch("anemoi.training.data.multidataset.get_base_seed", return_value=1000)

        multi_dataset.set_epoch(0)
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)
        seed_epoch_0 = multi_dataset.seed
        assert seed_epoch_0 == derive_seed(1000, 0)

        multi_dataset.set_epoch(5)
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)
        seed_epoch_5 = multi_dataset.seed
        assert seed_epoch_5 == derive_seed(1000, 5)

        assert seed_epoch_0 != seed_epoch_5

        multi_dataset.per_worker_init(n_workers=4, worker_id=3)
        assert multi_dataset.seed == seed_epoch_5

    def test_valid_date_indices_empty_dataset(self, multi_dataset: MultiDataset) -> None:
        """Test that MultiDataset raises ValueError when a dataset has no valid anchors."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # Make dataset_b return no valid anchors
        data_readers["dataset_b"].compute_anchors.return_value = np.empty((0, 2), dtype=np.int64)

        # Constructing MultiDataset should raise ValueError
        empty_dataset = data_readers["dataset_b"]
        err_msg = f"No valid anchors found for data reader 'dataset_b': {empty_dataset}"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices_empty_intersection(self, multi_dataset: MultiDataset) -> None:
        """Test that MultiDataset raises ValueError when intersection of valid anchors is empty."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # dataset_a has anchors at positions [0, 1, 2]; dataset_b at [5, 6, 7] — no overlap
        data_readers["dataset_a"].compute_anchors.return_value = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
        data_readers["dataset_b"].compute_anchors.return_value = np.array([[0, 5], [0, 6], [0, 7]], dtype=np.int64)

        with pytest.raises(ValueError, match="No valid anchors found after intersection across all datasets"):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)
