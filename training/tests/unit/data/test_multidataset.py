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
import torch
from pytest_mock import MockFixture

from anemoi.training.data.multidataset import MultiDataset


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

    def test_len_counts_valid_anchors(self, multi_dataset: MultiDataset) -> None:
        """Test that the dataset has one index per valid (sequence, position) anchor."""
        # relative_date_indices are: [0, 2, 6]
        # dataset_a has no missing → valid positions [0..23] at sequence 0
        # dataset_b has missing {7,8,9,10} → valid positions [0, 11..23] at sequence 0
        # intersection: [0, 11..23] → 14 anchors
        expected_positions = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

        assert len(multi_dataset) == len(expected_positions)
        assert np.array_equal(multi_dataset.anchors[:, 1], expected_positions)

    def test_set_relative_date_indices_updates_contiguous_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that the loaded time steps can be updated to contiguous relative date indices."""
        multi_dataset.set_relative_date_indices({"dataset_a": [0, 1, 2], "dataset_b": [0, 1, 2]})

        assert multi_dataset.relative_date_indices == {
            "dataset_a": slice(0, 3, 1),
            "dataset_b": slice(0, 3, 1),
        }
        assert len(multi_dataset) > 0

    def test_getitem_reads_the_sample_of_the_index(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Indexing reads the synchronized sample of that anchor."""
        get_sample = mocker.patch.object(multi_dataset, "get_sample", side_effect=lambda index: {"index": index})

        assert multi_dataset[3] == {"index": 3}
        assert multi_dataset[7] == {"index": 7}
        assert [call.args[0] for call in get_sample.call_args_list] == [3, 7]

    def test_fake_dataloading_reuses_first_sample(
        self,
        multi_dataset: MultiDataset,
        mocker: MockFixture,
    ) -> None:
        """Fake dataloading reads one valid sample and reuses its tensors."""
        multi_dataset.fake_dataloading = True
        get_sample = mocker.patch.object(
            multi_dataset,
            "get_sample",
            side_effect=lambda index: {"dataset_a": torch.tensor([index], dtype=torch.int64)},
        )

        samples = [multi_dataset[index] for index in range(len(multi_dataset))]

        assert get_sample.call_count == 1
        assert all(sample is samples[0] for sample in samples)

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
