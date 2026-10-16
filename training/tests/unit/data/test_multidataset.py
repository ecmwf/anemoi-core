# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import re

import numpy as np
import pytest
import torch
from pytest_mock import MockFixture

from anemoi.training.data.data_reader import RelativeTimeReader
from anemoi.training.data.multidataset import MixedFrequencyDataset
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

    def test_mixed_frequency_rejects_trajectory_datasets(self, mocker: MockFixture) -> None:
        reader = mocker.MagicMock()
        reader.num_sequences = 2

        with pytest.raises(ValueError, match="Trajectory datasets are unsupported"):
            MixedFrequencyDataset(
                data_readers={"trajectory": reader},
                relative_date_indices=[0, 1],
                frequency="1h",
            )


class RecordingReader:
    def __init__(self, *, missing: set[int] | None = None) -> None:
        self.data = "coarse"
        self.frequency = datetime.timedelta(hours=6)
        self.dates = np.arange(
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-02T00:00"),
            np.timedelta64(6, "h"),
        )
        self.missing = missing or set()
        self.raw_requested_indices = None
        self.requested_indices = None

    def get_sample(
        self,
        _sequence: int,
        time_indices: slice | int | list[int],
        _grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        self.raw_requested_indices = time_indices
        if isinstance(time_indices, slice):
            indices = list(range(time_indices.start, time_indices.stop, time_indices.step))
        elif isinstance(time_indices, int):
            indices = [time_indices]
        else:
            indices = list(time_indices)
        self.requested_indices = indices
        return torch.as_tensor(indices, dtype=torch.float32).reshape(len(indices), 1, 1, 1)


def test_relative_time_reader_uses_previous_native_time_without_dropping_time_axis() -> None:
    reader = RecordingReader()
    wrapped = RelativeTimeReader(
        reader,
        native_relative_indices=np.array([0]),
        model_relative_indices=np.array([0]),
        frequency_seconds=3600,
        anchor_dates_ns=np.array([np.datetime64("2020-01-01T04:00", "ns").astype(np.int64)]),
    )

    sample = wrapped.get_sample(sequence=0, position=0)

    assert not isinstance(reader.raw_requested_indices, int)
    assert reader.requested_indices == [0]
    assert sample.shape == (1, 1, 1, 1)


def test_relative_time_reader_skips_missing_exact_native_time() -> None:
    reader = RecordingReader(missing={1})
    wrapped = RelativeTimeReader(
        reader,
        native_relative_indices=np.array([0]),
        model_relative_indices=np.array([0]),
        frequency_seconds=3600,
        anchor_dates_ns=np.array([np.datetime64("2020-01-01T06:00", "ns").astype(np.int64)]),
    )

    sample = wrapped.get_sample(sequence=0, position=0)

    assert reader.requested_indices == [0]
    assert sample.shape == (1, 1, 1, 1)


def test_mixed_frequency_dataset_filters_anchors_before_coarse_data_start(mocker: MockFixture) -> None:
    fast = mocker.MagicMock()
    fast.num_sequences = 1
    fast.frequency = datetime.timedelta(minutes=5)
    fast.dates = np.arange(
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T02:00"),
        np.timedelta64(5, "m"),
    )
    fast.missing = set()
    fast.compute_anchors.return_value = np.column_stack(
        [np.zeros(23, dtype=np.int64), np.arange(23, dtype=np.int64)],
    )

    coarse = mocker.MagicMock()
    coarse.num_sequences = 1
    coarse.frequency = datetime.timedelta(hours=1)
    coarse.dates = np.arange(
        np.datetime64("2020-01-01T01:00"),
        np.datetime64("2020-01-01T03:00"),
        np.timedelta64(1, "h"),
    )
    coarse.missing = set()

    dataset = MixedFrequencyDataset(
        data_readers={"fast": fast, "coarse": coarse},
        relative_date_indices=[0, 1],
        frequency="5m",
    )

    assert dataset.anchors[0].tolist() == [0, 12]
