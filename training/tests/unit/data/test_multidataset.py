# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import re
from datetime import timedelta

import numpy as np
import pytest
import torch
from pytest_mock import MockFixture

from anemoi.models.data import BaseSample
from anemoi.models.data import Batch
from anemoi.models.data.sample import create_batched_struct
from anemoi.training.data.multidataset import MixedFrequencyDataset
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.data.usable_indices import get_usable_indices


class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @staticmethod
    def _mock_reader(mocker: MockFixture, num_dates: int, missing: set[int]) -> MockFixture:
        """Mock a single-sequence reader whose ``valid_anchors`` runs the real logic over ``reader.missing``.

        ``reader.missing`` is read at call time, so tests can change it after the mock is built.
        """
        reader = mocker.MagicMock()
        reader.missing = missing
        reader.dates = list(range(num_dates))
        reader.frequency = "3h"
        reader.sampling = None
        # Gridded samples carry no point times, so measuring them from the reference changes nothing.
        reader.measure_from_reference.side_effect = lambda sample, *_args: sample

        def valid_anchors(relative_indices: list[int]) -> ReaderAnchors:
            positions = get_usable_indices(reader.missing, len(reader.dates), relative_indices)
            return ReaderAnchors(positions.astype("datetime64[s]"), np.zeros_like(positions), positions)

        reader.valid_anchors.side_effect = valid_anchors
        return reader

    @pytest.fixture
    def multi_dataset(self, mocker: MockFixture) -> MultiDataset:
        """Fixture to provide a MultiDataset instance with mocked datasets."""
        data_readers = {
            "dataset_a": self._mock_reader(mocker, num_dates=30, missing=set()),
            "dataset_b": self._mock_reader(mocker, num_dates=30, missing={7, 8, 9, 10}),
        }
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_len_counts_valid_anchors(self, multi_dataset: MultiDataset) -> None:
        """The dataset has one index per (sequence, position) anchor every reader can sample."""
        # relative_date_indices are: [0, 2, 6]
        # dataset_a has no missing dates → valid positions [0..23]
        # dataset_b has missing {7, 8, 9, 10} → positions 1..10 read a missing date → valid [0, 11..23]
        # intersection: [0, 11..23], all in sequence 0
        expected_positions = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        assert len(multi_dataset) == len(expected_positions)
        for rows in multi_dataset.anchors.rows.values():
            np.testing.assert_array_equal(rows[:, 0], 0)
            np.testing.assert_array_equal(rows[:, 1], expected_positions)

    def test_get_sample_offsets_each_reader(self, multi_dataset: MultiDataset) -> None:
        """get_sample(i) asks every reader for anchor i's sequence and position + its relative date indices."""
        sample = multi_dataset.get_sample(1)  # anchor 1 is (sequence 0, position 11)

        for name, reader in multi_dataset.data_readers.items():
            reader.get_sample.assert_called_once_with(0, [11, 13, 17])
            reader.measure_from_reference.assert_called_once_with(reader.get_sample.return_value, 11, [11, 13, 17])
            assert sample[name] is reader.get_sample.return_value

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
        """Test that MultiDataset raises ValueError when a dataset has no valid anchor."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # Every date of dataset_b is missing, so no index can be sampled from it
        empty_dataset = data_readers["dataset_b"]
        empty_dataset.missing = set(range(30))

        err_msg = f"No valid anchors found for data reader 'dataset_b': {empty_dataset}"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices_empty_intersection(self, multi_dataset: MultiDataset) -> None:
        """Test that MultiDataset raises ValueError when the readers' valid anchors do not overlap."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0], "dataset_b": [0]}

        # dataset_a can only sample dates 0..9, dataset_b only dates 10..29
        data_readers["dataset_a"].missing = set(range(10, 30))
        data_readers["dataset_b"].missing = set(range(10))

        with pytest.raises(ValueError, match="No valid anchors found after intersection across all datasets"):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)


def _mixed_reader(mocker: MockFixture, frequency_minutes: int, dates: np.ndarray) -> MockFixture:
    reader = mocker.MagicMock()
    reader.frequency = timedelta(minutes=frequency_minutes)
    reader.sampling = {"stride": 1}
    reader.data_type = "gridded"

    def valid_anchors(relative_indices: list[int]) -> ReaderAnchors:
        positions = get_usable_indices(set(), len(dates), relative_indices)
        return ReaderAnchors(dates[positions], np.zeros_like(positions), positions)

    def get_sample(_sequence: int, positions: list[int] | np.ndarray) -> BaseSample:
        values = torch.as_tensor(np.atleast_1d(positions), dtype=torch.float32).reshape(-1, 1, 1, 1)
        return create_batched_struct(
            data_type="gridded",
            data=values,
            variables=["value"],
            layout=("time", "ensemble", "grid", "variables"),
            latitudes=[0.0],
            longitudes=[0.0],
        )

    reader.valid_anchors.side_effect = valid_anchors
    reader.get_sample.side_effect = get_sample
    reader.measure_from_reference.side_effect = lambda sample, *_args: sample
    return reader


def test_mixed_frequency_dataset_returns_collatable_samples_with_causal_coarse_times(
    mocker: MockFixture,
) -> None:
    start = np.datetime64("2020-01-01T00:00")
    fine_dates = start + np.arange(145) * np.timedelta64(5, "m")
    coarse_dates = start + np.arange(3) * np.timedelta64(6, "h")
    fine = _mixed_reader(mocker, 5, fine_dates)
    coarse = _mixed_reader(mocker, 360, coarse_dates)

    dataset = MixedFrequencyDataset(
        {"fine": fine, "coarse": coarse},
        relative_date_indices=[0, 72],
        frequency=timedelta(minutes=5),
    )

    sample = dataset[1]
    assert coarse.get_sample.call_args.args[0] == 0
    assert fine.get_sample.call_args.args[0] == 0
    np.testing.assert_array_equal(coarse.get_sample.call_args.args[1], [0, 1])
    np.testing.assert_array_equal(fine.get_sample.call_args.args[1], [1, 73])
    batch = Batch.collate([dataset[0], sample])
    assert batch["fine"].time_size == 2
    assert batch["coarse"].time_size == 2


def test_mixed_frequency_dataset_filters_anchors_without_a_prior_coarse_timestamp(mocker: MockFixture) -> None:
    start = np.datetime64("2020-01-01T00:00")
    fine_dates = start + np.arange(217) * np.timedelta64(5, "m")
    coarse_dates = start + np.arange(1, 4) * np.timedelta64(6, "h")
    dataset = MixedFrequencyDataset(
        {
            "fine": _mixed_reader(mocker, 5, fine_dates),
            "coarse": _mixed_reader(mocker, 360, coarse_dates),
        },
        relative_date_indices=[0, 72],
        frequency=timedelta(minutes=5),
    )

    assert dataset.anchors.times.min() == np.datetime64("2020-01-01T06:00")
