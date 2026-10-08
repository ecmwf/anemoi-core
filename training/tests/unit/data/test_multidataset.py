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

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed


class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @staticmethod
    def _mock_reader(mocker: MockFixture, num_dates: int, missing: set[int], num_sequences: int = 1) -> MockFixture:
        """Mock a reader whose ``compute_anchors`` runs the real anchor logic over ``reader.missing``.

        Each sequence has ``num_dates`` positions, and ``reader.missing`` is read at call time,
        so tests can change it after the mock is built.
        """
        reader = mocker.MagicMock()
        reader.missing = missing
        reader.num_sequences = num_sequences
        reader.missing_sequences = set()
        reader.default_sampling = {"stride": 1}
        reader.sequence_length.return_value = num_dates
        reader.missing_positions.side_effect = lambda sequence=0: set(reader.missing)  # noqa: ARG005
        reader.compute_anchors.side_effect = lambda relative_indices, sampling=None: BaseAnemoiReader.compute_anchors(
            reader,
            relative_indices,
            sampling,
        )
        reader.dates = list(range(num_dates))
        reader.frequency = "3h"
        reader.has_trajectories = False
        reader.sampling = None

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

    def test_valid_date_indices(self, multi_dataset: MultiDataset) -> None:
        """valid_date_indices is a flat range over the (sequence, position) anchors every reader can sample."""
        # relative_date_indices are: [0, 2, 6]
        # dataset_a has no missing dates → valid positions [0..23]
        # dataset_b has missing {7, 8, 9, 10} → positions 1..10 read a missing date → valid [0, 11..23]
        # intersection: [0, 11..23], all in sequence 0
        expected_positions = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        np.testing.assert_array_equal(multi_dataset.valid_date_indices, np.arange(len(expected_positions)))
        np.testing.assert_array_equal(multi_dataset.anchors[:, 0], 0)
        np.testing.assert_array_equal(multi_dataset.anchors[:, 1], expected_positions)

    def test_get_sample_offsets_each_reader(self, multi_dataset: MultiDataset) -> None:
        """get_sample(i) asks every reader for anchor i's sequence and position + its relative date indices."""
        multi_dataset.worker_id = 0
        sample = multi_dataset.get_sample(
            1,
        )  # anchor 1 is (sequence 0, position 1)  # anchor 1 is (sequence 0, position 11)

        for name, reader in multi_dataset.data_readers.items():
            reader.get_sample.assert_called_once_with(0, 0, [11, 13, 17])
            assert sample[name] is reader.get_sample.return_value

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
        assert seed_epoch_0 == derive_seed(1000, SeedContext.DATALOADER, 0)

        multi_dataset.set_epoch(5)
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)
        seed_epoch_5 = multi_dataset.seed
        assert seed_epoch_5 == derive_seed(1000, SeedContext.DATALOADER, 5)

        assert seed_epoch_0 != seed_epoch_5

        multi_dataset.per_worker_init(n_workers=4, worker_id=3)
        assert multi_dataset.seed == seed_epoch_5

    def test_worker_shuffle_repeats_for_same_epoch(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """New workers reproduce the shuffle when the base seed and epoch match."""
        mocker.patch("anemoi.training.data.multidataset.get_base_seed", return_value=1000)
        mocker.patch.object(multi_dataset, "get_sample", side_effect=lambda index: int(index))

        multi_dataset.set_epoch(5)
        multi_dataset.per_worker_init(n_workers=2, worker_id=1)
        uninterrupted_order = list(multi_dataset)

        multi_dataset.per_worker_init(n_workers=2, worker_id=1)
        resumed_order = list(multi_dataset)

        assert resumed_order == uninterrupted_order

    def test_fake_dataloading_reuses_first_batch(
        self,
        multi_dataset: MultiDataset,
        mocker: MockFixture,
    ) -> None:
        """Fake dataloading reads one valid batch and reuses its tensors."""
        multi_dataset.fake_dataloading = True
        get_sample = mocker.patch.object(
            multi_dataset,
            "get_sample",
            side_effect=lambda index: {"dataset_a": torch.tensor([index], dtype=torch.int64)},
        )
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)

        batches = list(multi_dataset)

        assert get_sample.call_count == 1
        assert len(batches) == len(multi_dataset.valid_date_indices)
        assert all(batch is batches[0] for batch in batches)
        assert all(torch.equal(batch["dataset_a"], batches[0]["dataset_a"]) for batch in batches)

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

    def test_mixing_single_and_multi_sequence_readers_raises(self, mocker: MockFixture) -> None:
        """A global-time-axis reader cannot be aligned with a trajectory reader's (sequence, position) anchors."""
        data_readers = {
            "analysis": self._mock_reader(mocker, num_dates=30, missing=set()),
            "forecast": self._mock_reader(mocker, num_dates=10, missing=set(), num_sequences=3),
        }
        relative_date_indices = {"analysis": [0, 1], "forecast": [0, 1]}

        with pytest.raises(ValueError, match=r"Single-sequence: \['analysis'\]. Trajectory: \['forecast'\]"):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)
