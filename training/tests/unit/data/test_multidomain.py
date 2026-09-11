# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
from pytest_mock import MockFixture

from anemoi.training.data.batch_meta import META_KEY
from anemoi.training.data.batch_meta import meta_participant
from anemoi.training.data.batch_meta import split_meta
from anemoi.training.data.datasets import MultiDomainDataset
from anemoi.training.data.datasets.multidomain import MultiDomainSampler


class TestMultiDomain:
    """Test MultiDomainDataset instantiation and properties."""

    @pytest.fixture
    def multi_domain(self, mocker: MockFixture) -> MultiDomainDataset:
        """Fixture to provide a MultiDomainDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = {7, 8, 9, 10}
        mock_dataset_a.dates = list(range(30))
        mock_dataset_a.frequency = "3h"
        mock_dataset_a.grid_size = 5
        mock_dataset_a.num_sequences = 1
        mock_dataset_a.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_a.compute_anchors.return_value = np.array(
            [[0, 0], *[[0, index] for index in range(11, 24)]],
        )

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = set()
        mock_dataset_b.dates = list(range(20, 60))
        mock_dataset_b.frequency = "1h"
        mock_dataset_b.grid_size = 8
        mock_dataset_b.num_sequences = 1
        mock_dataset_b.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_b.compute_anchors.return_value = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])

        # One dataset ("data") with two participants that share variables and frequency but
        # have independent anchors. Relative date indices are keyed by dataset name.
        data_readers = {"data": {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}}
        relative_date_indices = {"data": [0, 2, 6]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDomainDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_participants_of_single_dataset(self, multi_domain: MultiDomainDataset) -> None:
        readers = multi_domain.participant_readers["data"]
        assert multi_domain.dataset_name == "data"
        assert multi_domain.dataset_names == ["data"]
        assert multi_domain.participants == ["dataset_a", "dataset_b"]
        assert multi_domain.reference_readers == {"data": readers["dataset_a"]}
        assert multi_domain.participant_row("dataset_b") == {"data": readers["dataset_b"]}
        assert set(multi_domain.relative_date_indices) == {"data"}
        # dataset-level properties are keyed by dataset name (reference participant) ...
        assert multi_domain.metadata == {"data": readers["dataset_a"].metadata}
        assert set(multi_domain.shard_shapes) == {"data"}
        # ... per-participant values stay available
        assert multi_domain._collect_participants("grid_size") == {"data": {"dataset_a": 5, "dataset_b": 8}}

    def test_rejects_several_datasets(self, multi_domain: MultiDomainDataset) -> None:
        readers = multi_domain.participant_readers["data"]
        with pytest.raises(ValueError, match="exactly one dataset"):
            MultiDomainDataset(
                data_readers={"data": readers["dataset_a"], "other": readers["dataset_b"]},
                relative_date_indices={"data": [0, 2, 6], "other": [0, 2, 6]},
            )

    def test_participant_without_anchors_raises(self, multi_domain: MultiDomainDataset) -> None:
        readers = multi_domain.participant_readers["data"]
        readers["dataset_b"].compute_anchors.return_value = np.empty((0, 2), dtype=np.int64)
        with pytest.raises(ValueError, match="Participant 'dataset_b': No valid anchors"):
            MultiDomainDataset(
                data_readers=multi_domain.participant_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_sharding(self, multi_domain: MultiDomainDataset) -> None:
        """Test that sharding logic correctly partitions the dataset."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        expected_indices = {
            "dataset_a": np.array([0, 1, 2, 3, 4, 5, 6]),
            "dataset_b": np.array([0, 1]),
        }
        for key in expected_indices:
            assert np.array_equal(multi_domain.chunk_index_range[key], expected_indices[key])

    def test_valid_date_indices(self, multi_domain: MultiDomainDataset) -> None:
        """Test that valid_date_indices returns a dictionary of indices from all datasets.

        relative_date_indices = [0, 1, 2]

        Each reader supplies valid ``(sequence, position)`` anchors. The
        dataset keeps a one-dimensional index for shuffling and sharding.
        """
        expected_indices = {"dataset_a": np.arange(14), "dataset_b": np.arange(4)}
        for key in expected_indices:
            assert np.array_equal(multi_domain.valid_date_indices[key], expected_indices[key])

        assert np.array_equal(multi_domain.anchors["dataset_a"][:, 1], [0, *range(11, 24)])
        assert np.array_equal(multi_domain.anchors["dataset_b"], [[0, 0], [0, 1], [0, 2], [0, 3]])

    def test_per_worker_init_creates_domain_specific_worker_state(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        assert set(multi_domain.n_samples_per_worker) == {"dataset_a", "dataset_b"}
        assert set(multi_domain.chunk_index_range) == {"dataset_a", "dataset_b"}

        assert isinstance(multi_domain.chunk_index_range["dataset_a"], np.ndarray)
        assert isinstance(multi_domain.chunk_index_range["dataset_b"], np.ndarray)

    def test_worker_shards_do_not_overlap_per_domain(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        worker_0_ranges = {k: v.copy() for k, v in multi_domain.chunk_index_range.items()}

        multi_domain.per_worker_init(n_workers=2, worker_id=1)
        worker_1_ranges = {k: v.copy() for k, v in multi_domain.chunk_index_range.items()}

        for domain in multi_domain.participants:
            assert set(worker_0_ranges[domain]).isdisjoint(set(worker_1_ranges[domain]))

    def test_get_sample_dispatches_to_requested_domain(self, multi_domain: MultiDomainDataset) -> None:
        readers = multi_domain.participant_readers["data"]
        sample = multi_domain.get_sample("dataset_a", 0)

        # keyed by DATASET name, participant carried as metadata
        assert sample == {"data": readers["dataset_a"].get_sample.return_value, META_KEY: {"participant": "dataset_a"}}
        readers["dataset_a"].get_sample.assert_called_once()
        readers["dataset_b"].get_sample.assert_not_called()

        sample = multi_domain.get_sample("dataset_b", 2)
        assert meta_participant(split_meta(sample)[1]) == "dataset_b"
        readers["dataset_b"].get_sample.assert_called_once()

    def test_mixing_native_grid_and_trajectory_datasets_raises(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.participant_readers["data"]["dataset_b"].num_sequences = 2

        with pytest.raises(ValueError, match="same MultiDomainDataset is unsupported"):
            MultiDomainDataset(
                data_readers=multi_domain.participant_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_sampler_preserves_domain_order_across_sample_groups(self) -> None:
        valid_date_indices = {"dataset_a": np.arange(8), "dataset_b": np.arange(4)}
        group_0_ranges = {"dataset_a": np.arange(0, 4), "dataset_b": np.arange(0, 2)}
        group_1_ranges = {"dataset_a": np.arange(4, 8), "dataset_b": np.arange(2, 4)}

        group_0 = list(MultiDomainSampler(valid_date_indices, group_0_ranges, np.random.default_rng(42)))
        group_1 = list(MultiDomainSampler(valid_date_indices, group_1_ranges, np.random.default_rng(42)))

        assert [domain for domain, _ in group_0] == [domain for domain, _ in group_1]
        for domain in valid_date_indices:
            group_0_indices = {index for sampled_domain, index in group_0 if sampled_domain == domain}
            group_1_indices = {index for sampled_domain, index in group_1 if sampled_domain == domain}
            assert group_0_indices.isdisjoint(group_1_indices)

    def test_sampler_without_shuffle_preserves_domain_and_index_order(self) -> None:
        sampler = MultiDomainSampler(
            {"dataset_a": np.arange(4), "dataset_b": np.arange(3)},
            {"dataset_a": np.arange(1, 3), "dataset_b": np.arange(0, 2)},
            np.random.default_rng(42),
            shuffle=False,
        )

        assert len(sampler) == 4
        assert list(sampler) == [("dataset_a", 1), ("dataset_a", 2), ("dataset_b", 0), ("dataset_b", 1)]

    def test_sampler_repeats_for_same_seed(self) -> None:
        valid_date_indices = {"dataset_a": np.arange(8), "dataset_b": np.arange(4)}
        chunk_index_range = {"dataset_a": np.arange(0, 4), "dataset_b": np.arange(0, 2)}

        first = MultiDomainSampler(valid_date_indices, chunk_index_range, np.random.default_rng(42))
        second = MultiDomainSampler(valid_date_indices, chunk_index_range, np.random.default_rng(42))

        assert list(first) == list(second)

    @staticmethod
    def _batches(samples: list[tuple[str, int]], batch_size: int) -> list[list[tuple[str, int]]]:
        """Group consecutive samples the way the DataLoader collates one worker's iterator."""
        return [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_sampler_batches_are_domain_pure(self, batch_size: int) -> None:
        valid_date_indices = {"dataset_a": np.arange(23), "dataset_b": np.arange(9), "dataset_c": np.arange(4)}
        chunk_index_range = {domain: np.arange(len(indices)) for domain, indices in valid_date_indices.items()}

        sampler = MultiDomainSampler(
            valid_date_indices,
            chunk_index_range,
            np.random.default_rng(7),
            batch_size=batch_size,
        )
        samples = list(sampler)

        assert len(samples) == len(sampler)
        assert len(samples) % batch_size == 0
        for batch in self._batches(samples, batch_size):
            assert len({domain for domain, _ in batch}) == 1
        # every kept sample is unique
        assert len(set(samples)) == len(samples)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_sampler_drops_only_trailing_remainder_per_domain(self, batch_size: int) -> None:
        valid_date_indices = {"dataset_a": np.arange(23), "dataset_b": np.arange(9)}
        chunk_index_range = {domain: np.arange(len(indices)) for domain, indices in valid_date_indices.items()}

        sampler = MultiDomainSampler(
            valid_date_indices,
            chunk_index_range,
            np.random.default_rng(7),
            batch_size=batch_size,
        )
        samples = list(sampler)

        for domain, indices in valid_date_indices.items():
            n_kept = sum(1 for sampled_domain, _ in samples if sampled_domain == domain)
            assert n_kept == (len(indices) // batch_size) * batch_size
            assert sampler.num_blocks(domain) == len(indices) // batch_size
            assert sampler.num_dropped(domain) == len(indices) % batch_size
        assert len(sampler) == sum((len(indices) // batch_size) * batch_size for indices in valid_date_indices.values())

    def test_sampler_blocks_are_proportional_to_domain_size(self) -> None:
        valid_date_indices = {"dataset_a": np.arange(40), "dataset_b": np.arange(20), "dataset_c": np.arange(10)}
        chunk_index_range = {domain: np.arange(len(indices)) for domain, indices in valid_date_indices.items()}

        sampler = MultiDomainSampler(valid_date_indices, chunk_index_range, np.random.default_rng(3), batch_size=2)
        batches = self._batches(list(sampler), 2)
        domain_batches = {domain: sum(1 for batch in batches if batch[0][0] == domain) for domain in valid_date_indices}

        assert domain_batches == {"dataset_a": 20, "dataset_b": 10, "dataset_c": 5}

    def test_sampler_shuffles_block_order_not_only_within_blocks(self) -> None:
        valid_date_indices = {"dataset_a": np.arange(16), "dataset_b": np.arange(16)}
        chunk_index_range = {domain: np.arange(len(indices)) for domain, indices in valid_date_indices.items()}

        sampler = MultiDomainSampler(valid_date_indices, chunk_index_range, np.random.default_rng(11), batch_size=4)
        domain_sequence = [batch[0][0] for batch in self._batches(list(sampler), 4)]

        # without block shuffling all dataset_a batches would precede all dataset_b batches
        assert domain_sequence != sorted(domain_sequence)

    def test_sampler_without_shuffle_yields_full_blocks_in_order(self) -> None:
        sampler = MultiDomainSampler(
            {"dataset_a": np.arange(5), "dataset_b": np.arange(3)},
            {"dataset_a": np.arange(5), "dataset_b": np.arange(3)},
            np.random.default_rng(42),
            shuffle=False,
            batch_size=2,
        )

        assert len(sampler) == 6
        assert list(sampler) == [
            ("dataset_a", 0),
            ("dataset_a", 1),
            ("dataset_a", 2),
            ("dataset_a", 3),
            ("dataset_b", 0),
            ("dataset_b", 1),
        ]

    def test_sampler_same_seed_gives_same_domain_sequence_across_sample_groups(self) -> None:
        """All DDP ranks (sample comm groups) must see the same domain per step for batch_size > 1."""
        valid_date_indices = {"dataset_a": np.arange(16), "dataset_b": np.arange(8)}
        group_0_ranges = {"dataset_a": np.arange(0, 8), "dataset_b": np.arange(0, 4)}
        group_1_ranges = {"dataset_a": np.arange(8, 16), "dataset_b": np.arange(4, 8)}

        group_0 = list(MultiDomainSampler(valid_date_indices, group_0_ranges, np.random.default_rng(5), batch_size=4))
        group_1 = list(MultiDomainSampler(valid_date_indices, group_1_ranges, np.random.default_rng(5), batch_size=4))

        assert [domain for domain, _ in group_0] == [domain for domain, _ in group_1]
        for domain in valid_date_indices:
            group_0_indices = {index for sampled_domain, index in group_0 if sampled_domain == domain}
            group_1_indices = {index for sampled_domain, index in group_1 if sampled_domain == domain}
            assert group_0_indices.isdisjoint(group_1_indices)

    def test_sampler_reshuffles_for_different_seed(self) -> None:
        valid_date_indices = {"dataset_a": np.arange(16), "dataset_b": np.arange(8)}
        chunk_index_range = {domain: np.arange(len(indices)) for domain, indices in valid_date_indices.items()}

        first = list(MultiDomainSampler(valid_date_indices, chunk_index_range, np.random.default_rng(1), batch_size=2))
        second = list(MultiDomainSampler(valid_date_indices, chunk_index_range, np.random.default_rng(2), batch_size=2))

        assert first != second
        assert set(first) == set(second)

    def test_sampler_rejects_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            MultiDomainSampler(
                {"dataset_a": np.arange(4)},
                {"dataset_a": np.arange(4)},
                np.random.default_rng(0),
                batch_size=0,
            )

    def test_dataset_iter_yields_domain_pure_batches(
        self,
        multi_domain: MultiDomainDataset,
        mocker: MockFixture,
    ) -> None:
        """End-to-end: dataset __iter__ honours batch_size and epoch changes reshuffle the block order."""
        mocker.patch("anemoi.training.data.datasets.anemoidataset.get_base_seed", return_value=1000)
        dataset = MultiDomainDataset(
            data_readers=multi_domain.participant_readers,
            relative_date_indices=multi_domain.relative_date_indices,
            batch_size=2,
        )
        # make the yielded sample identify (domain, index) so the order can be inspected
        mocker.patch.object(dataset, "get_sample", side_effect=lambda domain, index: (domain, index))

        dataset.set_epoch(0)
        dataset.per_worker_init(n_workers=1, worker_id=0)
        epoch_0 = list(dataset)

        # dataset_a has 14 anchors -> 7 blocks, dataset_b has 4 anchors -> 2 blocks
        assert len(epoch_0) == 18
        assert len(set(epoch_0)) == 18
        for batch in self._batches(epoch_0, 2):
            assert len({domain for domain, _ in batch}) == 1

        dataset.set_epoch(1)
        dataset.per_worker_init(n_workers=1, worker_id=0)
        epoch_1 = list(dataset)

        assert epoch_1 != epoch_0
        assert set(epoch_1) == set(epoch_0)

    def test_check_datasets_units_runs_during_initialization(self, multi_domain: MultiDomainDataset) -> None:
        reader_b = multi_domain.participant_readers["data"]["dataset_b"]
        reader_b.metadata["variables_metadata"]["10u"]["units"] = "km/h"

        with pytest.raises(ValueError, match="Variable compatibility check failed"):
            MultiDomainDataset(
                data_readers=multi_domain.participant_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_check_datasets_units_accepts_compatibility_options(self, multi_domain: MultiDomainDataset) -> None:
        reader_b = multi_domain.participant_readers["data"]["dataset_b"]
        reader_b.metadata["variables_metadata"]["10u"]["units"] = "km/h"

        MultiDomainDataset(
            data_readers=multi_domain.participant_readers,
            relative_date_indices=multi_domain.relative_date_indices,
            check_variables_compatibility={"ignore_units": True},
        )

    @staticmethod
    def _set_participant_metadata(multi_domain: MultiDomainDataset, metadata: dict[str, dict]) -> None:
        for participant, meta in metadata.items():
            multi_domain.participant_readers["data"][participant].metadata = meta

    def test_check_datasets_units_raises_error_for_incompatible_units(self, multi_domain: MultiDomainDataset) -> None:
        self._set_participant_metadata(
            multi_domain,
            {
                "dataset_a": {"variables_metadata": {"10u": {"units": "m/s"}}},
                "dataset_b": {"variables_metadata": {"10u": {"units": "km/h"}}},
            },
        )
        with pytest.raises(
            ValueError,
            match="Variable compatibility check failed for domain1 'dataset_a' and domain2 'dataset_b'",
        ):
            multi_domain._check_datasets_units()

    def test_check_datasets_units_passes_for_compatible_units(self, multi_domain: MultiDomainDataset) -> None:
        self._set_participant_metadata(
            multi_domain,
            {
                "dataset_a": {"variables_metadata": {"10u": {"units": "m/s"}}},
                "dataset_b": {"variables_metadata": {"10u": {"units": "m/s"}}},
            },
        )

        assert multi_domain._check_datasets_units() is None

    def test_check_datasets_units_skips_when_no_dataset_has_metadata(self, multi_domain: MultiDomainDataset) -> None:
        self._set_participant_metadata(
            multi_domain,
            {
                "dataset_a": {"variables_metadata": {}},
                "dataset_b": {"variables_metadata": {}},
            },
        )

        assert multi_domain._check_datasets_units() is None

    def test_check_datasets_units_skips_when_only_one_dataset_has_metadata(
        self,
        multi_domain: MultiDomainDataset,
    ) -> None:
        self._set_participant_metadata(
            multi_domain,
            {
                "dataset_a": {"variables_metadata": {"10u": {"units": "m/s"}}},
                "dataset_b": {"variables_metadata": {}},
            },
        )
        assert (
            multi_domain._check_datasets_units() is None
        ), "Should skip units check when only one dataset has variable metadata"
