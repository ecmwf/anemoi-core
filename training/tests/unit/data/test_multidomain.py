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

from anemoi.training.data.multidomain import MultiDomainDataset
from anemoi.training.data.multidomain import MultiDomainSampler


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

        data_readers = {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 6, 18]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDomainDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

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

        for domain in multi_domain.dataset_names:
            assert set(worker_0_ranges[domain]).isdisjoint(set(worker_1_ranges[domain]))

    def test_get_sample_dispatches_to_requested_domain(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.get_sample("dataset_a", 0)

        multi_domain.data_readers["dataset_a"].get_sample.assert_called_once()
        multi_domain.data_readers["dataset_b"].get_sample.assert_not_called()

        multi_domain.get_sample("dataset_b", 2)
        multi_domain.data_readers["dataset_b"].get_sample.assert_called_once()

    def test_mixing_native_grid_and_trajectory_datasets_raises(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].num_sequences = 2

        with pytest.raises(ValueError, match="same MultiDomainDataset is unsupported"):
            MultiDomainDataset(
                data_readers=multi_domain.data_readers,
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

    def test_check_datasets_units_runs_during_initialization(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].metadata["variables_metadata"]["10u"]["units"] = "km/h"

        with pytest.raises(ValueError, match="Variable compatibility check failed"):
            MultiDomainDataset(
                data_readers=multi_domain.data_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_check_datasets_units_accepts_compatibility_options(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].metadata["variables_metadata"]["10u"]["units"] = "km/h"

        MultiDomainDataset(
            data_readers=multi_domain.data_readers,
            relative_date_indices=multi_domain.relative_date_indices,
            check_variables_compatibility={"ignore_units": True},
        )

    def test_check_datasets_units_raises_error_for_incompatible_units(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.metadata = {
            "dataset_a": {"variables_metadata": {"10u": {"units": "m/s"}}},
            "dataset_b": {"variables_metadata": {"10u": {"units": "km/h"}}},
        }
        with pytest.raises(
            ValueError,
            match="Variable compatibility check failed for domain1 'dataset_a' and domain2 'dataset_b'",
        ):
            multi_domain._check_datasets_units()

    def test_check_datasets_units_passes_for_compatible_units(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.metadata = {
            "dataset_a": {"variables_metadata": {"10u": {"units": "m/s"}}},
            "dataset_b": {"variables_metadata": {"10u": {"units": "m/s"}}},
        }

        assert multi_domain._check_datasets_units() is None

    def test_check_datasets_units_skips_when_no_dataset_has_metadata(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.metadata = {
            "dataset_a": {"variables_metadata": {}},
            "dataset_b": {"variables_metadata": {}},
        }

        assert multi_domain._check_datasets_units() is None

    def test_check_datasets_units_skips_when_only_one_dataset_has_metadata(
        self,
        multi_domain: MultiDomainDataset,
    ) -> None:
        multi_domain.metadata = {
            "dataset_a": {"variables_metadata": {"10u": {"units": "m/s"}}},
            "dataset_b": {"variables_metadata": {}},
        }
        assert (
            multi_domain._check_datasets_units() is None
        ), "Should skip units check when only one dataset has variable metadata"
