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


class TestMultiDomain:
    """Test MultiDomainDataset instantiation and properties."""

    @pytest.fixture
    def multi_domain(self, mocker: MockFixture) -> MultiDomainDataset:
        """Fixture to provide a MultiDomainDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = {7, 8, 9, 10}
        mock_dataset_a.dates = list(range(30))
        mock_dataset_a.has_trajectories = False
        mock_dataset_a.frequency = "3h"

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = set()
        mock_dataset_b.dates = list(range(20, 60))
        mock_dataset_b.has_trajectories = True
        mock_dataset_b.trajectory_ids = np.array([0] * 20 + [1] * 20)  # split at 40
        mock_dataset_b.frequency = "1h"

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

        dataset_a:
        dates:   [0, 1, 2, ..., 29]
        indices: [0, 1, 2, ..., 22, 23]
                where 23 = 29 - max(data_relative_time_indices) = 29 - 6

        dataset_b:
        missing indices: {26, 27, 28, 29}
        indices: [20, 21, 22, 23, 24, 25, 30, 31, 32]
                where 32 = 49 - max(data_relative_time_indices) = 49 - 18
        """
        # Test valid_date_indices property
        valid_indices = multi_domain.valid_date_indices

        # Should return a dictionary with concatenation [0, 11, 12, 13, ..., 22, 23]
        expected_indices = {
            "dataset_a": np.array(
                [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            ),
            "dataset_b": np.array([0, 1, 20, 21]),
        }
        for key in expected_indices:
            assert np.array_equal(valid_indices[key], expected_indices[key])

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
