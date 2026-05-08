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
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = list(range(30))  # 15 reference dates
        mock_dataset_a.has_trajectories = False
        mock_dataset_a.frequency = "3h"

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = {7, 8, 9, 10}
        mock_dataset_b.dates = list(range(30))  # 15 reference dates
        mock_dataset_b.has_trajectories = False
        mock_dataset_b.frequency = "3h"

        data_readers = {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDomainDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_sharding(self, multi_domain: MultiDomainDataset) -> None:
        """Test that sharding logic correctly partitions the dataset."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        expected_indices = {
            "dataset_a": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            "dataset_b": np.array([0, 1, 2, 3, 4, 5, 6]),
        }
        for key in expected_indices:
            assert np.array_equal(multi_domain.chunk_index_range[key], expected_indices[key])

    def test_valid_date_indices(self, multi_domain: MultiDomainDataset) -> None:
        """Test that valid_date_indices returns a dictionary of indices from all datasets."""
        # relative_date_indices are: [0, 1, 2]
        # dataset_a has dates [0, 1, 2, ..., 29]
        # dataset_a has indices [0, 1, 2, 3, 4, ..., 22, 23], where 23 = 29 - max(data_relative_time_indices) = 29 - 6
        # dataset_b has missing indices {7, 8, 9, 10}
        # dataset_b has missing indices {7, 8, 9, 10}
        # dataset_b has indices [0, 11, ..., 22, 23], where 23 = 29 - max(data_relative_time_indices) = 29 - 6

        # Test valid_date_indices property
        valid_indices = multi_domain.valid_date_indices

        # Should return a dictionary with concatenation [0, 11, 12, 13, ..., 22, 23]
        expected_indices = {
            "dataset_a": np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            ),
            "dataset_b": np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
        }
        for key in expected_indices:
            assert np.array_equal(valid_indices[key], expected_indices[key])
