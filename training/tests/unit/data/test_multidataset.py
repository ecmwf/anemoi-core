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


class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @pytest.fixture
    def multi_dataset(self, mocker: MockFixture) -> MultiDataset:
        """Fixture to provide a MultiDataset instance with mocked datasets."""
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

        return MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that valid_date_indices returns the intersection of indices from all datasets."""
        # relative_date_indices are: [0, 2, 6]
        # dataset_a|b has dates [0, 1, 2, ..., 29]
        # dataset_a has indices [0, 1, 2, 3, 4, ..., 22, 23], where 23 = 29 - max(data_relative_time_indices)
        # dataset_b has missing indices {7, 8, 9, 10}
        # dataset_b has missing indices {7, 8, 9, 10}
        # dataset_b has indices [0, 11, 12, 13, ..., 22, 23]
        # intersection should be [0, 11, 12, 13, ..., 22, 23]

        # Test valid_date_indices property
        valid_indices = multi_dataset.valid_date_indices

        # Should return intersection [0, 11, 12, 13, ..., 22, 23]
        expected_indices = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        assert np.array_equal(valid_indices, expected_indices)

    def test_valid_date_indices_empty_dataset(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that MultiDataset raises ValueError when a dataset has no valid indices."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # Mock get_usable_indices: dataset_a has valid indices, dataset_b has none.
        # Patch before constructing MultiDataset so it takes effect during __init__.
        mocker.patch(
            "anemoi.training.data.usable_indices.get_usable_indices",
            side_effect=[
                np.array([0, 1, 2, 3, 4, 5]),  # dataset_a
                np.array([]),  # dataset_b - empty!
            ],
        )

        # Constructing MultiDataset should raise ValueError
        empty_dataset = data_readers["dataset_b"]
        err_msg = f"No valid date indices found for data reader 'dataset_b': {empty_dataset}"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices_empty_intersection(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that MultiDataset raises ValueError when intersection of valid indices is empty."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # Mock get_usable_indices: both datasets have valid indices but no overlap
        # dataset_a has indices: [0, 1, 2]
        # dataset_b has indices: [5, 6, 7]
        # intersection should be empty ([]).
        # Patch before constructing MultiDataset so it takes effect during __init__.
        mocker.patch(
            "anemoi.training.data.usable_indices.get_usable_indices",
            side_effect=[
                np.array([0, 1, 2]),  # dataset_a
                np.array([5, 6, 7]),  # dataset_b
            ],
        )

        # Constructing MultiDataset should raise ValueError
        with pytest.raises(ValueError, match="No valid date indices found after intersection across all datasets"):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)
