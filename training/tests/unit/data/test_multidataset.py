# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.training.data.multidataset import MultiDataset



class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @pytest.fixture
    def dataset_config(self) -> dict:
        """Fixture to provide dataset configuration."""
        return {
            "timestep": "6h",
            "relative_date_indices": [0, 1, 3],  # e.g. f([t, t-6h]) = t+12h
            "relative_date_indices": [0, 1, 3],  # e.g. f([t, t-6h]) = t+12h
            "shuffle": True,
        }

    @pytest.fixture
    def multi_dataset(self, mocker: MockFixture, dataset_config: dict) -> MultiDataset:
        """Fixture to provide a MultiDataset instance with mocked datasets."""
        data_readers = {"dataset_a": None, "dataset_b": None}
        grid_indices = {"dataset_a": None, "dataset_b": None}

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

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_dataset_a, mock_dataset_b],
        )

        return MultiDataset(data_readers=data_readers, grid_indices=grid_indices, **dataset_config)

    def test_timeincrement(self, multi_dataset: MultiDataset) -> None:
        """Test that timeincrement is correctly computed from timestep."""
        expected_timeincrement = 2  # 6H (timestep) in 3h steps (frequency)
        assert multi_dataset.timeincrement == expected_timeincrement

    def test_valid_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that valid_date_indices returns the intersection of indices from all datasets."""
        # relative_date_indices: [0, 1, 3] (for 6H timestep)
        # data (3h) -> data_relative_time_indices: [0, 2, 6]
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
        # Clear the cached property if it was already computed
        if "valid_date_indices" in multi_dataset.__dict__:
            del multi_dataset.__dict__["valid_date_indices"]

        # Mock get_usable_indices: dataset_a has valid indices, dataset_b has none
        mocker.patch(
            "anemoi.training.data.multidataset.get_usable_indices",
            side_effect=[
                np.array([0, 1, 2, 3, 4, 5]),  # dataset_a
                np.array([]),  # dataset_b - empty!
            ],
        )

        # Accessing valid_date_indices should raise ValueError
        with pytest.raises(ValueError, match="No valid date indices found for dataset 'dataset_b'"):
            _ = multi_dataset.valid_date_indices

    def test_valid_date_indices_empty_intersection(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that MultiDataset raises ValueError when intersection of valid indices is empty."""
        # Clear the cached property if it was already computed
        if "valid_date_indices" in multi_dataset.__dict__:
            del multi_dataset.__dict__["valid_date_indices"]

        # Mock get_usable_indices: both datasets have valid indices but no overlap
        # dataset_a has indices: [0, 1, 2]
        # dataset_b has indices: [5, 6, 7]
        # intersection should be empty ([])
        mocker.patch(
            "anemoi.training.data.multidataset.get_usable_indices",
            side_effect=[
                np.array([0, 1, 2]),  # dataset_a
                np.array([5, 6, 7]),  # dataset_b
            ],
        )

        # Accessing valid_date_indices should raise ValueError
        with pytest.raises(ValueError, match="No valid date indices found after intersection across all datasets"):
            _ = multi_dataset.valid_date_indices
