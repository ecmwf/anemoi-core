# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pytest
import torch

from anemoi.training.data.dataset import NativeGridDataset
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline


class TestNativeGridDataset:
    """Test NativeGridDataset instantiation and properties."""

    @pytest.fixture
    def dataset_path(self, extract_dataset_path: tuple[str, str], get_test_archive: GetTestArchive) -> str:
        """Fixture to provide dataset path."""
        path, url_archive = extract_dataset_path
        get_test_archive(url_archive)
        return path

    @skip_if_offline
    @pytest.mark.parametrize("start", [None, 2017])
    @pytest.mark.parametrize("end", [None, 2017])
    def test_basic_instantiation(
        self,
        dataset_path: str,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> None:
        """Test basic instantiation of NativeGridDataset."""
        dataset = NativeGridDataset(dataset=dataset_path, start=start, end=end)

        assert dataset.data is not None
        assert not dataset.has_trajectories
        assert dataset.dates is not None
        assert dataset.variables is not None
        assert dataset.frequency is not None

    @skip_if_offline
    @pytest.mark.parametrize("frequency", [None, "6h", "12h"])
    @pytest.mark.parametrize("drop", [None, []])
    def test_instantiation_with_frequency_and_drop(
        self,
        dataset_path: str,
        frequency: str,
        drop: list[str],
    ) -> None:
        dataset = NativeGridDataset(dataset=dataset_path, frequency=frequency, drop=drop)

        assert dataset.data is not None
        assert not dataset.has_trajectories
        assert dataset.dates is not None
        assert dataset.variables is not None
        assert dataset.frequency is not None

    @skip_if_offline
    def test_instantiation_with_time_range(self, dataset_path: str) -> None:
        """Test NativeGridDataset with start and end dates."""
        original = NativeGridDataset(dataset=dataset_path)
        dates = original.dates

        if len(dates) < 10:
            pytest.skip("Dataset needs at least 10 dates for time range test")

        start = dates[2]
        end = dates[-3]

        dataset = NativeGridDataset(dataset=dataset_path, start=start, end=end)

        assert dataset.data is not None
        assert dataset.dates[0] >= start
        assert dataset.dates[-1] <= end

    @skip_if_offline
    def test_instantiation_with_drop(self, dataset_path: str) -> None:
        """Test NativeGridDataset with dropped variables."""
        # Get original variables to know what to drop
        original = NativeGridDataset(dataset=dataset_path)
        original_vars = original.variables.copy()

        if len(original_vars) < 2:
            pytest.skip("Dataset needs at least 2 variables for drop test")

        drop_vars = [original_vars[0]]
        dataset = NativeGridDataset(dataset=dataset_path, drop=drop_vars)

        assert dataset.data is not None
        assert drop_vars[0] not in dataset.variables
        assert len(dataset.variables) == len(original_vars) - 1

    @skip_if_offline
    def test_dataset_properties(self, dataset_path: str) -> None:
        """Test that dataset properties are correctly accessible."""
        dataset = NativeGridDataset(dataset=dataset_path)

        assert isinstance(dataset.dates, np.ndarray)
        assert len(dataset.dates) > 0
        assert isinstance(dataset.variables, list)
        assert len(dataset.variables) > 0
        assert isinstance(dataset.missing, set)
        assert isinstance(dataset.frequency, datetime.timedelta)
        assert isinstance(dataset.resolution, str)
        assert isinstance(dataset.name_to_index, dict)
        assert isinstance(dataset.statistics, dict)

    @skip_if_offline
    def test_get_sample_with_slice(self, dataset_path: str) -> None:
        """Test get_sample with grid shard as slice."""
        dataset = NativeGridDataset(dataset=dataset_path)

        # Get a sample
        sample = dataset.get_sample(time_indices=slice(0, 3), grid_shard_indices=slice(0, 50))

        assert isinstance(sample, torch.Tensor)
        assert sample.ndim == 4  # dates, ensemble, gridpoints, variables
        assert sample.shape[0] == 3  # 3 time steps
        assert sample.shape[2] == 50  # 50 gridpoints

    @skip_if_offline
    def test_get_sample_with_array_indices(self, dataset_path: str) -> None:
        """Test get_sample with grid shard as array indices."""
        dataset = NativeGridDataset(dataset=dataset_path)

        grid_indices = np.array([0, 10, 20, 30])
        sample = dataset.get_sample(time_indices=slice(0, 3), grid_shard_indices=grid_indices)

        assert isinstance(sample, torch.Tensor)
        assert sample.ndim == 4
        assert sample.shape[0] == 3  # 3 time steps
        assert sample.shape[2] == 4  # 4 selected gridpoints
