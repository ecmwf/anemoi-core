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
            _ = multi_dataset.valid_date_indices

    def test_mixed_frequency_relative_indices(self, mocker: MockFixture) -> None:
        """Mixed frequencies use per-dataset native-relative index maps."""
        data_readers = {"opera": None, "meps": None}

        mock_opera = mocker.MagicMock()
        mock_opera.missing = set()
        mock_opera.dates = list(range(200))
        mock_opera.has_trajectories = False
        mock_opera.frequency = "15m"

        mock_meps = mocker.MagicMock()
        mock_meps.missing = set()
        mock_meps.dates = list(range(200))
        mock_meps.has_trajectories = False
        mock_meps.frequency = "6h"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_opera, mock_meps],
        )

        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 1, 2, 3, 4, 5, 24],
            timestep="15m",
            shuffle=False,
        )

        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["opera"],
            np.array([0, 1, 2, 3, 4, 5, 24], dtype=np.int64),
        )
        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["meps"],
            np.array([0, 24], dtype=np.int64),
        )
        assert np.array_equal(
            dataset.data_relative_date_indices_by_dataset["meps"],
            np.array([0, 1], dtype=np.int64),
        )

    def test_dataset_num_inputs_with_window(self, mocker: MockFixture) -> None:
        """Window-based per-dataset inputs are added and validated."""
        data_readers = {"opera": None, "meps": None}

        mock_opera = mocker.MagicMock()
        mock_opera.missing = set()
        mock_opera.dates = list(range(200))
        mock_opera.has_trajectories = False
        mock_opera.frequency = "15m"

        mock_meps = mocker.MagicMock()
        mock_meps.missing = set()
        mock_meps.dates = list(range(200))
        mock_meps.has_trajectories = False
        mock_meps.frequency = "6h"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_opera, mock_meps],
        )

        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 24],
            timestep="15m",
            multistep_window="6h",
            dataset_num_inputs={"opera": 24, "meps": 1},
            shuffle=False,
        )

        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["opera"],
            np.arange(25, dtype=np.int64),
        )
        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["meps"],
            np.array([0, 24], dtype=np.int64),
        )

    def test_dataset_num_inputs_uniform_divides_window_constraint(self, mocker: MockFixture) -> None:
        """Uniform selection requires that available native points per window is divisible by num_inputs."""
        data_readers = {"opera": None}

        mock_opera = mocker.MagicMock()
        mock_opera.missing = set()
        mock_opera.dates = list(range(200))
        mock_opera.has_trajectories = False
        mock_opera.frequency = "15m"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            return_value=mock_opera,
        )

        with pytest.raises(ValueError, match="must divide `multistep_window // dataset_frequency`"):
            MultiDataset(
                data_readers=data_readers,
                relative_date_indices=[0, 24],
                timestep="15m",
                multistep_window="6h",
                dataset_num_inputs={"opera": 5},
                shuffle=False,
            )

    def test_dataset_num_inputs_selection_mode_last(self, mocker: MockFixture) -> None:
        """Selection mode 'last' picks the latest native points in the window."""
        data_readers = {"opera": None}

        mock_opera = mocker.MagicMock()
        mock_opera.missing = set()
        mock_opera.dates = list(range(200))
        mock_opera.has_trajectories = False
        mock_opera.frequency = "15m"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            return_value=mock_opera,
        )

        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 24],
            timestep="15m",
            multistep_window="6h",
            dataset_num_inputs={"opera": 6},
            dataset_input_selection={"opera": "last"},
            shuffle=False,
        )

        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["opera"],
            np.array([0, 18, 19, 20, 21, 22, 23, 24], dtype=np.int64),
        )

    def test_dataset_num_inputs_selection_mode_uniform_downsample(self, mocker: MockFixture) -> None:
        """Selection mode 'uniform' keeps evenly spaced points and preserves them as leading inputs."""
        data_readers = {"nordic_radar": None}

        mock_radar = mocker.MagicMock()
        mock_radar.missing = set()
        mock_radar.dates = list(range(300))
        mock_radar.has_trajectories = False
        mock_radar.frequency = "5m"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            return_value=mock_radar,
        )

        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 1, 3, 6, 12],
            timestep="5m",
            multistep_window="1h",
            dataset_num_inputs={"nordic_radar": 6},
            dataset_input_selection={"nordic_radar": "uniform"},
            shuffle=False,
        )

        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["nordic_radar"],
            np.array([0, 1, 2, 3, 4, 6, 8, 10, 12], dtype=np.int64),
        )

    def test_dataset_num_inputs_selection_mode_future(self, mocker: MockFixture) -> None:
        """Selection mode 'future' picks earliest native points in the future window."""
        data_readers = {"nordic_radar": None}

        mock_radar = mocker.MagicMock()
        mock_radar.missing = set()
        mock_radar.dates = list(range(300))
        mock_radar.has_trajectories = False
        mock_radar.frequency = "5m"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            return_value=mock_radar,
        )

        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 1, 3, 6, 12],
            timestep="5m",
            multistep_window="1h",
            dataset_num_inputs={"nordic_radar": 6},
            dataset_input_selection={"nordic_radar": "future"},
            shuffle=False,
        )

        assert np.array_equal(
            dataset.model_relative_date_indices_by_dataset["nordic_radar"],
            np.array([0, 1, 3, 6, 12, 13, 14, 15, 16, 17], dtype=np.int64),
        )

    def test_sparse_time_index_mode_uses_high_frequency_anchor(self, mocker: MockFixture) -> None:
        """Sparse mode should use high-frequency dataset indices instead of full intersection."""
        data_readers = {"nordic_radar": None, "opera": None}

        mock_radar = mocker.MagicMock()
        mock_radar.missing = set()
        mock_radar.dates = list(range(200))
        mock_radar.has_trajectories = False
        mock_radar.frequency = "5m"

        mock_opera = mocker.MagicMock()
        mock_opera.missing = set(range(30, 120))
        mock_opera.dates = list(range(200))
        mock_opera.has_trajectories = False
        mock_opera.frequency = "15m"

        # Dense mode uses the intersection.
        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_radar, mock_opera],
        )
        dense_dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 1, 2, 3, 4, 6, 8, 10, 12],
            timestep="5m",
            time_index_mode="dense",
            shuffle=False,
        )

        # Sparse mode anchors to the highest-frequency dataset (nordic_radar).
        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_radar, mock_opera],
        )
        sparse_dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 1, 2, 3, 4, 6, 8, 10, 12],
            timestep="5m",
            time_index_mode="sparse",
            shuffle=False,
        )

        assert sparse_dataset._anchor_dataset_name == "nordic_radar"
        assert len(sparse_dataset.valid_date_indices) > len(dense_dataset.valid_date_indices)

    def test_sparse_aux_indices_use_timestamp_alignment(self, mocker: MockFixture) -> None:
        """Sparse aux datasets should resolve native indices by timestamp, not by anchor index value."""
        data_readers = {"nordic_radar": None, "meps": None}

        mock_radar = mocker.MagicMock()
        mock_radar.missing = set()
        mock_radar.dates = np.arange(
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T03:00"),
            np.timedelta64(5, "m"),
        )
        mock_radar.has_trajectories = False
        mock_radar.frequency = "5m"

        mock_meps = mocker.MagicMock()
        mock_meps.missing = set()
        mock_meps.dates = np.arange(
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T04:00"),
            np.timedelta64(1, "h"),
        )
        mock_meps.has_trajectories = False
        mock_meps.frequency = "1h"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_radar, mock_meps],
        )
        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 12],
            timestep="5m",
            time_index_mode="sparse",
            shuffle=False,
        )

        # Anchor index 12 corresponds to 2020-01-01T01:00.
        # Requested meps times are [01:00, 02:00] => native indices [1, 2].
        assert dataset._resolve_dataset_time_indices("meps", 12) == [1, 2]

    def test_sparse_aux_indices_allow_phase_shift_via_nearest_timestamp(self, mocker: MockFixture) -> None:
        """Sparse aux lookup should pick nearest native times when exact timestamps are phase-shifted."""
        data_readers = {"nordic_radar": None, "meps": None}

        mock_radar = mocker.MagicMock()
        mock_radar.missing = set()
        mock_radar.dates = np.arange(
            np.datetime64("2020-01-01T00:05"),
            np.datetime64("2020-01-01T03:05"),
            np.timedelta64(5, "m"),
        )
        mock_radar.has_trajectories = False
        mock_radar.frequency = "5m"

        mock_meps = mocker.MagicMock()
        mock_meps.missing = set()
        mock_meps.dates = np.arange(
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T04:00"),
            np.timedelta64(1, "h"),
        )
        mock_meps.has_trajectories = False
        mock_meps.frequency = "1h"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_radar, mock_meps],
        )
        dataset = MultiDataset(
            data_readers=data_readers,
            relative_date_indices=[0, 12],
            timestep="5m",
            time_index_mode="sparse",
            shuffle=False,
        )

        # Anchor index 0 corresponds to 00:05, so requested times are [00:05, 01:05].
        # With nearest lookup on hourly meps data this resolves to [00:00, 01:00] => [0, 1].
        assert dataset._resolve_dataset_time_indices("meps", 0) == [0, 1]
