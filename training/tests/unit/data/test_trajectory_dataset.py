# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TrajectoryDataReader.

A TrajectoryDataReader is a gridded reader over a date-indexed dataset whose dates form
consecutive forecast runs ("trajectories") of ``trajectory_length`` steps, counted from
``trajectory_start``. ``trajectory_ids`` labels each date with its run, so that sample
windows never cross runs (see ``get_usable_indices`` and ``test_compute_valid_data_indices.py``).
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.data.sample import GriddedSourceSample
from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.data_reader import TrajectoryDataReader
from anemoi.training.data.data_reader import create_dataset

# Naive, like the dataset dates (numpy datetime64).
START = np.datetime64("2020-01-01T00:00:00", "s").astype(datetime.datetime)


class FakeGriddedDataset:
    """Fake object that mimics the anemoi-datasets interface read by gridded readers.

    Shape: (dates, variables, ensemble, gridpoints).
    """

    def __init__(
        self,
        num_dates: int = 24,
        variables: int = 3,
        ensemble: int = 2,
        gridpoints: int = 10,
        frequency: datetime.timedelta = datetime.timedelta(hours=1),
        first_date: datetime.datetime = START,
        missing: set[int] | None = None,
    ) -> None:
        self.shape = (num_dates, variables, ensemble, gridpoints)
        self._data = np.random.default_rng(42).standard_normal(self.shape).astype(np.float32)
        self.frequency = frequency
        self.dates = np.datetime64(first_date, "s") + np.arange(num_dates) * np.timedelta64(frequency)
        self.missing = missing or set()
        self.variables = [f"var_{i}" for i in range(variables)]
        self.name_to_index = {name: i for i, name in enumerate(self.variables)}
        self.resolution = "o96"
        self.grids = [gridpoints]
        self.latitudes = np.linspace(-90.0, 90.0, gridpoints)
        self.longitudes = np.linspace(0.0, 350.0, gridpoints)
        self.statistics = {"mean": np.zeros(variables), "stdev": np.ones(variables)}

    def metadata(self) -> dict:
        return {}

    def supporting_arrays(self) -> dict:
        return {}

    def __getitem__(self, key: tuple) -> np.ndarray:
        return self._data[key]


def _make_trajectory_dataset(
    trajectory_length: int = 6,
    trajectory_start: datetime.datetime = START,
    **fake_kwargs,
) -> TrajectoryDataReader:
    """Create a TrajectoryDataReader backed by a fake dataset (no real file I/O)."""
    fake = FakeGriddedDataset(**fake_kwargs)
    with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake):
        return TrajectoryDataReader(
            trajectory_start=trajectory_start,
            trajectory_length=trajectory_length,
            dataset="fake.zarr",
        )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestTrajectoryDataReaderProperties:
    """Test TrajectoryDataReader properties."""

    def test_is_a_gridded_reader_with_trajectories(self) -> None:
        ds = _make_trajectory_dataset()
        assert isinstance(ds, GriddedDataReader)
        assert ds.has_trajectories
        assert ds.is_static_grid

    def test_stores_trajectory_start_and_length(self) -> None:
        ds = _make_trajectory_dataset(trajectory_length=12, trajectory_start=START)
        assert ds.trajectory_length == 12
        assert ds.trajectory_start == START

    def test_all_dates_form_a_single_sequence(self) -> None:
        """Runs are delimited by ``trajectory_ids``, not by separate sequences."""
        ds = _make_trajectory_dataset(num_dates=24)
        assert ds.num_sequences == 1
        assert len(ds.dates) == 24

    def test_grid_size(self) -> None:
        ds = _make_trajectory_dataset(gridpoints=42)
        assert ds.grid_size == 42

    def test_frequency(self) -> None:
        ds = _make_trajectory_dataset(frequency=datetime.timedelta(hours=6))
        assert ds.frequency == datetime.timedelta(hours=6)

    def test_missing_dates_are_passed_through(self) -> None:
        ds = _make_trajectory_dataset(missing={1, 3})
        assert ds.missing == {1, 3}

    def test_tree_reports_trajectory_settings(self) -> None:
        ds = _make_trajectory_dataset(trajectory_length=6)
        text = repr(ds)
        assert "Trajectory start" in text
        assert "Trajectory length: 6 steps" in text


# ---------------------------------------------------------------------------
# trajectory_ids
# ---------------------------------------------------------------------------


class TestTrajectoryIds:
    """Each date is labelled with the forecast run it belongs to."""

    def test_consecutive_runs_of_trajectory_length(self) -> None:
        ds = _make_trajectory_dataset(num_dates=12, trajectory_length=4)
        np.testing.assert_array_equal(ds.trajectory_ids, np.repeat([0, 1, 2], 4))

    def test_one_id_per_date(self) -> None:
        ds = _make_trajectory_dataset(num_dates=24, trajectory_length=6)
        assert len(ds.trajectory_ids) == len(ds.dates)

    def test_runs_are_counted_from_trajectory_start(self) -> None:
        """Dates before ``trajectory_start`` belong to earlier runs (negative ids)."""
        ds = _make_trajectory_dataset(
            num_dates=8,
            trajectory_length=4,
            trajectory_start=START + datetime.timedelta(hours=2),
        )
        np.testing.assert_array_equal(ds.trajectory_ids, [-1, -1, 0, 0, 0, 0, 1, 1])

    def test_length_is_in_steps_of_the_dataset_frequency(self) -> None:
        """``trajectory_length`` counts steps, so the run duration scales with the frequency."""
        ds = _make_trajectory_dataset(num_dates=8, trajectory_length=2, frequency=datetime.timedelta(hours=6))
        np.testing.assert_array_equal(ds.trajectory_ids, [0, 0, 1, 1, 2, 2, 3, 3])


# ---------------------------------------------------------------------------
# get_sample
# ---------------------------------------------------------------------------


class TestTrajectoryDataReaderGetSample:
    """Test get_sample(time_indices)."""

    @pytest.mark.parametrize(
        ("time_indices", "num_steps"),
        [([2, 3, 4], 3), (slice(0, 5), 5), ([0, 6, 7], 3)],
    )
    def test_get_sample_shape(self, time_indices: list[int] | slice, num_steps: int) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, gridpoints=10)
        sample = ds.get_sample(time_indices)
        assert isinstance(sample, GriddedSourceSample)
        assert isinstance(sample.data, torch.Tensor)
        assert sample.data.shape == (num_steps, 2, 10, 3)  # (dates, ensemble, gridpoints, variables)
        assert sample.variables == ds.variables

    def test_get_sample_returns_correct_data(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, gridpoints=10)
        sample = ds.get_sample([0, 1, 2])
        raw = ds.data[[0, 1, 2], :, :, :]  # (dates, vars, ens, grid)
        expected = np.transpose(raw, (0, 2, 3, 1))  # (dates, ens, grid, vars)
        np.testing.assert_allclose(sample.data.numpy(), expected, rtol=1e-6)

    def test_get_sample_coordinates_in_radians(self) -> None:
        ds = _make_trajectory_dataset(gridpoints=10)
        sample = ds.get_sample([0, 1])
        assert sample.coordinates.shape == (10, 2)
        expected = np.deg2rad(np.stack([ds.data.latitudes, ds.data.longitudes], axis=-1))
        np.testing.assert_allclose(sample.coordinates.numpy(), expected, rtol=1e-6)

    def test_get_sample_full_grid_without_reader_group(self) -> None:
        ds = _make_trajectory_dataset(gridpoints=10)
        sample = ds.get_sample([0, 1])
        assert sample.data.shape[2] == 10
        assert sample.shard_sizes is None

    def test_get_sample_with_grid_shard(self) -> None:
        """With a reader group, each reader returns its share of the grid."""
        ds = _make_trajectory_dataset(variables=3, ensemble=2, gridpoints=10)
        full = ds.get_sample([0, 1, 2])

        ds.set_reader_group_info(reader_group_rank=1, reader_group_size=2)
        sample = ds.get_sample([0, 1, 2])

        assert ds.grid_shard_sizes == [5, 5]
        assert sample.shard_sizes == [5, 5]
        assert sample.data.shape == (3, 2, 5, 3)
        torch.testing.assert_close(sample.data, full.data[:, :, 5:])
        torch.testing.assert_close(sample.coordinates, full.coordinates[5:])


# ---------------------------------------------------------------------------
# create_dataset factory
# ---------------------------------------------------------------------------


class TestCreateDatasetWithTrajectory:
    """Test that create_dataset routes on the ``trajectory`` key."""

    def test_create_dataset_selects_trajectory_reader(self) -> None:
        """A ``trajectory`` section with ``start`` and ``length`` gives a TrajectoryDataReader."""
        config = OmegaConf.create(
            {
                "dataset_config": {"dataset": "fake.zarr"},
                "trajectory": {"start": "2020-01-01T00:00:00", "length": 6},
            },
        )
        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeGriddedDataset()):
            ds = create_dataset(config)

        assert isinstance(ds, TrajectoryDataReader)
        assert ds.trajectory_length == 6
        np.testing.assert_array_equal(ds.trajectory_ids, np.repeat([0, 1, 2, 3], 6))

    @pytest.mark.parametrize("trajectory", [None, "absent"])
    def test_create_dataset_without_trajectory_gives_gridded_reader(self, trajectory: str | None) -> None:
        """No ``trajectory`` key, or ``trajectory: null``, gives a plain GriddedDataReader."""
        config = {"dataset_config": {"dataset": "fake.zarr"}}
        if trajectory != "absent":
            config["trajectory"] = trajectory
        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeGriddedDataset()):
            ds = create_dataset(config)

        assert type(ds) is GriddedDataReader
        assert not ds.has_trajectories
