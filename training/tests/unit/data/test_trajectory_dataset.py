# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TrajectoryDataReader.

A TrajectoryDataReader wraps a 5-D ``(base_dates, variables, ensembles, steps, cells)``
trajectories dataset. Each base date (forecast initialisation) is an independent
sequence and samples select forecast steps within one sequence.
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.data.sample import GriddedSample
from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.data_reader import TrajectoryDataReader
from anemoi.training.data.data_reader import create_dataset


class FakeTrajectoryDataset:
    """Mimics the anemoi-datasets trajectories interface read by TrajectoryDataReader."""

    def __init__(
        self,
        num_base_dates: int = 4,
        variables: int = 3,
        ensemble: int = 2,
        steps: int = 6,
        gridpoints: int = 10,
        step_frequency: datetime.timedelta | None = datetime.timedelta(hours=6),
        missing: set[int] | None = None,
    ) -> None:
        self.shape = (num_base_dates, variables, ensemble, steps, gridpoints)
        self._data = np.random.default_rng(42).standard_normal(self.shape).astype(np.float32)
        self.step_frequency = step_frequency
        self.base_dates = np.datetime64("2020-01-01T00", "s") + np.arange(num_base_dates) * np.timedelta64(1, "D")
        self.steps = np.arange(steps) * np.timedelta64(step_frequency or datetime.timedelta(hours=6))
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

    def __getitem__(self, key: object) -> np.ndarray:
        return self._data[key]


def _make_reader(sampling: dict | None = None, **fake_kwargs) -> TrajectoryDataReader:
    fake = FakeTrajectoryDataset(**fake_kwargs)
    with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake):
        return TrajectoryDataReader(dataset="fake.zarr", sampling=sampling)


class TestTrajectoryDataReaderProperties:

    def test_is_a_gridded_reader_with_trajectories(self) -> None:
        reader = _make_reader()
        assert isinstance(reader, GriddedDataReader)
        assert reader.sample_type is GriddedSample
        assert reader.has_trajectories
        assert not reader.is_tabular

    def test_sequences_are_base_dates_and_positions_are_steps(self) -> None:
        reader = _make_reader(num_base_dates=5, steps=7)
        assert reader.num_sequences == 5
        assert reader.sequence_length() == 7

    def test_grid_size_is_number_of_cells(self) -> None:
        assert _make_reader(gridpoints=42).grid_size == 42

    def test_frequency_is_step_frequency(self) -> None:
        reader = _make_reader(step_frequency=datetime.timedelta(hours=3))
        assert reader.frequency == datetime.timedelta(hours=3)

    def test_missing_step_frequency_raises(self) -> None:
        reader = _make_reader(step_frequency=None)
        with pytest.raises(ValueError, match="step frequency"):
            _ = reader.frequency

    def test_missing_base_dates_have_no_anchors(self) -> None:
        reader = _make_reader(missing={1, 3})
        assert set(reader.valid_anchors([0]).sequences.tolist()) == {0, 2}

    def test_default_sampling(self) -> None:
        assert _make_reader().sampling == {"stride": None}
        assert _make_reader(sampling={"stride": 2}).sampling == {"stride": 2}

    def test_tree_reports_trajectory_settings(self) -> None:
        text = repr(_make_reader(num_base_dates=4, steps=6))
        assert "Num initialisations: 4" in text
        assert "Steps per initialisation: 6" in text


class TestTrajectoryDataReaderOpen:

    def test_start_end_become_base_start_end(self) -> None:
        start, end = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC), datetime.datetime(
            2020,
            2,
            1,
            tzinfo=datetime.UTC,
        )
        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeTrajectoryDataset()) as mock:
            TrajectoryDataReader(dataset="fake.zarr", start=start, end=end)
        mock.assert_called_once_with({"dataset": "fake.zarr"}, base_start=start, base_end=end)

    def test_unset_start_end_are_not_passed(self) -> None:
        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeTrajectoryDataset()) as mock:
            TrajectoryDataReader(dataset="fake.zarr")
        mock.assert_called_once_with({"dataset": "fake.zarr"})

    def test_rejects_frequency_in_dataset_config(self) -> None:
        with (
            patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeTrajectoryDataset()),
            pytest.raises(AssertionError, match="does not accept a 'frequency'"),
        ):
            TrajectoryDataReader(dataset_config={"dataset": "fake.zarr", "frequency": "6h"})


class TestTrajectoryDataReaderGetSample:

    @pytest.mark.parametrize(
        ("positions", "num_steps"),
        [([2, 3, 4], 3), (slice(0, 5), 5), ([0, 4, 5], 3)],
    )
    def test_get_sample_shape(self, positions: list[int] | slice, num_steps: int) -> None:
        reader = _make_reader(variables=3, ensemble=2, gridpoints=10)
        sample = reader.get_sample(1, positions)
        assert isinstance(sample, GriddedSample)
        assert sample.data.shape == (num_steps, 2, 10, 3)  # (steps, ensemble, gridpoints, variables)
        assert sample.variables == reader.variables
        assert sample.grid_size == 10

    def test_get_sample_returns_correct_data(self) -> None:
        reader = _make_reader()
        sample = reader.get_sample(2, [0, 1, 2])
        raw = reader.data[2][:, :, [0, 1, 2], :]  # (vars, ens, steps, grid)
        expected = np.transpose(raw, (2, 1, 3, 0))  # (steps, ens, grid, vars)
        np.testing.assert_allclose(sample.data.numpy(), expected, rtol=1e-6)

    def test_get_sample_coordinates_in_radians(self) -> None:
        reader = _make_reader(gridpoints=10)
        sample = reader.get_sample(0, [0, 1])
        expected = np.deg2rad(np.stack([reader.data.latitudes, reader.data.longitudes], axis=-1))
        np.testing.assert_allclose(sample.coordinates.numpy(), expected, rtol=1e-6)

    def test_get_sample_full_grid_without_reader_group(self) -> None:
        sample = _make_reader(gridpoints=10).get_sample(0, [0, 1])
        assert sample.data.shape[2] == 10
        assert sample.shard_sizes is None

    def test_get_sample_with_grid_shard(self) -> None:
        reader = _make_reader(variables=3, ensemble=2, gridpoints=10)
        full = reader.get_sample(0, [0, 1, 2])

        reader.set_reader_group_info(reader_group_rank=1, reader_group_size=2)
        sample = reader.get_sample(0, [0, 1, 2])

        assert sample.shard_sizes == [5, 5]
        assert sample.data.shape == (3, 2, 5, 3)
        torch.testing.assert_close(sample.data, full.data[:, :, 5:])
        torch.testing.assert_close(sample.coordinates, full.coordinates[5:])


class TestCreateDatasetWithTrajectory:

    @pytest.mark.parametrize("trajectory", [{}, {"sampling": {"stride": 2}}])
    def test_create_dataset_selects_trajectory_reader(self, trajectory: dict) -> None:
        config = OmegaConf.create({"dataset_config": {"dataset": "fake.zarr"}, "trajectory": trajectory})
        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeTrajectoryDataset()):
            reader = create_dataset(config)

        assert isinstance(reader, TrajectoryDataReader)
        assert reader.sampling == trajectory.get("sampling", {"stride": None})

    @pytest.mark.parametrize("trajectory", [None, "absent"])
    def test_create_dataset_without_trajectory_gives_gridded_reader(self, trajectory: str | None) -> None:
        config = {"dataset_config": {"dataset": "fake.zarr"}}
        if trajectory != "absent":
            config["trajectory"] = trajectory
        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeTrajectoryDataset()):
            reader = create_dataset(config)

        assert type(reader) is GriddedDataReader
        assert not reader.has_trajectories
