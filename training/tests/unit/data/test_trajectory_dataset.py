# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TrajectoryDataset (successor to the removed ForecastStepDataset)."""

import datetime
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import torch

from anemoi.training.data.data_reader import TrajectoryDataset


class FakeTrajectoriesDataset:
    """Fake object that mimics the TrajectoriesZarr interface expected by TrajectoryDataset.

    On-disk shape: (num_inits, variables, ensemble, steps, gridpoints)
    """

    def __init__(
        self,
        num_inits: int = 4,
        variables: int = 3,
        ensemble: int = 2,
        steps: int = 24,
        gridpoints: int = 10,
        step_frequency: datetime.timedelta | None = None,
        missing: set[int] | None = None,
    ):
        self._shape = (num_inits, variables, ensemble, steps, gridpoints)
        self._data = np.random.default_rng(42).standard_normal(self._shape).astype(np.float32)
        self._step_frequency = step_frequency or datetime.timedelta(hours=1)
        self._missing: set[int] = missing or set()

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def step_frequency(self) -> datetime.timedelta:
        return self._step_frequency

    @property
    def missing(self) -> set[int]:
        return self._missing

    @property
    def grids(self) -> list[int]:
        return [self._shape[4]]

    @property
    def variables(self) -> list[str]:
        return [f"var_{i}" for i in range(self._shape[1])]

    @property
    def resolution(self) -> str:
        return "o96"

    @property
    def name_to_index(self) -> dict[str, int]:
        return {f"var_{i}": i for i in range(self._shape[1])}

    def dataset_metadata(self) -> dict:
        return {}

    def metadata(self) -> dict:
        return {}

    def supporting_arrays(self) -> dict:
        return {}

    @property
    def statistics(self) -> dict:
        return {"mean": np.zeros(self._shape[1]), "stdev": np.ones(self._shape[1])}

    def __getitem__(self, n: int) -> np.ndarray:
        """Return (variables, ensemble, steps, gridpoints) for a scalar init index."""
        return self._data[n]


def _make_trajectory_dataset(
    num_inits: int = 4,
    variables: int = 3,
    ensemble: int = 2,
    steps: int = 24,
    gridpoints: int = 10,
    step_frequency: str | datetime.timedelta = "1h",
    missing: set[int] | None = None,
    sampling: dict | None = None,
) -> TrajectoryDataset:
    """Create a TrajectoryDataset backed by a fake dataset (no real file I/O)."""
    if isinstance(step_frequency, str):
        from anemoi.utils.dates import frequency_to_timedelta

        step_td = frequency_to_timedelta(step_frequency)
    else:
        step_td = step_frequency

    dataset = TrajectoryDataset.__new__(TrajectoryDataset)
    dataset.data = FakeTrajectoriesDataset(
        num_inits=num_inits,
        variables=variables,
        ensemble=ensemble,
        steps=steps,
        gridpoints=gridpoints,
        step_frequency=step_td,
        missing=missing,
    )
    dataset.default_sampling = sampling if sampling is not None else {"stride": 1}
    return dataset


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetProperties:
    """Test TrajectoryDataset geometry properties."""

    def test_num_sequences(self) -> None:
        ds = _make_trajectory_dataset(num_inits=7)
        assert ds.num_sequences == 7

    def test_sequence_length(self) -> None:
        ds = _make_trajectory_dataset(steps=18)
        assert ds.sequence_length() == 18

    def test_grid_size(self) -> None:
        ds = _make_trajectory_dataset(gridpoints=42)
        assert ds.grid_size == 42

    def test_frequency_returns_step_frequency(self) -> None:
        ds = _make_trajectory_dataset(step_frequency="1h")
        assert ds.frequency == datetime.timedelta(hours=1)

        ds = _make_trajectory_dataset(step_frequency="6h")
        assert ds.frequency == datetime.timedelta(hours=6)

    def test_missing_sequences(self) -> None:
        ds = _make_trajectory_dataset(num_inits=5, missing={1, 3})
        assert ds.missing_sequences == {1, 3}

    def test_missing_positions_is_always_empty(self) -> None:
        """TrajectoryDataset does not track per-step missing values."""
        ds = _make_trajectory_dataset(num_inits=4)
        assert ds.missing_positions(0) == set()
        assert ds.missing_positions(2) == set()

    def test_missing_sequences_empty_when_no_missing(self) -> None:
        ds = _make_trajectory_dataset()
        assert ds.missing_sequences == set()


# ---------------------------------------------------------------------------
# default_sampling
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetDefaultSampling:
    """Test that default_sampling is correctly set."""

    def test_default_sampling_includes_every_valid_anchor(self) -> None:
        """Without explicit sampling, every valid trajectory position is used."""
        ds = _make_trajectory_dataset()
        assert ds.default_sampling == {"stride": 1}

    def test_explicit_stride_stored(self) -> None:
        ds = _make_trajectory_dataset(sampling={"stride": 6})
        assert ds.default_sampling == {"stride": 6}

    def test_stride_1_stored(self) -> None:
        ds = _make_trajectory_dataset(sampling={"stride": 1})
        assert ds.default_sampling == {"stride": 1}


# ---------------------------------------------------------------------------
# get_sample
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetGetSample:
    """Test get_sample(sequence, positions, grid_shard_indices)."""

    def test_get_sample_list_positions(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, steps=24, gridpoints=10)
        sample = ds.get_sample(sequence=0, positions=[2, 3, 4], grid_shard_indices=None)
        assert sample.shape == (3, 2, 10, 3)
        assert isinstance(sample, torch.Tensor)

    def test_get_sample_slice_positions(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, steps=24, gridpoints=10)
        sample = ds.get_sample(sequence=0, positions=slice(0, 5), grid_shard_indices=None)
        assert sample.shape == (5, 2, 10, 3)

    def test_get_sample_second_sequence(self) -> None:
        ds = _make_trajectory_dataset(num_inits=4, variables=3, ensemble=2, steps=24, gridpoints=10)
        sample = ds.get_sample(sequence=2, positions=[0, 1, 2], grid_shard_indices=None)
        assert sample.shape == (3, 2, 10, 3)

    def test_get_sample_with_grid_shard_slice(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, steps=24, gridpoints=10)
        sample = ds.get_sample(sequence=0, positions=[0, 1], grid_shard_indices=slice(0, 5))
        assert sample.shape == (2, 2, 5, 3)

    def test_get_sample_with_grid_shard_array(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, steps=24, gridpoints=10)
        grid_indices = np.array([0, 2, 4, 6, 8])
        sample = ds.get_sample(sequence=0, positions=[0, 1, 2], grid_shard_indices=grid_indices)
        assert sample.shape == (3, 2, 5, 3)

    def test_get_sample_returns_correct_data(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, steps=24, gridpoints=10)
        sample = ds.get_sample(sequence=0, positions=[0, 1, 2], grid_shard_indices=None)
        # data[0] → (vars, ensemble, steps, gridpoints); select steps [0,1,2]
        raw = ds.data[0][:, :, [0, 1, 2], :]  # (vars, ens, 3, grid)
        expected = np.transpose(raw, (2, 1, 3, 0))  # (steps, ens, grid, vars)
        np.testing.assert_allclose(sample.numpy(), expected, rtol=1e-6)

    def test_get_sample_none_grid_indices(self) -> None:
        ds = _make_trajectory_dataset(variables=3, ensemble=2, steps=24, gridpoints=10)
        sample = ds.get_sample(sequence=0, positions=[0, 1], grid_shard_indices=None)
        assert sample.shape == (2, 2, 10, 3)


# ---------------------------------------------------------------------------
# create_dataset factory
# ---------------------------------------------------------------------------


class TestCreateDatasetWithTrajectory:
    """Test that create_dataset factory routes correctly based on the trajectory key."""

    def test_create_dataset_selects_trajectory_reader(self) -> None:
        """create_dataset should instantiate TrajectoryDataset when trajectory key is set."""
        fake = FakeTrajectoriesDataset(num_inits=4, variables=3, ensemble=2, steps=24, gridpoints=10)

        with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake):
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake.zarr"},
                "trajectory": {"sampling": {"stride": None}},
            }
            ds = create_dataset(config)
            assert isinstance(ds, TrajectoryDataset)

    def test_create_dataset_passes_sampling_to_reader(self) -> None:
        """create_dataset should forward trajectory.sampling to TrajectoryDataset."""
        fake = FakeTrajectoriesDataset()

        with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake):
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake.zarr"},
                "trajectory": {"sampling": {"stride": 6}},
            }
            ds = create_dataset(config)
            assert isinstance(ds, TrajectoryDataset)
            assert ds.default_sampling == {"stride": 6}

    def test_compute_anchors_excludes_out_of_bounds_positions(self) -> None:
        ds = _make_trajectory_dataset(num_inits=1, steps=6, sampling={"stride": 1})
        anchors = ds.compute_anchors([-1, 0, 1])

        assert anchors[:, 0].tolist() == [0, 0, 0, 0]
        assert anchors[:, 1].tolist() == [1, 2, 3, 4]

    def test_create_dataset_without_trajectory_gives_native(self) -> None:
        """create_dataset without trajectory key should give NativeGridDataset."""

        class FakeAnalysis:
            shape = (100, 3, 1, 1000)
            dates = np.array([np.datetime64("2020-01-01")])
            grids: ClassVar[list[int]] = [1000]
            variables: ClassVar[list[str]] = ["a", "b", "c"]
            frequency = datetime.timedelta(hours=6)
            resolution = "o96"
            name_to_index: ClassVar[dict[str, int]] = {"a": 0, "b": 1, "c": 2}
            missing: ClassVar[set[int]] = set()
            statistics: ClassVar[dict] = {}

            def metadata(self) -> dict:
                return {}

            def supporting_arrays(self) -> dict:
                return {}

        with patch("anemoi.training.data.data_reader.open_dataset", return_value=FakeAnalysis()):
            from anemoi.training.data.data_reader import NativeGridDataset
            from anemoi.training.data.data_reader import create_dataset

            config = {"dataset_config": {"dataset": "fake.zarr"}}
            ds = create_dataset(config)
            assert isinstance(ds, NativeGridDataset)
