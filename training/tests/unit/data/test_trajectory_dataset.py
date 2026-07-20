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

    @property
    def base_dates(self) -> np.ndarray:
        """Return one base date per initialisation (hourly, starting 2020-01-01)."""
        start = np.datetime64("2020-01-01T00:00", "h")
        return np.array([start + np.timedelta64(i, "h") for i in range(self._shape[0])])

    @property
    def steps(self) -> np.ndarray:
        """Return step offsets, one per step position."""
        freq_s = int(self._step_frequency.total_seconds())
        return np.array(
            [np.timedelta64(i * freq_s, "s") for i in range(self._shape[3])]
        )

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
    dataset.default_sampling = sampling if sampling is not None else {"stride": None}
    dataset._auxiliary_reader = None
    return dataset


# ---------------------------------------------------------------------------
# Auxiliary dataset helpers
# ---------------------------------------------------------------------------


class FakeGriddedDataset:
    """Fake 4-D gridded dataset used as the underlying data for an auxiliary NativeGridDataset.

    Shape: (num_dates, num_variables, ensemble=1, gridpoints)
    """

    def __init__(
        self,
        num_dates: int = 10,
        variables: int = 2,
        gridpoints: int = 10,
        step_hours: int = 6,
    ) -> None:
        self._shape = (num_dates, variables, 1, gridpoints)
        rng = np.random.default_rng(99)
        self._data = rng.standard_normal(self._shape).astype(np.float32)
        self._step_hours = step_hours
        start = np.datetime64("2020-01-01T00:00", "h")
        self._dates = np.array(
            [start + np.timedelta64(i * step_hours, "h") for i in range(num_dates)]
        )

    @property
    def dates(self) -> np.ndarray:
        return self._dates

    @property
    def variables(self) -> list[str]:
        return [f"aux_var_{i}" for i in range(self._shape[1])]

    @property
    def name_to_index(self) -> dict[str, int]:
        return {f"aux_var_{i}": i for i in range(self._shape[1])}

    @property
    def statistics(self) -> dict:
        return {"mean": np.zeros(self._shape[1]), "stdev": np.ones(self._shape[1])}

    @property
    def frequency(self) -> datetime.timedelta:
        return datetime.timedelta(hours=self._step_hours)

    @property
    def grids(self) -> list[int]:
        return [self._shape[3]]

    @property
    def missing(self) -> set[int]:
        return set()

    @property
    def resolution(self) -> str:
        return "o96"

    def metadata(self) -> dict:
        return {}

    def supporting_arrays(self) -> dict:
        return {}

    def __getitem__(self, index) -> np.ndarray:
        """Return (n_dates, n_vars, ensemble, gridpoints) for a list/array of date indices."""
        return self._data[index]


def _make_trajectory_dataset_with_aux(
    num_inits: int = 4,
    traj_variables: int = 3,
    aux_variables: int = 2,
    ensemble: int = 2,
    steps: int = 6,
    gridpoints: int = 10,
    aux_step_hours: int = 6,
    aux_num_dates: int = 10,
) -> TrajectoryDataset:
    """Create a TrajectoryDataset with an attached auxiliary NativeGridDataset.

    The trajectory uses 1-hour steps; the auxiliary dataset uses ``aux_step_hours``-hour steps.
    """
    from anemoi.training.data.data_reader import NativeGridDataset

    ds = _make_trajectory_dataset(
        num_inits=num_inits,
        variables=traj_variables,
        ensemble=ensemble,
        steps=steps,
        gridpoints=gridpoints,
        step_frequency="1h",
    )

    # Build a fake NativeGridDataset for the auxiliary reader without file I/O.
    aux_reader = NativeGridDataset.__new__(NativeGridDataset)
    aux_reader.data = FakeGriddedDataset(
        num_dates=aux_num_dates,
        variables=aux_variables,
        gridpoints=gridpoints,
        step_hours=aux_step_hours,
    )
    aux_reader.default_sampling = {"stride": 1}

    ds._auxiliary_reader = aux_reader
    ds._aux_dates_ns = (
        aux_reader.data.dates.astype("datetime64[ns]").astype(np.int64)
    )
    return ds


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

    def test_default_sampling_is_non_overlapping(self) -> None:
        """Without explicit sampling, stride=None (non-overlapping)."""
        ds = _make_trajectory_dataset()
        assert ds.default_sampling == {"stride": None}

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


# ---------------------------------------------------------------------------
# Auxiliary dataset: properties
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetWithAuxiliaryProperties:
    """Test TrajectoryDataset properties when an auxiliary reader is attached."""

    def test_variables_includes_aux_variables(self) -> None:
        ds = _make_trajectory_dataset_with_aux(traj_variables=3, aux_variables=2)
        assert ds.variables == ["var_0", "var_1", "var_2", "aux_var_0", "aux_var_1"]

    def test_name_to_index_aux_indices_are_offset(self) -> None:
        ds = _make_trajectory_dataset_with_aux(traj_variables=3, aux_variables=2)
        n2i = ds.name_to_index
        # Trajectory variables keep their original indices.
        assert n2i["var_0"] == 0
        assert n2i["var_1"] == 1
        assert n2i["var_2"] == 2
        # Auxiliary variables are offset by the number of trajectory variables.
        assert n2i["aux_var_0"] == 3
        assert n2i["aux_var_1"] == 4

    def test_statistics_are_concatenated(self) -> None:
        ds = _make_trajectory_dataset_with_aux(traj_variables=3, aux_variables=2)
        stats = ds.statistics
        assert stats["mean"].shape == (5,)   # 3 traj + 2 aux
        assert stats["stdev"].shape == (5,)

    def test_no_aux_variables_unchanged(self) -> None:
        """Without auxiliary, variables / name_to_index / statistics are unmodified."""
        ds = _make_trajectory_dataset(variables=3)
        assert ds.variables == ["var_0", "var_1", "var_2"]
        assert ds.name_to_index == {"var_0": 0, "var_1": 1, "var_2": 2}


# ---------------------------------------------------------------------------
# Auxiliary dataset: get_sample
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetWithAuxiliaryGetSample:
    """Test get_sample when an auxiliary reader provides extra variables."""

    def test_output_shape_includes_aux_variables(self) -> None:
        ds = _make_trajectory_dataset_with_aux(
            traj_variables=3, aux_variables=2, ensemble=2, steps=6, gridpoints=10
        )
        sample = ds.get_sample(sequence=0, positions=[0, 1, 2], grid_shard_indices=None)
        # Last dim = traj_variables + aux_variables = 5
        assert sample.shape == (3, 2, 10, 5)
        assert isinstance(sample, torch.Tensor)

    def test_traj_part_matches_raw_trajectory_data(self) -> None:
        """The first traj_variables channels of the output equal the raw trajectory data."""
        ds = _make_trajectory_dataset_with_aux(
            traj_variables=3, aux_variables=2, ensemble=2, steps=6, gridpoints=10
        )
        sample = ds.get_sample(sequence=0, positions=[0, 1, 2], grid_shard_indices=None)
        raw = ds.data[0][:, :, [0, 1, 2], :]  # (vars, ens, steps, grid)
        expected_traj = np.transpose(raw, (2, 1, 3, 0))  # (steps, ens, grid, vars)
        np.testing.assert_allclose(sample[..., :3].numpy(), expected_traj, rtol=1e-6)

    def test_forward_fill_uses_last_aux_date_leq_step_time(self) -> None:
        """Steps whose valid time falls between aux dates use the most recent aux value."""
        # Aux dates at 6h intervals: T+0h, T+6h, T+12h, ...
        # Step 0 valid time = base_date + 0h → aux index 0
        # Step 1 valid time = base_date + 1h → still aux index 0 (forward-filled)
        # Step 6 valid time = base_date + 6h → aux index 1
        ds = _make_trajectory_dataset_with_aux(
            traj_variables=3, aux_variables=2, ensemble=2, steps=12,
            gridpoints=10, aux_step_hours=6, aux_num_dates=10,
        )
        base_date_ns = ds.data.base_dates[0].astype("datetime64[ns]").astype(np.int64)
        steps_ns = ds.data.steps[[0, 1, 6]].astype("timedelta64[ns]").astype(np.int64)
        abs_ts_ns = base_date_ns + steps_ns
        expected_aux_indices = np.searchsorted(ds._aux_dates_ns, abs_ts_ns, side="right") - 1

        sample = ds.get_sample(sequence=0, positions=[0, 1, 6], grid_shard_indices=None)
        aux_part = sample[..., 3:]  # (3 steps, ens=2, grid=10, aux_vars=2)

        raw_aux = ds._auxiliary_reader.data[expected_aux_indices.tolist(), :, :, :]
        # raw_aux shape: (steps, aux_vars, ens=1, grid) → transpose → (steps, ens=1, grid, aux_vars)
        # The aux ensemble dim is broadcast to match the trajectory ensemble size.
        expected_aux = np.transpose(raw_aux, (0, 2, 3, 1))
        expected_aux = np.broadcast_to(expected_aux, aux_part.shape)
        np.testing.assert_allclose(aux_part.numpy(), expected_aux, rtol=1e-6)

    def test_forward_fill_clips_before_first_aux_date(self) -> None:
        """If a step valid time is before all aux dates, clip to index 0."""
        ds = _make_trajectory_dataset_with_aux(
            traj_variables=3, aux_variables=2, ensemble=2, steps=6,
            gridpoints=10, aux_step_hours=6, aux_num_dates=10,
        )
        # Step 0 valid time == first aux date → searchsorted(..., side='right') - 1 = 0
        # (no negative index is possible due to the clip)
        sample = ds.get_sample(sequence=0, positions=[0], grid_shard_indices=None)
        assert sample.shape == (1, 2, 10, 5)

    def test_get_sample_with_grid_shard_slice_and_aux(self) -> None:
        ds = _make_trajectory_dataset_with_aux(
            traj_variables=3, aux_variables=2, ensemble=2, steps=6, gridpoints=10
        )
        sample = ds.get_sample(sequence=0, positions=[0, 1], grid_shard_indices=slice(0, 5))
        assert sample.shape == (2, 2, 5, 5)

    def test_get_sample_with_grid_shard_array_and_aux(self) -> None:
        ds = _make_trajectory_dataset_with_aux(
            traj_variables=3, aux_variables=2, ensemble=2, steps=6, gridpoints=10
        )
        grid_idx = np.array([0, 2, 4])
        sample = ds.get_sample(sequence=0, positions=[0, 1, 2], grid_shard_indices=grid_idx)
        assert sample.shape == (3, 2, 3, 5)


# ---------------------------------------------------------------------------
# create_dataset factory: auxiliary
# ---------------------------------------------------------------------------


class TestCreateDatasetWithAuxiliary:
    """Test that create_dataset correctly wires up the auxiliary reader."""

    def test_create_dataset_with_auxiliary_key(self) -> None:
        """create_dataset should open the auxiliary dataset and attach it."""
        fake_traj = FakeTrajectoriesDataset(
            num_inits=4, variables=3, ensemble=2, steps=6, gridpoints=10
        )
        fake_aux = FakeGriddedDataset(num_dates=10, variables=2, gridpoints=10, step_hours=6)

        with patch(
            "anemoi.training.data.data_reader.open_dataset",
            side_effect=[fake_traj, fake_aux],
        ):
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake_traj.zarr"},
                "trajectory": {"sampling": {"stride": None}},
                "auxiliary": {
                    "dataset_config": {"dataset": "fake_aux.zarr"},
                },
            }
            ds = create_dataset(config)

        assert isinstance(ds, TrajectoryDataset)
        assert ds._auxiliary_reader is not None
        assert ds.variables == [
            "var_0", "var_1", "var_2", "aux_var_0", "aux_var_1"
        ]

    def test_auxiliary_without_dataset_config_raises(self) -> None:
        """auxiliary block missing dataset_config must raise ValueError."""
        import pytest

        fake_traj = FakeTrajectoriesDataset()

        with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake_traj):
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake_traj.zarr"},
                "trajectory": {},
                "auxiliary": {},   # missing dataset_config
            }
            with pytest.raises(ValueError, match="auxiliary must contain 'dataset_config'"):
                create_dataset(config)

    def test_no_auxiliary_key_leaves_auxiliary_reader_none(self) -> None:
        """Without auxiliary key, _auxiliary_reader stays None."""
        fake_traj = FakeTrajectoriesDataset()

        with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake_traj):
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake_traj.zarr"},
                "trajectory": {},
            }
            ds = create_dataset(config)

        assert isinstance(ds, TrajectoryDataset)
        assert ds._auxiliary_reader is None
