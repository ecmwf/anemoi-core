# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import torch

from anemoi.training.data.data_reader import ForecastStepDataset


class FakeZarr5D:
    """Fake 5D zarr-like object for testing ForecastStepDataset.

    Shape: (num_inits, variables, ensemble, forecast_steps, gridpoints)
    Uses orthogonal (zarr-style) indexing, not numpy broadcast indexing.
    """

    def __init__(
        self,
        num_inits: int = 4,
        variables: int = 3,
        ensemble: int = 2,
        steps: int = 24,
        gridpoints: int = 10,
        frequency: datetime.timedelta | None = None,
    ):
        self._shape = (num_inits, variables, ensemble, steps, gridpoints)
        self._data = np.random.default_rng(42).standard_normal(self._shape).astype(np.float32)
        self._dates = np.array(
            [np.datetime64("2020-01-01T00:00") + np.timedelta64(12 * i, "h") for i in range(num_inits)],
        )
        self._frequency = frequency or datetime.timedelta(hours=12)

        class _FakeStore:
            path = "fake.zarr"

        self.store = _FakeStore()
        self.attrs = {"variables": [f"var_{i}" for i in range(variables)], "resolution": "o96"}

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dates(self) -> np.ndarray:
        return self._dates

    @property
    def grids(self) -> list[int]:
        return [self._shape[4]]

    @property
    def variables(self) -> list[str]:
        return [f"var_{i}" for i in range(self._shape[1])]

    @property
    def frequency(self) -> datetime.timedelta:
        return self._frequency

    @property
    def resolution(self) -> str:
        return "o96"

    @property
    def name_to_index(self) -> dict[str, int]:
        return {f"var_{i}": i for i in range(self._shape[1])}

    @property
    def missing(self) -> set[int]:
        return set()

    def metadata(self) -> dict:
        return {}

    def supporting_arrays(self) -> dict:
        return {}

    @property
    def statistics(self) -> dict:
        return {"mean": np.zeros(self._shape[1]), "stdev": np.ones(self._shape[1])}

    def __getitem__(self, item: "tuple | int | slice | list | np.ndarray | str"):
        """Zarr-style orthogonal indexing or string-key group access."""
        # Support zarr group-like string key access
        if isinstance(item, str):
            if item == "data":
                return self  # data array is self (has .shape and supports indexing)
            if item == "base_dates":
                return type("Arr", (), {"__getitem__": lambda _s, _k: self._dates})()
            return None
        if not isinstance(item, tuple):
            return self._data[item]

        result = self._data
        removed = 0
        for i, idx in enumerate(item):
            axis = i - removed
            if isinstance(idx, (int, np.integer)):
                result = np.take(result, idx, axis=axis)
                removed += 1
            elif isinstance(idx, slice):
                slices = [slice(None)] * result.ndim
                slices[axis] = idx
                result = result[tuple(slices)]
            elif isinstance(idx, (list, np.ndarray)):
                result = np.take(result, idx, axis=axis)
        return result

    def __contains__(self, key: str) -> bool:
        """Support 'in' checks like a zarr group."""
        return key in ("data", "base_dates")

    @property
    def oindex(self) -> "FakeZarr5D":
        """Support .oindex[...] access (delegates to __getitem__)."""
        return self


def _make_forecast_step_dataset(
    num_inits: int = 4,
    variables: int = 3,
    ensemble: int = 2,
    steps: int = 24,
    gridpoints: int = 10,
    forecast_steps: int = 24,
    step_frequency: str = "1h",
) -> ForecastStepDataset:
    """Create a ForecastStepDataset with a fake 5D zarr backend."""
    dataset = ForecastStepDataset.__new__(ForecastStepDataset)

    if isinstance(step_frequency, str):
        from anemoi.utils.dates import frequency_to_timedelta

        dataset._step_frequency = frequency_to_timedelta(step_frequency)
    else:
        dataset._step_frequency = step_frequency

    fake_zarr = FakeZarr5D(
        num_inits=num_inits,
        variables=variables,
        ensemble=ensemble,
        steps=steps,
        gridpoints=gridpoints,
        frequency=dataset._step_frequency,
    )
    dataset.data = fake_zarr

    dataset._forecast_steps = forecast_steps

    # Set attributes that ForecastStepDataset normally creates in __init__
    dataset._init_dates = fake_zarr.dates
    dataset._init_indices = np.arange(num_inits)
    dataset._zarr = fake_zarr  # stand-in for the zarr group
    dataset._var_indices = list(range(variables))

    return dataset


class TestForecastStepDatasetProperties:
    """Test ForecastStepDataset properties."""

    def test_has_trajectories(self) -> None:
        ds = _make_forecast_step_dataset()
        assert ds.has_trajectories is True

    def test_num_initializations(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=7)
        assert ds.num_initializations == 7

    def test_grid_size(self) -> None:
        ds = _make_forecast_step_dataset(gridpoints=42)
        assert ds.grid_size == 42

    def test_frequency_returns_step_frequency(self) -> None:
        ds = _make_forecast_step_dataset(step_frequency="1h")
        assert ds.frequency == datetime.timedelta(hours=1)

        ds = _make_forecast_step_dataset(step_frequency="6h")
        assert ds.frequency == datetime.timedelta(hours=6)


class TestForecastStepDatasetDates:
    """Test virtual dates expansion."""

    def test_dates_length(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=4, forecast_steps=24)
        assert len(ds.dates) == 4 * 24

    def test_dates_spacing_within_forecast(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=2, forecast_steps=5, step_frequency="1h")
        dates = ds.dates

        # Within the first forecast, dates should be 1h apart
        for i in range(4):
            diff = (dates[i + 1] - dates[i]) / np.timedelta64(1, "s")
            assert diff == 3600.0

    def test_dates_start_from_init_dates(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=3, forecast_steps=4, step_frequency="1h")
        dates = ds.dates
        init_dates = ds.data.dates

        # First date of each forecast block matches the init date
        assert dates[0] == init_dates[0]
        assert dates[4] == init_dates[1]
        assert dates[8] == init_dates[2]

    def test_dates_6h_step_frequency(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=2, forecast_steps=4, step_frequency="6h")
        dates = ds.dates

        for i in range(3):
            diff = (dates[i + 1] - dates[i]) / np.timedelta64(1, "s")
            assert diff == 6 * 3600.0


class TestForecastStepDatasetTrajectoryIds:
    """Test trajectory ID assignment."""

    def test_trajectory_ids_shape(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=4, forecast_steps=10)
        assert len(ds.trajectory_ids) == 40

    def test_trajectory_ids_values(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=3, forecast_steps=5)
        ids = ds.trajectory_ids
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(ids, expected)

    def test_trajectory_ids_unique_per_init(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=6, forecast_steps=12)
        ids = ds.trajectory_ids
        assert len(np.unique(ids)) == 6


class TestForecastStepDatasetGetSample:
    """Test get_sample method."""

    def test_get_sample_with_list_indices(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        # Request steps 2,3,4 from the first initialization
        sample = ds.get_sample(time_indices=[2, 3, 4], grid_shard_indices=slice(None))
        # Output shape: (steps, ensemble, gridpoints, variables)
        assert sample.shape == (3, 2, 10, 3)
        assert isinstance(sample, torch.Tensor)

    def test_get_sample_with_slice_indices(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        sample = ds.get_sample(time_indices=slice(0, 5), grid_shard_indices=slice(None))
        assert sample.shape == (5, 2, 10, 3)

    def test_get_sample_second_initialization(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        # Virtual indices 24-26 map to init_idx=1, step_indices=[0,1,2]
        sample = ds.get_sample(time_indices=[24, 25, 26], grid_shard_indices=slice(None))
        assert sample.shape == (3, 2, 10, 3)

    def test_get_sample_with_grid_shard_slice(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        sample = ds.get_sample(time_indices=[0, 1], grid_shard_indices=slice(0, 5))
        assert sample.shape == (2, 2, 5, 3)

    def test_get_sample_with_grid_shard_array(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        grid_indices = np.array([0, 2, 4, 6, 8])
        sample = ds.get_sample(time_indices=[0, 1, 2], grid_shard_indices=grid_indices)
        assert sample.shape == (3, 2, 5, 3)

    def test_get_sample_returns_correct_data(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        sample = ds.get_sample(time_indices=[0, 1, 2], grid_shard_indices=slice(None))

        # Verify data matches what we'd get from the underlying fake zarr
        # init_idx=0, step_indices=[0,1,2]
        # Orthogonal indexing: data[0, :, :, [0,1,2], :] → (vars, ensemble, 3, gridpoints)
        raw = ds.data[0, slice(None), slice(None), [0, 1, 2], slice(None)]
        expected = np.transpose(raw, (2, 1, 3, 0))  # (steps, ensemble, gridpoints, vars)
        np.testing.assert_allclose(sample.numpy(), expected, rtol=1e-6)

    def test_get_sample_none_grid_indices(self) -> None:
        ds = _make_forecast_step_dataset(
            num_inits=4,
            variables=3,
            ensemble=2,
            steps=24,
            gridpoints=10,
            forecast_steps=24,
        )
        sample = ds.get_sample(time_indices=[0, 1], grid_shard_indices=None)
        assert sample.shape == (2, 2, 10, 3)


class TestForecastStepDatasetValidation:
    """Test initialization validation."""

    def test_rejects_non_5d_data(self) -> None:
        """ForecastStepDataset should raise if the zarr is not 5D."""

        class Fake4D:
            shape = (10, 3, 2, 100)
            dates = np.array([np.datetime64("2020-01-01")])
            grids: ClassVar[list[int]] = [100]
            variables: ClassVar[list[str]] = ["a", "b", "c"]
            frequency = datetime.timedelta(hours=6)
            resolution = "o96"
            name_to_index: ClassVar[dict[str, int]] = {"a": 0, "b": 1, "c": 2}
            missing: ClassVar[set[int]] = set()

            def metadata(self) -> dict:
                return {}

            def supporting_arrays(self) -> dict:
                return {}

            statistics: ClassVar[dict] = {}

        def _raise_for_4d() -> None:
            ds = ForecastStepDataset.__new__(ForecastStepDataset)
            ds.data = Fake4D()
            ds._forecast_steps = 10
            if len(ds.data.shape) != 5:
                shape = ds.data.shape
                msg = (
                    "ForecastStepDataset expects a 5D zarr "
                    f"(inits, variables, ensemble, steps, gridpoints), got shape {shape}"
                )
                raise ValueError(msg)

        with pytest.raises(ValueError, match="expects a 5D zarr"):
            _raise_for_4d()

    def test_rejects_insufficient_forecast_steps(self) -> None:
        """Should raise if dataset has fewer steps than requested."""

        def _raise_for_insufficient_steps() -> None:
            ds = ForecastStepDataset.__new__(ForecastStepDataset)
            ds.data = FakeZarr5D(steps=10)
            ds._forecast_steps = 20
            actual_steps = ds.data.shape[3]
            if actual_steps < ds._forecast_steps:
                msg = f"Dataset has {actual_steps} forecast steps but config requests {ds._forecast_steps}."
                raise ValueError(msg)

        with pytest.raises(ValueError, match="forecast steps but config requests"):
            _raise_for_insufficient_steps()


class TestForecastStepDatasetTree:
    """Test tree representation."""

    def test_tree_includes_forecast_info(self) -> None:
        ds = _make_forecast_step_dataset(num_inits=4, forecast_steps=24, step_frequency="1h")
        # Use repr() which renders the Rich tree to a string
        rendered = repr(ds)
        assert "Forecast steps: 24" in rendered
        assert "Num initializations: 4" in rendered
        assert "Virtual length: 96" in rendered


class TestCreateDatasetWithForecastSteps:
    """Test that create_dataset factory correctly routes to ForecastStepDataset."""

    def test_create_dataset_selects_forecast_reader(self) -> None:
        """create_dataset should instantiate ForecastStepDataset when forecast_steps is set."""
        fake_zarr = FakeZarr5D(num_inits=4, variables=3, ensemble=2, steps=24, gridpoints=10)

        with patch("zarr.open", return_value=fake_zarr):
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake.zarr"},
                "forecast_steps": 24,
                "step_frequency": "1h",
            }
            ds = create_dataset(config)
            assert isinstance(ds, ForecastStepDataset)
            assert ds._forecast_steps == 24
            assert ds.has_trajectories is True

    def test_create_dataset_without_forecast_steps_gives_native(self) -> None:
        """create_dataset without forecast_steps should give NativeGridDataset."""
        fake_zarr = FakeZarr5D(num_inits=10, variables=3, ensemble=2, steps=24, gridpoints=10)

        with patch("anemoi.training.data.data_reader.open_dataset", return_value=fake_zarr):
            from anemoi.training.data.data_reader import NativeGridDataset
            from anemoi.training.data.data_reader import create_dataset

            config = {
                "dataset_config": {"dataset": "fake.zarr"},
            }
            ds = create_dataset(config)
            assert isinstance(ds, NativeGridDataset)
