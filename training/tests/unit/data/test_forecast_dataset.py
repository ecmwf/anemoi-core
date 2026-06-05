# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the 5-D ForecastDataset reader and the (sequence, position) anchor model.

These tests build a small on-disk ``trajectories``-layout zarr and open it through
``anemoi.datasets.open_dataset`` via the ForecastDataset reader, so the full 5-D path
(reader -> anchors -> get_sample) is exercised end to end.
"""

import datetime

import numpy as np
import pytest
import torch
import zarr

from anemoi.training.data.data_reader import ForecastDataset
from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import offset_time_indices


def make_trajectories_zarr(
    path: str,
    n_base: int = 4,
    n_vars: int = 3,
    n_ens: int = 1,
    n_steps: int = 5,
    n_cells: int = 10,
    base_frequency_h: int = 6,
    step_hours: list[int] | None = None,
    missing: tuple[int, ...] = (),
) -> str:
    """Build a minimal on-disk ``trajectories``-layout zarr and return its path."""
    if step_hours is None:
        step_hours = [6 * (i + 1) for i in range(n_steps)]
    variables = [chr(ord("a") + i) for i in range(n_vars)]

    root = zarr.open(path, mode="w")
    base_dates = np.array(
        [datetime.datetime(2021, 1, 1) + datetime.timedelta(hours=base_frequency_h * i) for i in range(n_base)],
        dtype="datetime64[s]",
    )
    steps = np.array([np.timedelta64(h, "h") for h in step_hours])
    data = np.arange(n_base * n_vars * n_ens * n_steps * n_cells, dtype="float32").reshape(
        n_base,
        n_vars,
        n_ens,
        n_steps,
        n_cells,
    )
    root.create_dataset("data", data=data, chunks=data.shape)
    root.create_dataset("base_dates", data=base_dates)
    root.create_dataset("steps", data=steps)
    root.create_dataset("latitudes", data=np.linspace(-90, 90, n_cells))
    root.create_dataset("longitudes", data=np.linspace(0, 360, n_cells))
    for key in ("mean", "stdev", "minimum", "maximum"):
        root.create_dataset(key, data=np.zeros(n_vars))

    root.attrs.update(
        layout="trajectories",
        frequency=f"{base_frequency_h}h",
        resolution="o96",
        name_to_index={v: i for i, v in enumerate(variables)},
        variables_metadata={v: {} for v in variables},
    )
    if missing:
        root.attrs["missing_dates"] = [str(base_dates[i]) for i in missing]
    return path


@pytest.fixture
def traj_path(tmp_path) -> str:
    return make_trajectories_zarr(str(tmp_path / "forecast.zarr"))


class TestForecastDataset:
    def test_factory_creates_forecast_dataset(self, traj_path: str) -> None:
        reader = create_dataset({"dataset_config": {"dataset": traj_path}, "forecast": {"sampling": "all"}})
        assert isinstance(reader, ForecastDataset)
        assert reader.default_sampling == "all"

    def test_metadata_uses_step_axis(self, traj_path: str) -> None:
        reader = ForecastDataset(dataset=traj_path)
        assert reader.num_sequences == 4
        assert reader.sequence_length() == 5
        assert reader.frequency == datetime.timedelta(hours=6)
        assert reader.variables == ["a", "b", "c"]
        assert reader.grid_size == 10
        assert reader.statistics_tendencies("6h") is None

    def test_anchors_default_non_overlapping(self, traj_path: str) -> None:
        reader = ForecastDataset(dataset=traj_path)  # default sampling = non_overlapping
        anchors = reader.compute_anchors([0, 1, 2])
        # window = 3, each of 4 inits yields one anchor at position 0 (1 non-overlapping window fits in 5 steps)
        assert np.array_equal(anchors[:, 0], np.arange(4))
        assert np.array_equal(anchors[:, 1], np.zeros(4))

    def test_anchors_all_sampling(self, traj_path: str) -> None:
        reader = ForecastDataset(dataset=traj_path)
        anchors = reader.compute_anchors([0, 1, 2], sampling="all")
        # positions 0, 1, 2 valid in each of 4 inits (5 steps, max offset 2) -> 12 anchors
        assert anchors.shape == (12, 2)
        assert set(np.unique(anchors[:, 1])) == {0, 1, 2}

    def test_anchors_never_cross_initialisation(self, traj_path: str) -> None:
        reader = ForecastDataset(dataset=traj_path)
        anchors = reader.compute_anchors([0, 1], sampling="all")
        for seq, pos in anchors:
            # the whole window pos..pos+max_offset stays inside one sequence
            assert 0 <= pos and pos + 1 < reader.sequence_length(seq)

    def test_missing_base_date_dropped(self, tmp_path) -> None:
        path = make_trajectories_zarr(str(tmp_path / "m.zarr"), missing=(1,))
        reader = ForecastDataset(dataset=path)
        assert reader.missing_sequences == {1}
        anchors = reader.compute_anchors([0, 1], sampling="all")
        assert 1 not in set(anchors[:, 0])  # missing init produces no anchors

    def test_get_sample_values_and_shape(self, traj_path: str) -> None:
        reader = ForecastDataset(dataset=traj_path)
        rel = [0, 1, 2]
        seq = 2
        positions = offset_time_indices(0, rel)
        sample = reader.get_sample(seq, positions, slice(None))

        assert isinstance(sample, torch.Tensor)
        # (steps, ensemble, gridpoints, variables)
        assert tuple(sample.shape) == (3, 1, 10, 3)

        raw = reader.data[seq][:, :, [0, 1, 2], :]  # (vars, ens, steps, cells)
        expected = np.transpose(raw, (2, 1, 3, 0))  # steps, ens, cells, vars
        assert np.allclose(sample.numpy(), expected)

    def test_get_sample_grid_shard(self, traj_path: str) -> None:
        reader = ForecastDataset(dataset=traj_path)
        sample = reader.get_sample(0, [0, 1], grid_shard_indices=slice(0, 4))
        assert tuple(sample.shape) == (2, 1, 4, 3)

        grid = np.array([0, 5, 9])
        sample = reader.get_sample(0, [0, 1], grid_shard_indices=grid)
        assert tuple(sample.shape) == (2, 1, 3, 3)


class TestAnchorModelAnalysis:
    """Anchor model backwards-compatibility for analysis (NativeGridDataset)."""

    def test_analysis_anchor_is_flat_time_index(self) -> None:
        # Build a NativeGridDataset without opening a dataset; drive the anchor model directly.
        reader = NativeGridDataset.__new__(NativeGridDataset)

        class _Series:
            dates = list(range(10))
            missing: set[int] = set()

        reader.data = _Series()

        anchors = reader.compute_anchors([0, 1, 2])
        # one sequence (seq=0); positions 0..7 (10 - max offset 2 - 1)
        assert np.array_equal(anchors[:, 0], np.zeros(8))
        assert np.array_equal(anchors[:, 1], np.arange(8))


def test_compute_valid_anchors_intersection(traj_path: str) -> None:
    reader = ForecastDataset(dataset=traj_path)
    # Two readers over the same forecast dataset with different windows.
    anchors = compute_valid_anchors(
        {"a": reader, "b": reader},
        {"a": [0, 1], "b": [0, 1, 2]},
    )
    # Intersection is bounded by the tighter window ([0,1,2]); each init contributes
    # its non-overlapping anchor at position 0.
    assert anchors.shape[1] == 2
    assert set(np.unique(anchors[:, 1])) <= {0}
