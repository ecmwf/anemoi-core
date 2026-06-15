# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for coordinate-aware reader API and MultiDataset emit_coords path."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_mock import MockFixture

from anemoi.models.data.batch import STATIC_COORDS_META_KEY
from anemoi.models.data.batch import Batch
from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.multidataset import MultiDataset

# ---------------------------------------------------------------- Reader API


def _make_reader(grid: int = 5, mocker: MockFixture | None = None) -> NativeGridDataset:
    """Build a NativeGridDataset with a mocked underlying anemoi.datasets payload."""
    mock_data = mocker.MagicMock()
    mock_data.latitudes = np.linspace(-90.0, 90.0, grid)
    mock_data.longitudes = np.linspace(0.0, 360.0, grid, endpoint=False)
    mock_data.grids = (grid,)
    reader = NativeGridDataset.__new__(NativeGridDataset)
    reader.data = mock_data
    return reader


def test_reader_latitudes_longitudes_in_radians(mocker: MockFixture) -> None:
    reader = _make_reader(grid=4, mocker=mocker)
    np.testing.assert_allclose(reader.latitudes, np.deg2rad([-90.0, -30.0, 30.0, 90.0]))
    np.testing.assert_allclose(reader.longitudes, np.deg2rad([0.0, 90.0, 180.0, 270.0]))


def test_reader_get_coordinates_full_grid(mocker: MockFixture) -> None:
    reader = _make_reader(grid=5, mocker=mocker)
    coords = reader.get_coordinates()
    assert set(coords) == {"latitudes", "longitudes"}
    assert coords["latitudes"].shape == (5,)
    assert coords["longitudes"].dtype == torch.float64
    np.testing.assert_allclose(coords["latitudes"].numpy(), reader.latitudes)


def test_reader_get_coordinates_with_grid_shard(mocker: MockFixture) -> None:
    reader = _make_reader(grid=8, mocker=mocker)
    coords = reader.get_coordinates(grid_shard_indices=slice(2, 6))
    assert coords["latitudes"].shape == (4,)
    np.testing.assert_allclose(coords["latitudes"].numpy(), reader.latitudes[2:6])


def test_reader_is_static_grid_default_true(mocker: MockFixture) -> None:
    reader = _make_reader(grid=3, mocker=mocker)
    assert reader.is_static_grid is True


# -------------------------------------------------------- MultiDataset coords


def _make_multidataset(
    mocker: MockFixture,
    *,
    a_static: bool = True,
    b_static: bool = True,
) -> MultiDataset:
    grid_a, grid_b = 6, 4

    mock_a = mocker.MagicMock()
    mock_a.missing = set()
    mock_a.dates = list(range(20))
    mock_a.has_trajectories = False
    mock_a.frequency = "3h"
    mock_a.is_static_grid = a_static
    mock_a.get_sample.return_value = torch.zeros(1, 1, grid_a, 2)
    mock_a.get_coordinates.return_value = {
        "latitudes": torch.linspace(-1.0, 1.0, grid_a),
        "longitudes": torch.linspace(0.0, 6.0, grid_a),
    }

    mock_b = mocker.MagicMock()
    mock_b.missing = set()
    mock_b.dates = list(range(20))
    mock_b.has_trajectories = False
    mock_b.frequency = "3h"
    mock_b.is_static_grid = b_static
    mock_b.get_sample.return_value = torch.zeros(1, 1, grid_b, 2)
    mock_b.get_coordinates.return_value = {
        "latitudes": torch.linspace(-0.5, 0.5, grid_b),
        "longitudes": torch.linspace(0.0, 3.0, grid_b),
    }

    return MultiDataset(
        data_readers={"a": mock_a, "b": mock_b},
        relative_date_indices={"a": [0, 1], "b": [0, 1]},
    )


def test_multidataset_emit_coords_payload(mocker: MockFixture) -> None:
    ds = _make_multidataset(mocker)
    sample = ds.get_sample(0)
    assert set(sample) == {"a", "b"}
    assert set(sample["a"]) == {"data", "coords"}
    assert set(sample["a"]["coords"]) == {"latitudes", "longitudes"}


def test_multidataset_static_dataset_detection(mocker: MockFixture) -> None:
    ds = _make_multidataset(mocker, a_static=True, b_static=False)
    assert ds.static_coord_datasets == ("a",)


def test_multidataset_emit_coords_collates_to_batch(mocker: MockFixture) -> None:
    ds = _make_multidataset(mocker)
    samples = [ds.get_sample(0), ds.get_sample(0)]
    batch = Batch.collate(samples, static_coord_datasets=ds.static_coord_datasets)

    assert isinstance(batch, Batch)
    assert set(batch.dataset_names) == {"a", "b"}
    # Data is stacked along the batch dim.
    assert batch.data["a"].shape[0] == 2
    # Both datasets are static -> coords shared by reference, no batch dim.
    assert batch.coords["a"]["latitudes"].shape == (6,)
    assert STATIC_COORDS_META_KEY in batch.metadata
    assert batch.metadata[STATIC_COORDS_META_KEY] == frozenset({"a", "b"})


def test_multidataset_dynamic_dataset_stacks_coords(mocker: MockFixture) -> None:
    ds = _make_multidataset(mocker, a_static=True, b_static=False)
    samples = [ds.get_sample(0), ds.get_sample(0)]
    batch = Batch.collate(samples, static_coord_datasets=ds.static_coord_datasets)

    # Static: shared by reference, no batch dim.
    assert batch.coords["a"]["latitudes"].shape == (6,)
    # Dynamic: stacked along a new batch dim.
    assert batch.coords["b"]["latitudes"].shape == (2, 4)


# ------------------------------------------------------ DataModule collate_fn


def test_datamodule_collate_factory_returns_batch(mocker: MockFixture) -> None:
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    dm = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    ds = _make_multidataset(mocker)
    collate = dm._make_collate_fn(ds)
    assert callable(collate)
    batch = collate([ds.get_sample(0), ds.get_sample(0)])
    assert isinstance(batch, Batch)


# ---------------------------------------------------------- Memory invariant


def test_static_coords_share_same_object_through_full_pipeline(mocker: MockFixture) -> None:
    """End-to-end: the static reader's coord tensor object survives collate."""
    ds = _make_multidataset(mocker, a_static=True, b_static=True)
    s1 = ds.get_sample(0)
    s2 = ds.get_sample(0)
    batch = Batch.collate([s1, s2], static_coord_datasets=ds.static_coord_datasets)
    # The collated coord tensor is the same Python object as the first sample's.
    assert batch.coords["a"]["latitudes"] is s1["a"]["coords"]["latitudes"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
