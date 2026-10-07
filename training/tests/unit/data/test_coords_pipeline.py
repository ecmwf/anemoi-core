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

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from anemoi.models.data import TensorLayout
from anemoi.models.data.batch import Batch
from anemoi.models.data.sample import GriddedSourceSample
from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.multidataset import MultiDataset

if TYPE_CHECKING:
    from pytest_mock import MockFixture

# ---------------------------------------------------------------- Reader API


def _make_reader(grid: int = 5, mocker: MockFixture | None = None) -> GriddedDataReader:
    """Build a GriddedDataReader with a mocked underlying anemoi.datasets payload."""
    mock_data = mocker.MagicMock()
    mock_data.latitudes = np.linspace(-90.0, 90.0, grid)
    mock_data.longitudes = np.linspace(0.0, 360.0, grid, endpoint=False)
    mock_data.grids = (grid,)
    mock_data.shape = (1, 1, 1, grid)
    reader = GriddedDataReader.__new__(GriddedDataReader)
    reader.data = mock_data
    reader.reader_group_rank = 0
    reader.reader_group_size = 1
    reader.grid_shard_sizes = None
    reader.grid_shard_slice = None
    return reader


def test_reader_latitudes_longitudes_in_radians(mocker: MockFixture) -> None:
    reader = _make_reader(grid=4, mocker=mocker)
    np.testing.assert_allclose(reader.latitudes, np.deg2rad([-90.0, -30.0, 30.0, 90.0]), rtol=1e-6)
    np.testing.assert_allclose(reader.longitudes, np.deg2rad([0.0, 90.0, 180.0, 270.0]), rtol=1e-6)


def test_reader_get_coordinates_full_grid(mocker: MockFixture) -> None:
    reader = _make_reader(grid=5, mocker=mocker)
    coords = reader.get_coordinates()
    assert coords.shape == (5, 2)
    assert coords.dtype == torch.float32
    np.testing.assert_allclose(coords[:, 0].numpy(), reader.latitudes)
    np.testing.assert_allclose(coords[:, 1].numpy(), reader.longitudes)


def test_reader_get_coordinates_with_grid_shard(mocker: MockFixture) -> None:
    reader = _make_reader(grid=8, mocker=mocker)
    reader.set_reader_group_info(reader_group_rank=1, reader_group_size=2)
    coords = reader.get_coordinates()
    assert coords.shape == (4, 2)
    np.testing.assert_allclose(coords[:, 0].numpy(), reader.latitudes[4:8])


def test_gridded_reader_is_not_tabular(mocker: MockFixture) -> None:
    reader = _make_reader(grid=3, mocker=mocker)
    assert reader.is_tabular is False


# -------------------------------------------------------- MultiDataset coords


def _make_mock_reader(mocker: MockFixture, grid: int) -> MockFixture:
    reader = mocker.MagicMock()
    reader.num_sequences = 1
    # (sequence, position) anchors of a 20-date series sampled with relative indices [0, 1]
    reader.compute_anchors.return_value = np.column_stack([np.zeros(19, dtype=np.int64), np.arange(19)])
    reader.get_sample.return_value = GriddedSourceSample(
        data=torch.zeros(2, 1, grid, 2),
        variables=["x", "y"],
        layout=TensorLayout(time=0, ensemble=1, grid=2, variables=3),
        coordinates=torch.stack([torch.linspace(-1.0, 1.0, grid), torch.linspace(0.0, 6.0, grid)], dim=-1),
        grid_size=grid,
    )
    return reader


def _make_multidataset(mocker: MockFixture) -> MultiDataset:
    ds = MultiDataset(
        data_readers={
            "a": _make_mock_reader(mocker, grid=6),
            "b": _make_mock_reader(mocker, grid=4),
        },
        relative_date_indices={"a": [0, 1], "b": [0, 1]},
    )
    ds.worker_id = 0  # normally set by worker_init_func
    return ds


def test_multidataset_get_sample_returns_source_samples(mocker: MockFixture) -> None:
    ds = _make_multidataset(mocker)
    sample = ds.get_sample(0)
    assert set(sample) == {"a", "b"}
    assert isinstance(sample["a"], GriddedSourceSample)
    assert sample["a"].coordinates.shape == (6, 2)
    # Relative indices [0, 1] are normalized to a slice and offset by the anchor's position.
    ds.data_readers["a"].get_sample.assert_called_with(0, slice(0, 2, 1))


def test_multidataset_collates_to_batch(mocker: MockFixture) -> None:
    ds = _make_multidataset(mocker)
    samples = [ds.get_sample(0), ds.get_sample(0)]
    batch = Batch.collate(samples)

    assert isinstance(batch, Batch)
    assert set(batch.dataset_names) == {"a", "b"}
    # Data is stacked along the batch dim.
    assert batch["a"].data.shape[0] == 2
    # Gridded coords are static -> shared by reference, no batch dim.
    assert batch["a"].coordinates.shape == (6, 2)
    assert batch.static_coord_datasets == frozenset({"a", "b"})


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
    ds = _make_multidataset(mocker)
    s1 = ds.get_sample(0)
    s2 = ds.get_sample(0)
    batch = Batch.collate([s1, s2])
    # The collated coord tensor is the same Python object as the first sample's.
    assert batch["a"].coordinates is s1["a"].coordinates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
