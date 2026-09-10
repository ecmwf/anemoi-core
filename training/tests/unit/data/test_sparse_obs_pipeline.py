# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the sparse-observation reader/batch pipeline.

Covers:

* :meth:`ObservationDataReader.get_sample` — single-round-trip unpack.
* :meth:`Batch.collate` on a mixed gridded + sparse batch.
* :meth:`Batch.to` on the same mixed batch (CPU-only round-trip; the
  test asserts behaviour, not GPU availability).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from anemoi.models.data.batch import BOUNDARIES_META_KEY
from anemoi.models.data.batch import STATIC_COORDS_META_KEY
from anemoi.models.data.batch import Batch
from anemoi.models.data.batch import TensorLayout
from anemoi.training.data.data_reader import ObservationDataReader

_DATASET_NAME = "npp_atms"


def test_make_anemoi_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _make_obs_payload(v=5)
    dataset = _make_obs_reader(payload).data
    monkeypatch.setattr("anemoi.training.data.data_reader.open_dataset", lambda _config: dataset)
    reader = ObservationDataReader(dataset_config={"dataset": "test-observations"})

    sample = reader.get_sample(slice(0, 2))

    dataset.__getitem__.assert_called_once_with(slice(0, 2))
    assert sample["data"].shape == (1, 5, 5)
    assert sample["variables"] == dataset.variables
    assert sample["statistics"] is dataset.statistics


def test_batch_collate_and_to() -> None:
    """Test that we can collate a batch of two observation samples and move it to the GPU."""
    reader = _make_obs_reader(_make_obs_payload())

    # ``Batch`` is a per-dataset envelope, so each sample must be wrapped
    # under its dataset name (here "npp_atms") before collation.
    sample1 = {_DATASET_NAME: reader.get_sample(slice(20, 24))}
    sample2 = {_DATASET_NAME: reader.get_sample(slice(40, 44))}

    # Collate the samples into a batch.
    batch = Batch.collate([sample1, sample2], static_coord_datasets=())

    # Move the batch to the GPU (if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved_batch = batch.to(device)

    # Assert that the data is on the correct device.
    assert moved_batch.data[_DATASET_NAME][0].device.type == device.type
    assert moved_batch.data[_DATASET_NAME][1].device.type == device.type


# ----------------------------------------------------------------- helpers


def _make_obs_payload(n: int = 5, v: int = 3, n_times: int = 2) -> SimpleNamespace:
    """Return controlled observation dataset output.

    Boundaries partition all rows into the requested time windows.
    """
    boundaries = tuple(slice(i * n // n_times, (i + 1) * n // n_times) for i in range(n_times))
    return SimpleNamespace(
        data=np.arange(n * v, dtype=np.float32).reshape(n, v),
        latitudes=np.linspace(-90.0, 90.0, n, dtype=np.float64),
        longitudes=np.linspace(0.0, 360.0, n, endpoint=False, dtype=np.float64),
        timedeltas=np.linspace(0.0, 3600.0, n, dtype=np.float64),
        boundaries=boundaries,
    )


def _make_obs_reader(payload: SimpleNamespace) -> ObservationDataReader:
    """Attach controlled dataset output to the real reader."""
    dataset = MagicMock()
    dataset.__getitem__.return_value = payload
    dataset.variables = [f"variable_{i}" for i in range(payload.data.shape[1])]
    dataset.statistics = {"mean": np.zeros(payload.data.shape[1], dtype=np.float32)}
    reader = ObservationDataReader.__new__(ObservationDataReader)
    reader.data = dataset
    reader.reader_group_rank = 0
    reader.reader_group_size = 1
    return reader


def _make_obs_sample(n: int = 5, v: int = 3, n_times: int = 2) -> dict:
    """Build a sparse sample matching the ObservationDataReader contract."""
    return _make_obs_reader(_make_obs_payload(n=n, v=v, n_times=n_times)).get_sample(slice(0, n_times))


def _make_grid_sample(grid: int = 4, vars_: int = 2, t: int = 1, e: int = 1) -> dict:
    coords = torch.stack(
        [torch.linspace(-1.0, 1.0, grid), torch.linspace(0.0, 6.0, grid)],
        dim=-1,
    )
    return {
        "data": torch.arange(t * e * grid * vars_, dtype=torch.float32).reshape(t, e, grid, vars_),
        "coordinates": coords,
        "metadata": {},
    }


# ---------------------------------------------- ObservationDataReader.get_sample


def test_get_sample_returns_unified_contract() -> None:
    n, v = 6, 3
    payload = _make_obs_payload(n=n, v=v, n_times=2)

    reader = _make_obs_reader(payload)
    sample = reader.get_sample(slice(0, 2))
    reader.data.__getitem__.assert_called_once_with(slice(0, 2))

    assert {"data", "layout", "coordinates", "timedeltas", "metadata"}.issubset(sample)
    # Each sample has one ensemble member and no explicit time axis.
    assert sample["data"].shape == (1, n, v)
    assert sample["layout"] == TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)
    assert sample["data"].dtype == torch.float32
    np.testing.assert_allclose(sample["data"][0].numpy(), payload.data)

    # Coordinates: single (N, 2) tensor stacking lat/lon (in radians).
    assert sample["coordinates"].shape == (n, 2)
    np.testing.assert_allclose(
        sample["coordinates"][:, 0].numpy(),
        np.deg2rad(payload.latitudes),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample["coordinates"][:, 1].numpy(),
        np.deg2rad(payload.longitudes),
        atol=1e-6,
    )

    # Timedeltas live at the top level, separate from coordinates.
    assert sample["timedeltas"].shape == (n,)
    torch.testing.assert_close(sample["timedeltas"], torch.tensor(payload.timedeltas, dtype=torch.float32))

    # The reader retains both time windows and reports their shard sizes.
    assert sample["metadata"][BOUNDARIES_META_KEY] == list(payload.boundaries)
    assert sample["shard_sizes"] == [[3], [3]]
    assert all(isinstance(s, slice) for s in sample["metadata"][BOUNDARIES_META_KEY])


def test_observation_reader_is_not_static_grid() -> None:
    reader = ObservationDataReader.__new__(ObservationDataReader)
    assert reader.is_static_grid is False


# ----------------------------------------------- Batch.collate (mixed batch)


def test_collate_mixed_gridded_and_sparse_batch() -> None:
    grid = 4
    vars_ = 2
    samples = [
        {
            "grid": _make_grid_sample(grid=grid, vars_=vars_),
            "obs": _make_obs_sample(n=5, v=vars_, n_times=2),
        },
        {
            "grid": _make_grid_sample(grid=grid, vars_=vars_),
            # Different N per sample — exactly the case default_collate cannot handle.
            "obs": _make_obs_sample(n=7, v=vars_, n_times=2),
        },
    ]

    batch = Batch.collate(samples, static_coord_datasets=("grid",))

    # Gridded path: stacked along a new leading batch dim.
    assert isinstance(batch.data["grid"], torch.Tensor)
    assert batch.data["grid"].shape[0] == 2

    # Sparse path: list[Tensor] of length B with varying N_i.
    assert isinstance(batch.data["obs"], list)
    assert len(batch.data["obs"]) == 2
    assert batch.data["obs"][0].shape == (1, 5, vars_)
    assert batch.data["obs"][1].shape == (1, 7, vars_)

    # Sparse coordinates are list[(N_i, 2)] tensors per sample.
    assert isinstance(batch.coordinates["obs"], list)
    assert len(batch.coordinates["obs"]) == 2
    assert batch.coordinates["obs"][0].shape == (5, 2)
    assert batch.coordinates["obs"][1].shape == (7, 2)

    # Sparse timedeltas are list[(N_i,)] tensors per sample, stored separately
    # from coordinates.
    assert isinstance(batch.timedeltas["obs"], list)
    assert len(batch.timedeltas["obs"]) == 2
    assert batch.timedeltas["obs"][0].shape == (5,)
    assert batch.timedeltas["obs"][1].shape == (7,)

    # Static-grid coordinates reused by reference (single tensor, no batch dim).
    assert batch.coordinates["grid"].shape == (grid, 2)
    assert batch.metadata[STATIC_COORDS_META_KEY] == frozenset({"grid"})

    # Boundaries gathered into per-dataset metadata as list[tuple[slice, ...]].
    boundaries = batch.metadata["obs"][BOUNDARIES_META_KEY]
    assert isinstance(boundaries, list)
    assert len(boundaries) == 2
    for entry in boundaries:
        assert all(isinstance(s, slice) for s in entry)


def test_collate_rejects_sparse_dataset_in_static_set() -> None:
    samples = [{"obs": _make_obs_sample()}, {"obs": _make_obs_sample(n=6)}]
    with pytest.raises(ValueError, match="sparse"):
        Batch.collate(samples, static_coord_datasets=("obs",))


# ------------------------------------------------ Batch.to (mixed CPU round-trip)


def test_to_mixed_batch_moves_tensors_and_preserves_boundaries() -> None:
    grid = 4
    vars_ = 2
    samples = [
        {"grid": _make_grid_sample(grid=grid, vars_=vars_), "obs": _make_obs_sample(n=5, v=vars_)},
        {"grid": _make_grid_sample(grid=grid, vars_=vars_), "obs": _make_obs_sample(n=7, v=vars_)},
    ]
    batch = Batch.collate(samples, static_coord_datasets=("grid",))

    moved = batch.to("cpu", non_blocking=False)

    # Static-grid coordinates short-circuit: same Python object.
    assert moved.coordinates["grid"] is batch.coordinates["grid"]

    # Gridded data is a tensor on cpu.
    assert isinstance(moved.data["grid"], torch.Tensor)
    assert moved.data["grid"].device.type == "cpu"

    # Sparse data: list of cpu tensors, one per batch sample.
    assert isinstance(moved.data["obs"], list)
    assert len(moved.data["obs"]) == 2
    assert all(t.device.type == "cpu" for t in moved.data["obs"])

    # Sparse coordinates and timedeltas moved per-list-entry.
    assert isinstance(moved.coordinates["obs"], list)
    assert all(t.device.type == "cpu" for t in moved.coordinates["obs"])
    assert isinstance(moved.timedeltas["obs"], list)
    assert all(t.device.type == "cpu" for t in moved.timedeltas["obs"])

    # Boundaries are passed through unchanged (identity-preserved).
    assert moved.metadata["obs"][BOUNDARIES_META_KEY] is batch.metadata["obs"][BOUNDARIES_META_KEY]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
