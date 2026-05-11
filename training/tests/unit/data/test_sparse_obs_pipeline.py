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

* :meth:`ObservationDataReader._unpack_sample` — single-round-trip unpack.
* :meth:`Batch.collate` on a mixed gridded + sparse batch.
* :meth:`Batch.to` on the same mixed batch (CPU-only round-trip; the
  test asserts behaviour, not GPU availability).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.models.data.batch import BOUNDARIES_META_KEY
from anemoi.models.data.batch import STATIC_COORDS_META_KEY
from anemoi.models.data.batch import Batch
from anemoi.training.data.data_reader import ObservationDataReader

# ----------------------------------------------------------------- test on some "real" obs data
# TODO: replace this with a mock anemoi.dataset object

from anemoi.datasets import open_dataset

_DATASET_NAME = "npp_atms"
_DATASET_CONFIG = {
    "dataset": "observations-ea-ofb-0001-2012-2023-npp-atms-radiances-v2-from-dop-try-2",
    "window": "(-6h, 0]",
    "frequency": "6h",
    "select": ["obsvalue_rawbt_1", "obsvalue_rawbt_7", "cos_vza", "cos_latitude", "cos_longitude"],
}

def _get_obs_sample(time_index: int | slice | list[int] = 0):
    ds = open_dataset(**_DATASET_CONFIG)
    return ds[time_index, ...]


def test_make_anemoi_reader(start: int = 2015, end: int = 2017) -> None:
    """Test that we can make an ObservationDataReader and unpack a sample."""
    reader = ObservationDataReader(dataset_config=_DATASET_CONFIG, start=start, end=end)

    # Get a sample from the dataset (duck-typed contract).
    payload = _get_obs_sample(slice(start, end))

    # Unpack the sample into the unified contract.
    sample = reader._unpack_sample(payload)

    assert set(sample) == {"data", "coordinates", "timedeltas", "metadata"}
    assert sample["data"].shape[-1] == 5  # 5 variables
    assert sample["coordinates"].shape[-1] == 2  # (N, 2) lat/lon


def test_batch_collate_and_to() -> None:
    """Test that we can collate a batch of two observation samples and move it to the GPU."""
    reader = ObservationDataReader(dataset_config=_DATASET_CONFIG, start=2015, end=2017)

    # Get two samples from the dataset (duck-typed contract).
    payload1 = _get_obs_sample(slice(20, 24))
    payload2 = _get_obs_sample(slice(40, 44))

    # ``Batch`` is a per-dataset envelope, so each sample must be wrapped
    # under its dataset name (here "npp_atms") before collation.
    sample1 = {_DATASET_NAME: reader._unpack_sample(payload1)}
    sample2 = {_DATASET_NAME: reader._unpack_sample(payload2)}

    # Collate the samples into a batch.
    batch = Batch.collate([sample1, sample2], static_coord_datasets=())

    # Move the batch to the GPU (if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved_batch = batch.to(device)

    # Assert that the data is on the correct device.
    assert moved_batch.data[_DATASET_NAME][0].device.type == device.type
    assert moved_batch.data[_DATASET_NAME][1].device.type == device.type

# ----------------------------------------------------------------- helpers


def _make_obs_payload(n: int = 5, v: int = 3, n_times: int = 2):
    """Return a fake ``self.data[time_indices, ...]`` payload.

    Mirrors the duck-typed object documented in READER.md: ``data``,
    ``latitudes``, ``longitudes``, ``timedeltas`` are ``np.ndarray`` and
    ``boundaries`` is a tuple of ``slice`` objects splitting ``N`` per time.
    """
    boundaries = tuple(slice(i * (n // n_times), (i + 1) * (n // n_times)) for i in range(n_times))
    return SimpleNamespace(
        data=np.arange(n * v, dtype=np.float32).reshape(n, v),
        latitudes=np.linspace(-90.0, 90.0, n, dtype=np.float64),
        longitudes=np.linspace(0.0, 360.0, n, endpoint=False, dtype=np.float64),
        timedeltas=np.linspace(0.0, 3600.0, n, dtype=np.float64),
        boundaries=boundaries,
    )


def _make_obs_sample(n: int = 5, v: int = 3, n_times: int = 2) -> dict:
    """Build a sparse sample matching the ObservationDataReader contract."""
    reader = ObservationDataReader.__new__(ObservationDataReader)
    return reader._unpack_sample(_make_obs_payload(n=n, v=v, n_times=n_times))


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


# ---------------------------------------------- ObservationDataReader.unpack


def test_unpack_sample_returns_unified_contract() -> None:
    n, v = 6, 3
    payload = _make_obs_payload(n=n, v=v, n_times=2)
    reader = ObservationDataReader.__new__(ObservationDataReader)

    sample = reader._unpack_sample(payload)

    assert set(sample) == {"data", "coordinates", "timedeltas", "metadata"}
    # Data: dummy ensemble dim, no time axis, original (N, V) preserved.
    assert sample["data"].shape == (1, n, v)
    assert sample["data"].dtype == torch.float32
    np.testing.assert_allclose(sample["data"][0].numpy(), payload.data)

    # Coordinates: single (N, 2) tensor stacking lat/lon (in radians).
    assert sample["coordinates"].shape == (n, 2)
    np.testing.assert_allclose(
        sample["coordinates"][:, 0].numpy(),
        np.deg2rad(payload.latitudes),
    )
    np.testing.assert_allclose(
        sample["coordinates"][:, 1].numpy(),
        np.deg2rad(payload.longitudes),
    )

    # Timedeltas live at the top level, separate from coordinates.
    assert sample["timedeltas"].shape == (n,)

    # Boundaries are passed through unchanged (Python slice tuple).
    assert sample["metadata"][BOUNDARIES_META_KEY] is payload.boundaries
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
    assert batch.data["obs"][0].shape[1] == 5
    assert batch.data["obs"][1].shape[1] == 7

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
