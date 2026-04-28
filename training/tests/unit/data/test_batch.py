# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import pytest
import torch

from anemoi.training.data.batch import STATIC_COORDS_META_KEY
from anemoi.training.data.batch import Batch
from anemoi.training.data.batch import DatasetView


def _make_data_tensor(grid: int = 4, vars_: int = 2) -> torch.Tensor:
    # (time, ensemble, grid, vars) per-sample shape
    return torch.arange(1 * 1 * grid * vars_, dtype=torch.float32).reshape(1, 1, grid, vars_)


def _make_coords(grid: int = 4) -> dict[str, torch.Tensor]:
    return {
        "latitudes": torch.linspace(-1.0, 1.0, grid),
        "longitudes": torch.linspace(0.0, 6.0, grid),
    }


# ---------------------------------------------------------------- construction


def test_batch_basic_construction_and_access() -> None:
    data = {"a": torch.zeros(2, 1, 1, 4, 2)}
    coords = {"a": _make_coords()}
    batch = Batch(data=data, coords=coords, metadata={STATIC_COORDS_META_KEY: frozenset({"a"})})

    assert batch.dataset_names == ("a",)
    assert "a" in batch
    assert len(batch) == 1
    assert batch.is_static_coords("a")
    assert batch.static_coord_datasets == frozenset({"a"})

    # Mapping behaviour: batch[name] returns the data tensor.
    assert batch["a"] is data["a"]
    assert list(batch.keys()) == ["a"]
    assert next(iter(batch.values())) is data["a"]
    assert dict(batch.items()) == data

    # Rich per-dataset view available via .view().
    view = batch.view("a")
    assert isinstance(view, DatasetView)
    assert view.name == "a"
    assert view.data is data["a"]
    assert view.coords is coords["a"]
    assert view.is_static is True


def test_batch_missing_dataset_raises_keyerror() -> None:
    batch = Batch(data={"a": torch.zeros(1)})
    with pytest.raises(KeyError, match="missing"):
        _ = batch["missing"]


def test_batch_is_immutable() -> None:
    batch = Batch(data={"a": torch.zeros(1)})
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        batch.data = {}  # type: ignore[misc]


# --------------------------------------------------------------------- collate


def test_collate_stacks_data_along_batch_dim() -> None:
    samples = [
        {"a": {"data": _make_data_tensor(), "coords": _make_coords()}},
        {"a": {"data": _make_data_tensor() + 100, "coords": _make_coords()}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=["a"])

    assert batch.data["a"].shape == (2, 1, 1, 4, 2)
    assert torch.equal(batch.data["a"][0], samples[0]["a"]["data"])
    assert torch.equal(batch.data["a"][1], samples[1]["a"]["data"])


def test_collate_static_coords_share_reference() -> None:
    """The performance-critical invariant: static coords are NOT stacked or copied."""
    coords_ref = _make_coords()
    samples = [
        {"a": {"data": _make_data_tensor(), "coords": coords_ref}},
        {"a": {"data": _make_data_tensor() + 1, "coords": coords_ref}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=["a"])

    # Same object identity as the first sample's coords dict and tensors.
    assert batch.coords["a"] is samples[0]["a"]["coords"]
    assert batch.coords["a"]["latitudes"] is coords_ref["latitudes"]
    assert batch.coords["a"]["longitudes"] is coords_ref["longitudes"]

    # Shape unchanged: no leading batch dimension was added.
    assert batch.coords["a"]["latitudes"].shape == coords_ref["latitudes"].shape

    # Metadata records the static set.
    assert batch.static_coord_datasets == frozenset({"a"})


def test_collate_dynamic_coords_are_stacked() -> None:
    samples = [
        {"a": {"data": _make_data_tensor(), "coords": _make_coords()}},
        {"a": {"data": _make_data_tensor() + 1, "coords": _make_coords()}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=())

    # Dynamic path: a leading batch dimension is added.
    assert batch.coords["a"]["latitudes"].shape == (2, 4)
    assert batch.coords["a"]["longitudes"].shape == (2, 4)
    assert batch.static_coord_datasets == frozenset()


def test_collate_empty_samples_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        Batch.collate([])


def test_collate_supports_multiple_datasets() -> None:
    samples = [
        {
            "a": {"data": _make_data_tensor(grid=4), "coords": _make_coords(grid=4)},
            "b": {"data": _make_data_tensor(grid=2), "coords": _make_coords(grid=2)},
        },
        {
            "a": {"data": _make_data_tensor(grid=4) + 1, "coords": _make_coords(grid=4)},
            "b": {"data": _make_data_tensor(grid=2) + 1, "coords": _make_coords(grid=2)},
        },
    ]
    batch = Batch.collate(samples, static_coord_datasets=["a"])
    assert batch.dataset_names == ("a", "b")
    # "a" is static -> shared reference
    assert batch.coords["a"]["latitudes"] is samples[0]["a"]["coords"]["latitudes"]
    # "b" is dynamic -> stacked
    assert batch.coords["b"]["latitudes"].shape == (2, 2)


# --------------------------------------------------------------- device / pin


def test_to_skips_static_coordinates() -> None:
    coords_ref = _make_coords()
    batch = Batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coords={"a": coords_ref},
        metadata={STATIC_COORDS_META_KEY: frozenset({"a"})},
    )

    moved = batch.to("cpu")  # CPU-to-CPU, but identity tells us whether transfer was attempted

    # Data is always moved (even CPU-to-CPU may produce a new tensor object via .to()).
    assert moved.data["a"].device.type == "cpu"
    # Static coords are passed by reference, untouched.
    assert moved.coords["a"] is coords_ref
    assert moved.coords["a"]["latitudes"] is coords_ref["latitudes"]


def test_to_moves_dynamic_coordinates() -> None:
    coords = _make_coords()
    batch = Batch(data={"a": torch.zeros(2, 1, 1, 4, 2)}, coords={"a": coords})

    moved = batch.to("cpu")

    # Dynamic coords go through .to(); the dict is rebuilt (not the same object).
    assert moved.coords["a"] is not coords
    assert torch.equal(moved.coords["a"]["latitudes"], coords["latitudes"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_skips_static_coordinates() -> None:
    coords_ref = _make_coords()
    batch = Batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coords={"a": coords_ref},
        metadata={STATIC_COORDS_META_KEY: frozenset({"a"})},
    )
    pinned = batch.pin_memory()
    # Static coords untouched.
    assert pinned.coords["a"] is coords_ref
