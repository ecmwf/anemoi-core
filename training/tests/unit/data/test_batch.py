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

from anemoi.training.data.batch import BOUNDARIES_META_KEY
from anemoi.training.data.batch import STATIC_COORDS_META_KEY
from anemoi.training.data.batch import Batch
from anemoi.training.data.batch import DatasetView
from anemoi.models.data.tensor_layout import TensorLayout


def _gridded_layout() -> TensorLayout:
    return TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


def _make_data_tensor(grid: int = 4, vars_: int = 2) -> torch.Tensor:
    # (time, ensemble, grid, vars) per-sample shape
    return torch.arange(1 * 1 * grid * vars_, dtype=torch.float32).reshape(1, 1, grid, vars_)


def _make_coordinates(grid: int = 4) -> torch.Tensor:
    """Return a stacked ``(N, 2)`` tensor of (latitudes, longitudes)."""
    return torch.stack(
        [torch.linspace(-1.0, 1.0, grid), torch.linspace(0.0, 6.0, grid)],
        dim=-1,
    )


# ---------------------------------------------------------------- construction


def test_batch_basic_construction_and_access() -> None:
    data = {"a": torch.zeros(2, 1, 1, 4, 2)}
    coordinates = {"a": _make_coordinates()}
    batch = Batch(
        data=data,
        coordinates=coordinates,
        metadata={STATIC_COORDS_META_KEY: frozenset({"a"})},
        layouts={"a": _gridded_layout()},
    )

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
    view = batch["a"]
    assert isinstance(view, DatasetView)
    assert view.name == "a"
    assert view.data is data["a"]
    assert view.coordinates is coordinates["a"]
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
        {"a": {"data": _make_data_tensor(), "coordinates": _make_coordinates()}},
        {"a": {"data": _make_data_tensor() + 100, "coordinates": _make_coordinates()}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=["a"])

    assert batch.data["a"].shape == (2, 1, 1, 4, 2)
    assert torch.equal(batch.data["a"][0], samples[0]["a"]["data"])
    assert torch.equal(batch.data["a"][1], samples[1]["a"]["data"])


def test_collate_static_coords_share_reference() -> None:
    """The performance-critical invariant: static coords are NOT stacked or copied."""
    coords_ref = _make_coordinates()
    samples = [
        {"a": {"data": _make_data_tensor(), "coordinates": coords_ref}},
        {"a": {"data": _make_data_tensor() + 1, "coordinates": coords_ref}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=["a"])

    # Same object identity as the first sample's coordinates tensor.
    assert batch.coordinates["a"] is coords_ref

    # Shape unchanged: no leading batch dimension was added.
    assert batch.coordinates["a"].shape == coords_ref.shape

    # Metadata records the static set.
    assert batch.static_coord_datasets == frozenset({"a"})


def test_collate_dynamic_coords_are_stacked() -> None:
    samples = [
        {"a": {"data": _make_data_tensor(), "coordinates": _make_coordinates()}},
        {"a": {"data": _make_data_tensor() + 1, "coordinates": _make_coordinates()}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=())

    # Dynamic path: a leading batch dimension is added.
    assert batch.coordinates["a"].shape == (2, 4, 2)
    assert batch.static_coord_datasets == frozenset()


def test_collate_empty_samples_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        Batch.collate([])


def test_collate_supports_multiple_datasets() -> None:
    samples = [
        {
            "a": {"data": _make_data_tensor(grid=4), "coordinates": _make_coordinates(grid=4)},
            "b": {"data": _make_data_tensor(grid=2), "coordinates": _make_coordinates(grid=2)},
        },
        {
            "a": {"data": _make_data_tensor(grid=4) + 1, "coordinates": _make_coordinates(grid=4)},
            "b": {"data": _make_data_tensor(grid=2) + 1, "coordinates": _make_coordinates(grid=2)},
        },
    ]
    batch = Batch.collate(samples, static_coord_datasets=["a"])
    assert batch.dataset_names == ("a", "b")
    # "a" is static -> shared reference
    assert batch.coordinates["a"] is samples[0]["a"]["coordinates"]
    # "b" is dynamic -> stacked
    assert batch.coordinates["b"].shape == (2, 2, 2)


# --------------------------------------------------------------- device / pin


def test_to_skips_static_coordinates() -> None:
    coords_ref = _make_coordinates()
    batch = Batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coordinates={"a": coords_ref},
        metadata={STATIC_COORDS_META_KEY: frozenset({"a"})},
    )

    moved = batch.to("cpu")  # CPU-to-CPU, but identity tells us whether transfer was attempted

    # Data is always moved (even CPU-to-CPU may produce a new tensor object via .to()).
    assert moved.data["a"].device.type == "cpu"
    # Static coords are passed by reference, untouched.
    assert moved.coordinates["a"] is coords_ref


def test_to_moves_dynamic_coordinates() -> None:
    coords = _make_coordinates()
    batch = Batch(data={"a": torch.zeros(2, 1, 1, 4, 2)}, coordinates={"a": coords})

    moved = batch.to("cpu")

    # Dynamic coords are routed through ``.to()`` (no static-coord
    # short-circuit). Identity is not asserted: ``Tensor.to`` is a no-op
    # when the tensor already lives on the requested device, so the same
    # object may be returned.
    assert moved.coordinates["a"].device.type == "cpu"
    assert torch.equal(moved.coordinates["a"], coords)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_skips_static_coordinates() -> None:
    coords_ref = _make_coordinates()
    batch = Batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coordinates={"a": coords_ref},
        metadata={STATIC_COORDS_META_KEY: frozenset({"a"})},
    )
    pinned = batch.pin_memory()
    # Static coords untouched.
    assert pinned.coordinates["a"] is coords_ref


# -------------------------------------------------------------------- with_data


def test_with_data_replaces_data_and_shares_envelope() -> None:
    coords_ref = _make_coordinates()
    metadata_ref = {STATIC_COORDS_META_KEY: frozenset({"a"})}
    batch = Batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coordinates={"a": coords_ref},
        metadata=metadata_ref,
    )

    new_tensor = torch.ones(2, 1, 1, 4, 2)
    new_batch = batch.with_data({"a": new_tensor})

    # Data is replaced.
    assert new_batch.data["a"] is new_tensor
    assert torch.equal(new_batch.data["a"], torch.ones(2, 1, 1, 4, 2))

    # Coordinates and metadata are shared by reference (no copy).
    assert new_batch.coordinates is batch.coordinates
    assert new_batch.coordinates["a"] is coords_ref
    assert new_batch.metadata is batch.metadata

    # Static-coord membership preserved through the envelope.
    assert new_batch.is_static_coords("a")

    # Result is a new instance; receiver is not mutated (frozen dataclass).
    assert new_batch is not batch
    assert batch.data["a"].sum().item() == 0.0


def test_with_data_rejects_mismatched_keys() -> None:
    batch = Batch(data={"a": torch.zeros(2, 1, 1, 4, 2), "b": torch.zeros(2, 1, 1, 4, 2)})

    with pytest.raises(ValueError, match="must match the existing dataset names"):
        batch.with_data({"a": torch.zeros(2, 1, 1, 4, 2)})

    with pytest.raises(ValueError, match="must match the existing dataset names"):
        batch.with_data({"a": torch.zeros(2, 1, 1, 4, 2), "c": torch.zeros(2, 1, 1, 4, 2)})


def test_with_data_supports_multiple_datasets() -> None:
    coords_a = _make_coordinates()
    coords_b = _make_coordinates(grid=8)
    batch = Batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2), "b": torch.zeros(2, 1, 1, 8, 2)},
        coordinates={"a": coords_a, "b": coords_b},
    )

    new_a = torch.ones(2, 1, 1, 4, 2)
    new_b = torch.full((2, 1, 1, 8, 2), 2.0)
    new_batch = batch.with_data({"a": new_a, "b": new_b})

    assert new_batch.data["a"] is new_a
    assert new_batch.data["b"] is new_b
    # Per-dataset coordinates identity preserved.
    assert new_batch.coordinates["a"] is coords_a
    assert new_batch.coordinates["b"] is coords_b


# ---------------------------------------------------------------- node_coords


def test_node_coords_returns_stacked_latlon() -> None:
    coords = _make_coordinates()
    batch = Batch(data={"a": torch.zeros(2, 1, 1, 4, 2)}, coordinates={"a": coords})

    out = batch.node_coords("a")

    assert out is not None
    assert out.shape == (4, 2)
    # node_coords is just the stored ``coordinates`` tensor (already stacked).
    assert out is coords


def test_node_coords_returns_none_when_missing() -> None:
    batch = Batch(data={"a": torch.zeros(2, 1, 1, 4, 2)})

    assert batch.node_coords("a") is None


def test_node_coords_returns_none_for_sparse_coordinates() -> None:
    """Sparse datasets store ``coordinates`` as ``list[Tensor]``; node_coords returns None."""
    batch = Batch(
        data={"a": [torch.zeros(1, 4, 2), torch.zeros(1, 6, 2)]},
        coordinates={"a": [torch.zeros(4, 2), torch.zeros(6, 2)]},
    )

    assert batch.node_coords("a") is None



# ----------------------------------------------------- TensorLayout helpers


def test_tensor_layout_with_batch_dim_shifts_positive_axes() -> None:
    from anemoi.training.data.batch import TensorLayout

    layout = TensorLayout(time=0, ensemble=1, grid=2, variables=3)
    shifted = layout.with_batch_dim()
    assert shifted.batch == 0
    assert shifted.time == 1
    assert shifted.ensemble == 2
    assert shifted.grid == 3
    assert shifted.variables == 4


def test_tensor_layout_without_batch_dim_is_inverse() -> None:
    from anemoi.training.data.batch import TensorLayout

    layout = TensorLayout(time=0, ensemble=1, grid=2, variables=3, time_in_grid=False)
    roundtrip = layout.with_batch_dim().without_batch_dim()
    assert roundtrip == layout


def test_tensor_layout_without_batch_dim_sparse_roundtrip() -> None:
    from anemoi.training.data.batch import TensorLayout

    layout = TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)
    roundtrip = layout.with_batch_dim().without_batch_dim()
    assert roundtrip == layout


def test_tensor_layout_without_batch_dim_noop_when_already_unset() -> None:
    from anemoi.training.data.batch import TensorLayout

    layout = TensorLayout(ensemble=0, grid=1, variables=2)
    assert layout.without_batch_dim() is layout


def test_tensor_layout_repr_elides_none_fields() -> None:
    from anemoi.training.data.batch import TensorLayout

    r = repr(TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True))
    assert "ensemble=0" in r
    assert "grid=1" in r
    assert "variables=2" in r
    assert "time_in_grid=True" in r
    assert "batch=" not in r
    assert "time=" not in r


def test_batch_repr_summarises_per_dataset() -> None:
    from anemoi.training.data.batch import TensorLayout

    batch = Batch(
        data={"grid": torch.zeros(2, 1, 1, 4, 3), "obs": [torch.zeros(1, 5, 3), torch.zeros(1, 7, 3)]},
        layouts={
            "grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
            "obs": TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True),
        },
    )
    out = repr(batch)
    assert "grid:" in out
    assert "(2, 1, 1, 4, 3)" in out
    assert "obs:" in out
    assert "list[2]" in out
    assert "TensorLayout" in out
    assert "batch=0" in out


def test_batch_collate_rejects_invalid_layout_position() -> None:
    """Layouts that point at a non-existent axis must be rejected by collate."""
    from anemoi.training.data.batch import TensorLayout

    samples = [
        {"a": {"data": torch.zeros(2, 3), "layout": TensorLayout(grid=0, variables=5)}},
    ]
    with pytest.raises(ValueError, match="TensorLayout"):
        Batch.collate(samples, static_coord_datasets=())


def test_batch_collate_keeps_sparse_layout_unshifted() -> None:
    """For sparse list-payload datasets the collated layout must NOT be batch-shifted.

    Sparse datasets are stored as ``list[Tensor]`` of length B (each entry
    keeps its per-sample shape ``(1, N_i, V)``), so the batch dim is the
    list itself — no new tensor axis is added.
    """
    from anemoi.training.data.batch import TensorLayout

    sample_layout = TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)
    # Sparse samples are tagged via the ``boundaries`` metadata key; without
    # it ``Batch.collate`` would treat them as gridded and try to stack
    # tensors of different ``N``.
    samples = [
        {
            "obs": {
                "data": torch.zeros(1, 5, 3),
                "layout": sample_layout,
                "metadata": {BOUNDARIES_META_KEY: (slice(0, 5),)},
            },
        },
        {
            "obs": {
                "data": torch.zeros(1, 7, 3),
                "layout": sample_layout,
                "metadata": {BOUNDARIES_META_KEY: (slice(0, 7),)},
            },
        },
    ]
    batch = Batch.collate(samples, static_coord_datasets=())
    assert isinstance(batch.data["obs"], list)
    assert batch.layouts["obs"] == sample_layout
    assert batch.layouts["obs"].batch is None


def test_batch_collate_shifts_gridded_layout_with_batch_dim() -> None:
    """Gridded (stacked) datasets get their layout shifted by ``with_batch_dim``."""
    from anemoi.training.data.batch import TensorLayout

    sample_layout = TensorLayout(time=0, ensemble=1, grid=2, variables=3)
    samples = [
        {"grid": {"data": torch.zeros(1, 1, 4, 3), "layout": sample_layout}},
        {"grid": {"data": torch.zeros(1, 1, 4, 3), "layout": sample_layout}},
    ]
    batch = Batch.collate(samples, static_coord_datasets=())
    assert isinstance(batch.data["grid"], torch.Tensor)
    assert batch.data["grid"].shape == (2, 1, 1, 4, 3)
    assert batch.layouts["grid"] == sample_layout.with_batch_dim()
    assert batch.layouts["grid"].batch == 0
