# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from anemoi.models.data import SourceSample
from anemoi.models.data import TensorLayout
from anemoi.models.data.batch import Batch
from anemoi.models.data.sources.base import Source
from tests.batch_builders import build_batch


def _gridded_layout() -> TensorLayout:
    return TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


def _make_data_tensor(grid: int = 4, vars_: int = 2) -> torch.Tensor:
    # (time, ensemble, grid, vars) per-sample shape
    return torch.arange(1 * 1 * grid * vars_, dtype=torch.float32).reshape(1, 1, grid, vars_)


def _sample_layout() -> TensorLayout:
    """Per-sample gridded layout, before collate adds the batch axis."""
    return TensorLayout(time=0, ensemble=1, grid=2, variables=3)


def _gridded_payload(
    data: torch.Tensor,
    coordinates: torch.Tensor | None = None,
    *,
    static: bool = False,
) -> SourceSample:
    """A gridded sample in the reader contract."""
    return SourceSample(
        data=data,
        layout=_sample_layout(),
        variables=[f"v{i}" for i in range(data.shape[_sample_layout().variables])],
        coordinates=coordinates,
        coordinates_are_static=static,
    )


def _simple_batch(**kwargs) -> Batch:
    """A minimal single-dataset gridded batch."""
    kwargs.setdefault("data", {"a": torch.zeros(2, 1, 1, 4, 2)})
    kwargs.setdefault("layouts", {"a": _gridded_layout()})
    kwargs.setdefault("variables", {"a": ["x", "y"]})
    return build_batch(**kwargs)


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
    batch = build_batch(
        data=data,
        coordinates=coordinates,
        static_coords=frozenset({"a"}),
        layouts={"a": _gridded_layout()},
        variables={"a": ["a", "b"]},
    )

    assert batch.dataset_names == ("a",)
    assert "a" in batch
    assert len(batch) == 1
    assert batch.is_static_coords("a")
    assert batch.static_coord_datasets == frozenset({"a"})

    # Mapping behaviour: batch[name] returns a rich per-dataset Source.
    view = batch["a"]
    assert isinstance(view, Source)
    assert view.data is data["a"]
    assert list(batch.keys()) == ["a"]
    assert next(iter(batch.values())).data is data["a"]
    assert dict(batch.items())["a"].data is data["a"]

    assert view.name == "a"
    assert view.coordinates is coordinates["a"]
    assert view.coordinates_are_static is True


def test_batch_missing_dataset_raises_keyerror() -> None:
    batch = _simple_batch()
    with pytest.raises(KeyError, match="missing"):
        _ = batch["missing"]


def test_batch_is_immutable() -> None:
    batch = _simple_batch()
    with pytest.raises(FrozenInstanceError):
        batch.sources = {}  # type: ignore[misc]


# --------------------------------------------------------------------- collate


def test_collate_stacks_data_along_batch_dim() -> None:
    samples = [
        {"a": _gridded_payload(_make_data_tensor(), _make_coordinates())},
        {"a": _gridded_payload(_make_data_tensor() + 100, _make_coordinates())},
    ]
    batch = Batch.collate(samples)

    assert batch["a"].data.shape == (2, 1, 1, 4, 2)
    assert torch.equal(batch["a"].data[0], samples[0]["a"].data)
    assert torch.equal(batch["a"].data[1], samples[1]["a"].data)


def test_collate_static_coords_share_reference() -> None:
    """The performance-critical invariant: static coords are NOT stacked or copied."""
    coords_ref = _make_coordinates()
    samples = [
        {"a": _gridded_payload(_make_data_tensor(), coords_ref, static=True)},
        {"a": _gridded_payload(_make_data_tensor() + 1, coords_ref, static=True)},
    ]
    batch = Batch.collate(samples)

    # Same object identity as the first sample's coordinates tensor.
    assert batch["a"].coordinates is coords_ref

    # Shape unchanged: no leading batch dimension was added.
    assert batch["a"].coordinates.shape == coords_ref.shape

    # Metadata records the static set.
    assert batch.static_coord_datasets == frozenset({"a"})


def test_collate_dynamic_coords_are_stacked() -> None:
    samples = [
        {"a": _gridded_payload(_make_data_tensor(), _make_coordinates())},
        {"a": _gridded_payload(_make_data_tensor() + 1, _make_coordinates())},
    ]
    batch = Batch.collate(samples)

    # Dynamic path: a leading batch dimension is added.
    assert batch["a"].coordinates.shape == (2, 4, 2)
    assert batch.static_coord_datasets == frozenset()


def test_gridded_source_view_flatten_repeats_static_coordinates_over_batch_and_ensemble() -> None:
    coordinates = _make_coordinates(grid=4)
    batch = build_batch(
        data={"a": torch.zeros(2, 3, 2, 4, 1)},
        coordinates={"a": coordinates},
        static_coords=frozenset({"a"}),
        layouts={"a": _gridded_layout()},
        variables={"a": ["x"]},
        statistics={"a": {}},
    )

    flat = batch["a"].flatten()

    expected_coordinates = coordinates.unsqueeze(0).unsqueeze(0).expand(2, 2, 4, 2).reshape(16, 2)
    assert flat.data.shape == (16, 3)
    torch.testing.assert_close(flat.coordinates, expected_coordinates)


def test_gridded_source_view_flatten_repeats_dynamic_coordinates_over_ensemble() -> None:
    coordinates = torch.stack([_make_coordinates(grid=4), _make_coordinates(grid=4) + 10.0], dim=0)
    batch = build_batch(
        data={"a": torch.zeros(2, 3, 2, 4, 1)},
        coordinates={"a": coordinates},
        layouts={"a": _gridded_layout()},
        variables={"a": ["x"]},
        statistics={"a": {}},
    )

    flat = batch["a"].flatten()

    expected_coordinates = coordinates.unsqueeze(1).expand(2, 2, 4, 2).reshape(16, 2)
    assert flat.data.shape == (16, 3)
    torch.testing.assert_close(flat.coordinates, expected_coordinates)


def test_collate_empty_samples_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        Batch.collate([])


def test_collate_supports_multiple_datasets() -> None:
    samples = [
        {
            "a": _gridded_payload(_make_data_tensor(grid=4), _make_coordinates(grid=4), static=True),
            "b": _gridded_payload(_make_data_tensor(grid=2), _make_coordinates(grid=2)),
        },
        {
            "a": _gridded_payload(_make_data_tensor(grid=4) + 1, _make_coordinates(grid=4), static=True),
            "b": _gridded_payload(_make_data_tensor(grid=2) + 1, _make_coordinates(grid=2)),
        },
    ]
    batch = Batch.collate(samples)
    assert batch.dataset_names == ("a", "b")
    # "a" is static -> shared reference
    assert batch["a"].coordinates is samples[0]["a"].coordinates
    # "b" is dynamic -> stacked
    assert batch["b"].coordinates.shape == (2, 2, 2)


# --------------------------------------------------------------- device / pin


def test_to_skips_static_coordinates() -> None:
    coords_ref = _make_coordinates()
    batch = _simple_batch(
        coordinates={"a": coords_ref},
        static_coords=frozenset({"a"}),
    )

    moved = batch.to("cpu")  # CPU-to-CPU, but identity tells us whether transfer was attempted

    # Data is always moved (even CPU-to-CPU may produce a new tensor object via .to()).
    assert moved["a"].data.device.type == "cpu"
    # Static coords are passed by reference, untouched.
    assert moved["a"].coordinates is coords_ref


def test_to_moves_dynamic_coordinates() -> None:
    coords = _make_coordinates()
    batch = _simple_batch(coordinates={"a": coords})

    moved = batch.to("cpu")

    # Dynamic coords are routed through ``.to()`` (no static-coord
    # short-circuit). Identity is not asserted: ``Tensor.to`` is a no-op
    # when the tensor already lives on the requested device, so the same
    # object may be returned.
    assert moved["a"].coordinates.device.type == "cpu"
    assert torch.equal(moved["a"].coordinates, coords)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_skips_static_coordinates() -> None:
    coords_ref = _make_coordinates()
    batch = _simple_batch(
        coordinates={"a": coords_ref},
        static_coords=frozenset({"a"}),
    )
    pinned = batch.pin_memory()
    # Static coords untouched.
    assert pinned["a"].coordinates is coords_ref


# -------------------------------------------------------------------- with_data


def test_with_data_replaces_data_and_shares_envelope() -> None:
    coords_ref = _make_coordinates()
    batch = _simple_batch(
        coordinates={"a": coords_ref},
        static_coords=frozenset({"a"}),
    )

    new_tensor = torch.ones(2, 1, 1, 4, 2)
    new_batch = batch.with_data({"a": new_tensor})

    # Data is replaced.
    assert new_batch["a"].data is new_tensor
    assert torch.equal(new_batch["a"].data, torch.ones(2, 1, 1, 4, 2))

    # Coordinates are shared by reference (no copy). ``Batch.coordinates`` is a
    # derived view now, so the invariant is per-tensor identity, not dict identity.
    assert new_batch["a"].coordinates is coords_ref

    # Static-coord membership preserved through the envelope.
    assert new_batch.is_static_coords("a")

    # Result is a new instance; receiver is not mutated (frozen dataclass).
    assert new_batch is not batch
    assert batch["a"].data.sum().item() == 0.0


def test_with_data_can_subset_datasets_and_envelope() -> None:
    coords_a = _make_coordinates()
    batch = build_batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2), "b": torch.zeros(2, 1, 1, 4, 2)},
        coordinates={"a": coords_a},
        static_coords=frozenset({"a"}),
        layouts={"a": _gridded_layout(), "b": _gridded_layout()},
        variables={"a": ["x", "y"], "b": ["x", "y"]},
        statistics={"a": {}},
    )

    new_a = torch.ones(2, 1, 1, 4, 2)
    subset = batch.with_data({"a": new_a})

    assert subset.dataset_names == ("a",)
    assert subset["a"].data is new_a
    assert subset["a"].coordinates is coords_a
    assert subset.static_coord_datasets == frozenset({"a"})

    with pytest.raises(ValueError, match="unknown dataset names"):
        batch.with_data({"a": torch.zeros(2, 1, 1, 4, 2), "c": torch.zeros(2, 1, 1, 4, 2)})


def test_with_data_supports_multiple_datasets() -> None:
    coords_a = _make_coordinates()
    coords_b = _make_coordinates(grid=8)
    batch = build_batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2), "b": torch.zeros(2, 1, 1, 8, 2)},
        coordinates={"a": coords_a, "b": coords_b},
        layouts={"a": _gridded_layout(), "b": _gridded_layout()},
        variables={"a": ["x", "y"], "b": ["x", "y"]},
    )

    new_a = torch.ones(2, 1, 1, 4, 2)
    new_b = torch.full((2, 1, 1, 8, 2), 2.0)
    new_batch = batch.with_data({"a": new_a, "b": new_b})

    assert new_batch["a"].data is new_a
    assert new_batch["b"].data is new_b
    # Per-dataset coordinates identity preserved.
    assert new_batch["a"].coordinates is coords_a
    assert new_batch["b"].coordinates is coords_b


def test_source_view_apply_func_uses_processor_and_preserves_envelope() -> None:
    coords_ref = _make_coordinates()
    layout = _gridded_layout()
    batch = build_batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coordinates={"a": coords_ref},
        static_coords=frozenset({"a"}),
        layouts={"a": layout},
        variables={"a": ["a", "b"]},
    )

    seen_layouts = []

    def processor(tensor: torch.Tensor, *, layout: TensorLayout, **_kwargs) -> torch.Tensor:
        seen_layouts.append(layout)
        return tensor + 1

    result = batch["a"].apply_func(processor, layout=layout)

    assert torch.equal(result.data, torch.ones(2, 1, 1, 4, 2))
    assert seen_layouts == [layout]
    assert result.coordinates is coords_ref


def test_source_view_apply_func_handles_sparse_list_payloads() -> None:
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    batch = build_batch(
        data={"obs": [torch.zeros(5, 3), torch.ones(7, 3)]},
        layouts={"obs": layout},
        variables={"obs": ["a", "b", "c"]},
        statistics={"obs": {}},
    )

    seen_shapes = []

    def processor(tensor: torch.Tensor, *, layout: TensorLayout, **_kwargs) -> torch.Tensor:
        seen_shapes.append(tuple(tensor.shape))
        assert layout.time_in_grid
        return tensor + 2

    result = batch["obs"].apply_func(processor, layout=layout)

    assert isinstance(result.data, list)
    assert seen_shapes == [(5, 3), (7, 3)]
    assert torch.equal(result.data[0], torch.full((5, 3), 2.0))
    assert torch.equal(result.data[1], torch.full((7, 3), 3.0))


# ---------------------------------------------------------- source coordinates


def test_source_view_returns_stacked_latlon_coordinates() -> None:
    coords = _make_coordinates()
    batch = build_batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        coordinates={"a": coords},
        layouts={"a": _gridded_layout()},
        variables={"a": ["a", "b"]},
    )

    out = batch["a"].coordinates

    assert out is not None
    assert out.shape == (4, 2)
    assert out is coords


def test_source_view_coordinates_none_when_missing() -> None:
    batch = build_batch(
        data={"a": torch.zeros(2, 1, 1, 4, 2)},
        layouts={"a": _gridded_layout()},
        variables={"a": ["a", "b"]},
    )

    assert batch["a"].coordinates is None


def test_source_view_returns_sparse_coordinate_lists() -> None:
    """Sparse datasets store ``coordinates`` as ``list[Tensor]``."""
    batch = build_batch(
        data={"a": [torch.zeros(4, 2), torch.zeros(6, 2)]},
        coordinates={"a": [torch.zeros(4, 2), torch.zeros(6, 2)]},
        layouts={"a": TensorLayout(grid=0, variables=1, time_in_grid=True)},
        variables={"a": ["a", "b"]},
    )

    assert isinstance(batch["a"].coordinates, list)


# ----------------------------------------------------- TensorLayout helpers


def test_tensor_layout_with_batch_dim_shifts_positive_axes() -> None:
    from anemoi.models.data.batch import TensorLayout

    layout = TensorLayout(time=0, ensemble=1, grid=2, variables=3)
    shifted = layout.with_batch_dim()
    assert shifted.batch == 0
    assert shifted.time == 1
    assert shifted.ensemble == 2
    assert shifted.grid == 3
    assert shifted.variables == 4


def test_tensor_layout_without_batch_dim_is_inverse() -> None:
    from anemoi.models.data.batch import TensorLayout

    layout = TensorLayout(time=0, ensemble=1, grid=2, variables=3, time_in_grid=False)
    roundtrip = layout.with_batch_dim().without_batch_dim()
    assert roundtrip == layout


def test_tensor_layout_without_batch_dim_sparse_roundtrip() -> None:
    from anemoi.models.data.batch import TensorLayout

    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    roundtrip = layout.with_batch_dim().without_batch_dim()
    assert roundtrip == layout


def test_tensor_layout_without_batch_dim_noop_when_already_unset() -> None:
    from anemoi.models.data.batch import TensorLayout

    layout = TensorLayout(ensemble=0, grid=1, variables=2)
    assert layout.without_batch_dim() is layout


def test_tensor_layout_repr_elides_none_fields() -> None:
    from anemoi.models.data.batch import TensorLayout

    r = repr(TensorLayout(grid=0, variables=1, time_in_grid=True))
    assert "grid=0" in r
    assert "variables=1" in r
    assert "time_in_grid=True" in r
    assert "ensemble=" not in r
    assert "batch=" not in r
    assert "time=" not in r


def test_batch_repr_summarises_per_dataset() -> None:
    from anemoi.models.data.batch import TensorLayout

    batch = build_batch(
        data={"grid": torch.zeros(2, 1, 1, 4, 3), "obs": [torch.zeros(5, 3), torch.zeros(7, 3)]},
        layouts={
            "grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
            "obs": TensorLayout(grid=0, variables=1, time_in_grid=True),
        },
        variables={"grid": ["x", "y", "z"], "obs": ["x", "y", "z"]},
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
    from anemoi.models.data.batch import TensorLayout

    samples = [
        {
            "a": SourceSample(
                data=torch.zeros(2, 3),
                layout=TensorLayout(grid=0, variables=5),
                variables=["x", "y", "z"],
            ),
        },
    ]
    with pytest.raises(ValueError, match="TensorLayout"):
        Batch.collate(samples)


def test_batch_collate_keeps_sparse_layout_unshifted() -> None:
    """For sparse list-payload datasets the collated layout must NOT be batch-shifted.

    Sparse datasets are stored as ``list[Tensor]`` of length B (each entry
    keeps its per-sample shape ``(N_i, V)``), so the batch dim is the list
    itself — no new tensor axis is added.
    """
    from anemoi.models.data.batch import TensorLayout

    sample_layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    # Sparse samples are recognised by ``layout.time_in_grid``; that is also what
    # makes ``boundaries`` mandatory, since they carry the time axis.
    samples = [
        {
            "obs": SourceSample(
                data=torch.zeros(5, 3),
                layout=sample_layout,
                variables=["x", "y", "z"],
                boundaries=(slice(0, 5),),
            ),
        },
        {
            "obs": SourceSample(
                data=torch.zeros(7, 3),
                layout=sample_layout,
                variables=["x", "y", "z"],
                boundaries=(slice(0, 7),),
            ),
        },
    ]
    batch = Batch.collate(samples)
    assert isinstance(batch["obs"].data, list)
    assert batch["obs"].layout == sample_layout
    assert batch["obs"].layout.batch is None


def test_batch_collate_shifts_gridded_layout_with_batch_dim() -> None:
    """Gridded (stacked) datasets get their layout shifted by ``with_batch_dim``."""
    from anemoi.models.data.batch import TensorLayout

    sample_layout = TensorLayout(time=0, ensemble=1, grid=2, variables=3)
    samples = [
        {"grid": SourceSample(data=torch.zeros(1, 1, 4, 3), layout=sample_layout, variables=["x", "y", "z"])},
        {"grid": SourceSample(data=torch.zeros(1, 1, 4, 3), layout=sample_layout, variables=["x", "y", "z"])},
    ]
    batch = Batch.collate(samples)
    assert isinstance(batch["grid"].data, torch.Tensor)
    assert batch["grid"].data.shape == (2, 1, 1, 4, 3)
    assert batch["grid"].layout == sample_layout.with_batch_dim()
    assert batch["grid"].layout.batch == 0


def test_batch_select_time_updates_sparse_envelope() -> None:
    """Selecting sparse time slices must also update coords, timedeltas and boundaries."""
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)

    data = [
        torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2),
        torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2),
    ]
    coordinates = [
        torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2),
        torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2),
    ]
    timedeltas = [
        torch.arange(5, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
    ]
    boundaries = [
        (slice(0, 2), slice(2, 5)),
        (slice(0, 1), slice(1, 6)),
    ]

    batch = build_batch(
        data={"obs": data},
        coordinates={"obs": coordinates},
        timedeltas={"obs": timedeltas},
        boundaries={"obs": boundaries},
        layouts={"obs": layout},
        variables={"obs": ["a", "b"]},
    )

    selected = batch.select(time={"obs": [1]})

    assert isinstance(selected["obs"].data, list)
    assert selected["obs"].data[0].shape == (3, 2)
    assert selected["obs"].data[1].shape == (5, 2)

    assert isinstance(selected["obs"].coordinates, list)
    assert selected["obs"].coordinates[0].shape == (3, 2)
    assert selected["obs"].coordinates[1].shape == (5, 2)

    assert isinstance(selected["obs"].timedeltas, list)
    assert selected["obs"].timedeltas[0].shape == (3,)
    assert selected["obs"].timedeltas[1].shape == (5,)

    assert selected["obs"].boundaries == [
        (slice(0, 3),),
        (slice(0, 5),),
    ]
