# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import SourceSample
from anemoi.models.data import SourceSpec
from anemoi.models.data import TensorLayout
from anemoi.models.data.views import GriddedSourceView
from anemoi.models.data.views import TabularSourceView
from anemoi.models.data.views import create_source_view
from batch_builders import make_batch
from batch_builders import make_source

GRIDDED_LAYOUT = TensorLayout(time=0, ensemble=1, grid=2, variables=3)
TABULAR_LAYOUT = TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)


def gridded_payload(variables: list[str] = ["a", "b", "c"]) -> SourceSample:
    n_vars = len(variables)
    return SourceSample(
        data=torch.arange(2 * 1 * 4 * n_vars, dtype=torch.float32).reshape(2, 1, 4, n_vars),
        variables=variables,
        statistics={"mean": torch.arange(n_vars, dtype=torch.float32)},
        layout=GRIDDED_LAYOUT,
        coordinates=torch.zeros(4, 2),
        grid_size=4,
        coordinates_are_static=True,
    )


def tabular_payload(n_points: int = 4) -> SourceSample:
    return SourceSample(
        data=torch.ones(1, n_points, 2),
        variables=["t2m", "sp"],
        statistics={"mean": torch.tensor([1.0, 2.0])},
        layout=TABULAR_LAYOUT,
        coordinates=torch.zeros(n_points, 2),
        timedeltas=torch.zeros(n_points),
        boundaries=(slice(0, n_points // 2), slice(n_points // 2, n_points)),
    )


def gridded_batch(variables: list[str] = ["a", "b", "c"]) -> Batch:
    n_vars = len(variables)
    return make_batch(
        data={"grid": torch.arange(2 * 2 * 1 * 4 * n_vars, dtype=torch.float32).reshape(2, 2, 1, 4, n_vars)},
        coordinates={"grid": torch.zeros(4, 2)},
        layouts={"grid": GRIDDED_LAYOUT.with_batch_dim()},
        variables={"grid": variables},
        statistics={"grid": {"mean": torch.arange(n_vars, dtype=torch.float32)}},
        grid_sizes={"grid": 4},
        static_coords=("grid",),
    )


class TestSourceSpec:
    def test_rejects_duplicate_variable_names(self) -> None:
        with pytest.raises(ValueError, match="unique variable names"):
            SourceSpec(name="grid", variables=["a", "a"], layout=GRIDDED_LAYOUT)

    def test_name_to_index_is_memoised(self) -> None:
        spec = SourceSpec(name="grid", variables=["a", "b"], layout=GRIDDED_LAYOUT)
        assert spec.name_to_index == {"a": 0, "b": 1}
        assert spec.name_to_index is spec.name_to_index

    def test_select_variables_indexes_names_and_statistics_together(self) -> None:
        spec = SourceSpec(
            name="grid",
            variables=["a", "b", "c"],
            layout=GRIDDED_LAYOUT,
            statistics={"mean": torch.tensor([10.0, 20.0, 30.0])},
        )
        selected = spec.select_variables([0, 2])
        assert selected.variables == ["a", "c"]
        assert selected.statistics["mean"].tolist() == [10.0, 30.0]
        # the receiver is not mutated
        assert spec.variables == ["a", "b", "c"]

    def test_select_variables_accepts_a_slice(self) -> None:
        spec = SourceSpec(name="grid", variables=["a", "b", "c"], layout=GRIDDED_LAYOUT)
        assert spec.select_variables(slice(1, 3)).variables == ["b", "c"]

    @pytest.mark.parametrize(
        ("layout", "expected_type"),
        [(GRIDDED_LAYOUT, GriddedSourceView), (TABULAR_LAYOUT, TabularSourceView)],
    )
    def test_empty_builds_a_dataless_source_of_the_right_kind(self, layout, expected_type) -> None:
        spec = SourceSpec(name="src", variables=["a", "b"], layout=layout)
        empty = spec.empty(batch_size=3)

        assert isinstance(empty, expected_type)
        assert empty.spec is spec

        samples = empty.data if isinstance(empty.data, list) else [empty.data]
        if layout.time_in_grid:
            assert len(samples) == 3
        for sample in samples:
            # full variable axis, zero-length grid axis
            normalized = layout.normalized(sample.ndim)
            assert sample.shape[normalized.variables] == 2
            assert sample.shape[normalized.grid] == 0


class TestSpecOnViews:
    def test_spec_fields_read_through_the_view(self) -> None:
        view = gridded_batch()["grid"]
        assert view.name == "grid"
        assert view.variables == ["a", "b", "c"]
        assert view.layout == GRIDDED_LAYOUT.with_batch_dim()
        assert view.grid_size == 4
        assert view.coordinates_are_static is True
        assert view.name_to_index == {"a": 0, "b": 1, "c": 2}

    def test_repeated_access_reuses_one_spec(self) -> None:
        batch = gridded_batch()
        assert batch["grid"].spec is batch["grid"].spec
        assert batch["grid"].name_to_index is batch["grid"].name_to_index

    def test_batch_spec_covers_every_dataset(self) -> None:
        batch = Batch.collate([{"grid": gridded_payload(), "obs": tabular_payload()}])  # noqa: E501
        assert set(batch.spec) == {"grid", "obs"}
        assert batch.spec["obs"].layout.time_in_grid is True
        assert batch.spec["obs"].grid_size is None

    def test_create_source_view_dispatches_on_the_spec_layout(self) -> None:
        spec = SourceSpec(name="grid", variables=["a", "b"], layout=GRIDDED_LAYOUT, coordinates_are_static=True)
        view = create_source_view(spec, data=torch.zeros(1, 1, 4, 2), coordinates=torch.zeros(4, 2))
        assert isinstance(view, GriddedSourceView)
        assert view.spec is spec
        assert view.coordinates_are_static is True

    def test_clone_replaces_the_spec_wholesale(self) -> None:
        view = gridded_batch()["grid"]
        cloned = view.clone(spec=view.spec.clone(variables=["x", "y", "z"]))
        assert cloned.variables == ["x", "y", "z"]
        assert view.variables == ["a", "b", "c"]
        # payload is shared by reference when it is not replaced
        assert cloned.data is view.data

    def test_select_variables_keeps_data_and_spec_consistent(self) -> None:
        view = gridded_batch()["grid"]
        selected = view.select(variables=[0, 2])
        assert selected.variables == ["a", "c"]
        assert selected.statistics["mean"].tolist() == [0.0, 2.0]
        assert selected.data.shape[selected.layout.variables] == 2


class TestBatchTransformations:
    """Transformations must carry the spec through, and never mutate the receiver."""

    def test_replace_swaps_one_source(self) -> None:
        batch = gridded_batch()
        renamed = batch.replace("grid", batch["grid"].clone(spec=batch["grid"].spec.clone(variables=["x", "y", "z"])))
        assert renamed["grid"].variables == ["x", "y", "z"]
        assert batch["grid"].variables == ["a", "b", "c"]

    def test_select_narrows_variables_and_their_lookup(self) -> None:
        batch = gridded_batch()
        selected = batch.select(variables=[1])
        assert selected["grid"].variables == ["b"]
        assert selected["grid"].name_to_index == {"b": 0}
        assert batch["grid"].variables == ["a", "b", "c"]

    def test_device_transfer_preserves_the_spec(self) -> None:
        batch = gridded_batch()
        moved = batch.to("cpu")
        assert moved["grid"].spec is batch["grid"].spec

    def test_apply_pairwise_runs_per_dataset(self) -> None:
        batch = gridded_batch()
        other = gridded_batch()
        out = batch.apply_pairwise(other, lambda a, b, **_: (a - b).abs().sum())
        assert set(out) == {"grid"}
        assert out["grid"].item() == 0.0
