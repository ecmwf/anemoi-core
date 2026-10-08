# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
from dataclasses import replace

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import GriddedSourceSample
from anemoi.models.data import TabularSourceSample
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources.gridded import GriddedSource
from anemoi.models.data.sources.gridded import GriddedTemplate
from anemoi.models.data.sources.tabular import TabularSource
from anemoi.models.data.sources.tabular import TabularTemplate
from tests.batch_builders import build_batch

GRIDDED_LAYOUT = TensorLayout(time=0, ensemble=1, grid=2, variables=3)
TABULAR_LAYOUT = TensorLayout(ensemble=0, grid=1, variables=2)

# One statistics object per dataset, shared by its samples as a reader does
# (SourceSample.collate checks the statistics by identity).
TABULAR_STATISTICS = {"mean": torch.tensor([1.0, 2.0])}


@functools.cache
def gridded_statistics(n_vars: int) -> dict[str, torch.Tensor]:
    return {"mean": torch.arange(n_vars, dtype=torch.float32)}


def gridded_payload(variables: list[str] = ["a", "b", "c"]) -> GriddedSourceSample:
    n_vars = len(variables)
    return GriddedSourceSample(
        data=torch.arange(2 * 1 * 4 * n_vars, dtype=torch.float32).reshape(2, 1, 4, n_vars),
        variables=variables,
        statistics=gridded_statistics(n_vars),
        layout=GRIDDED_LAYOUT,
        coordinates=torch.zeros(4, 2),
        grid_size=4,
    )


def tabular_payload(n_points: int = 4) -> TabularSourceSample:
    return TabularSourceSample(
        data=torch.ones(1, n_points, 2),
        variables=["t2m", "sp"],
        statistics=TABULAR_STATISTICS,
        layout=TABULAR_LAYOUT,
        coordinates=torch.zeros(n_points, 2),
        timedeltas=torch.zeros(n_points),
        boundaries=(slice(0, n_points // 2), slice(n_points // 2, n_points)),
    )


def gridded_batch(variables: list[str] = ["a", "b", "c"]) -> Batch:
    n_vars = len(variables)
    return build_batch(
        data={"grid": torch.arange(2 * 2 * 1 * 4 * n_vars, dtype=torch.float32).reshape(2, 2, 1, 4, n_vars)},
        coordinates={"grid": torch.zeros(4, 2)},
        layouts={"grid": GRIDDED_LAYOUT.with_batch_dim()},
        variables={"grid": variables},
        statistics={"grid": {"mean": torch.arange(n_vars, dtype=torch.float32)}},
    )


class TestSourceMetadata:
    def test_rejects_duplicate_variable_names(self) -> None:
        with pytest.raises(ValueError, match="unique variable names"):
            GriddedSource(
                name="grid",
                variables=["a", "a"],
                layout=GRIDDED_LAYOUT,
                data=torch.zeros(1, 1, 4, 2),
                coordinates=torch.zeros(4, 2),
            )

    def test_fields_read_through_the_view(self) -> None:
        view = gridded_batch()["grid"]
        assert view.name == "grid"
        assert view.variables == ["a", "b", "c"]
        assert view.n_variables == 3
        assert view.layout == GRIDDED_LAYOUT.with_batch_dim()
        assert view.grid_size == 4
        assert view.coordinates_are_static is True
        assert view.name_to_index == {"a": 0, "b": 1, "c": 2}

    def test_name_to_index_is_memoised(self) -> None:
        batch = gridded_batch()
        assert batch["grid"].name_to_index is batch["grid"].name_to_index

    def test_name_to_index_follows_a_variable_rename(self) -> None:
        renamed = gridded_batch()["grid"].clone(variables=["x", "y", "z"])
        assert renamed.name_to_index == {"x": 0, "y": 1, "z": 2}

    @pytest.mark.parametrize(
        ("payload", "expected_type"),
        [(gridded_payload, GriddedTemplate), (tabular_payload, TabularTemplate)],
    )
    def test_template_keeps_everything_but_the_data(self, payload, expected_type) -> None:
        view = Batch.collate([{"src": payload()}])["src"]
        template = view.template()

        assert isinstance(template, expected_type)
        assert not hasattr(template, "data")
        assert (template.name, template.variables, template.layout) == (view.name, view.variables, view.layout)
        assert template.statistics is view.statistics
        assert template.coordinates is view.coordinates
        assert (template.batch_size, template.ensemble_size, template.time_size) == (
            view.batch_size,
            view.ensemble_size,
            view.time_size,
        )
        assert template.coordinates_are_static == view.coordinates_are_static

    @pytest.mark.parametrize("payload", [gridded_payload, tabular_payload])
    def test_template_unflatten_inverts_flatten(self, payload) -> None:
        view = Batch.collate([{"src": payload()}, {"src": payload()}])["src"]
        rebuilt = view.template().unflatten(view.flatten().data)

        assert type(rebuilt) is type(view)
        assert rebuilt.variables == view.variables
        data, expected = (rebuilt.data, view.data) if isinstance(view.data, list) else ([rebuilt.data], [view.data])
        assert all(torch.equal(a, b) for a, b in zip(data, expected, strict=True))

    @pytest.mark.parametrize("payload", [gridded_payload, tabular_payload])
    def test_template_flatten_matches_the_source_nodes(self, payload) -> None:
        view = Batch.collate([{"src": payload()}])["src"]
        flat, nodes = view.flatten(), view.template().flatten()

        assert nodes.data is None
        torch.testing.assert_close(nodes.coordinates, flat.coordinates)
        assert nodes.batch_sizes == flat.batch_sizes
        assert nodes.shard_sizes == flat.shard_sizes

    @pytest.mark.parametrize("payload", [gridded_payload, tabular_payload])
    def test_template_unflatten_rejects_a_wrong_shape(self, payload) -> None:
        template = Batch.collate([{"src": payload()}])["src"].template()
        with pytest.raises(ValueError, match="expects flat data of shape"):
            template.unflatten(torch.zeros(1, 1))

    def test_template_decodes_its_own_variables(self) -> None:
        view = gridded_batch()["grid"]
        template = view.template().with_variables(["z"], {"mean": torch.tensor([5.0])})
        rows = view.batch_size * view.ensemble_size * view.grid_size
        out = template.unflatten(torch.ones(rows, view.time_size * 1))

        assert out.variables == ["z"]
        torch.testing.assert_close(out.statistics["mean"], torch.tensor([5.0]))
        assert out.data.shape[out.layout.variables] == 1

    def test_template_with_ensemble_size_tiles_the_members(self) -> None:
        template = Batch.collate([{"obs": tabular_payload()}])["obs"].template().with_ensemble_size(3)
        assert template.ensemble_size == 3
        assert template.flatten().batch_sizes == (4, 4, 4)

    @pytest.mark.parametrize("payload", [gridded_payload, tabular_payload])
    def test_sources_require_data(self, payload) -> None:
        view = Batch.collate([{"src": payload()}])["src"]
        with pytest.raises(ValueError, match="requires data"):
            view.clone(data=None)


class TestSourceTransformations:
    """Transformations must carry the metadata through, and never mutate the receiver."""

    def test_clone_replaces_metadata(self) -> None:
        view = gridded_batch()["grid"]
        cloned = view.clone(variables=["x", "y", "z"])
        assert cloned.variables == ["x", "y", "z"]
        assert view.variables == ["a", "b", "c"]
        # payload is shared by reference when it is not replaced
        assert cloned.data is view.data

    def test_select_variables_keeps_data_and_metadata_consistent(self) -> None:
        view = gridded_batch()["grid"]
        selected = view.select(variables=[0, 2])
        assert selected.variables == ["a", "c"]
        assert selected.statistics["mean"].tolist() == [0.0, 2.0]
        assert selected.data.shape[selected.layout.variables] == 2
        # the receiver is not mutated
        assert view.variables == ["a", "b", "c"]

    def test_select_variables_accepts_a_slice(self) -> None:
        selected = gridded_batch()["grid"].select(variables=slice(1, 3))
        assert selected.variables == ["b", "c"]
        assert selected.statistics["mean"].tolist() == [1.0, 2.0]

    def test_select_variables_accepts_a_tensor(self) -> None:
        selected = gridded_batch()["grid"].select(variables=torch.tensor([2]))
        assert selected.variables == ["c"]
        assert selected.statistics["mean"].tolist() == [2.0]

    @pytest.mark.parametrize("indices", [1, slice(1, 2), [1], torch.tensor([1])])
    def test_select_time_keeps_the_time_axis(self, indices) -> None:
        view = gridded_batch()["grid"]
        selected = view.select(time=indices)
        time_axis = selected.layout.time
        assert selected.data.shape[time_axis] == 1
        torch.testing.assert_close(selected.data, view.data.narrow(time_axis, 1, 1))

    def test_select_time_on_a_tabular_source(self) -> None:
        view = Batch.collate([{"obs": tabular_payload()}])["obs"]
        selected = view.select(time=slice(1, None))
        assert selected.time_size == 1
        assert selected.boundaries == [(slice(0, 2),)]

    def test_select_variables_on_a_tabular_source(self) -> None:
        view = Batch.collate([{"obs": tabular_payload()}])["obs"]
        selected = view.select(variables=[1])
        assert selected.variables == ["sp"]
        assert selected.statistics["mean"].tolist() == [2.0]
        assert all(sample.shape[selected.layout.variables] == 1 for sample in selected.data)

    def test_map_data_applies_a_tensor_function_and_keeps_metadata(self) -> None:
        view = gridded_batch()["grid"]
        mapped = view.map_data(lambda t: t.to(torch.float64))
        assert mapped.dtype == torch.float64
        torch.testing.assert_close(mapped.data, view.data.to(torch.float64))
        assert (mapped.name, mapped.variables, mapped.layout) == (view.name, view.variables, view.layout)
        assert mapped.coordinates is view.coordinates
        # the receiver is not mutated
        assert view.dtype == torch.float32

    def test_map_data_does_not_clone_the_data(self) -> None:
        view = gridded_batch()["grid"]
        assert view.map_data(lambda t: t).data is view.data

    def test_map_data_applies_per_sample_on_a_tabular_source(self) -> None:
        view = Batch.collate([{"obs": tabular_payload(4)}, {"obs": tabular_payload(6)}])["obs"]
        seen = []
        mapped = view.map_data(lambda t: seen.append(tuple(t.shape)) or t * 2)
        assert seen == [(1, 4, 2), (1, 6, 2)]
        assert all(torch.equal(new, old * 2) for new, old in zip(mapped.data, view.data))
        assert mapped.timedeltas is view.timedeltas
        assert mapped.boundaries is view.boundaries


class TestCollate:
    def test_unsharded_tabular_samples_collate_to_replicated(self) -> None:
        batch = Batch.collate([{"obs": tabular_payload()}, {"obs": tabular_payload()}])
        assert batch["obs"].shard_sizes is None
        # replicated sources are returned unchanged by allgather and flatten without shard sizes
        assert batch["obs"].allgather(None) is batch["obs"]
        assert batch["obs"].flatten().shard_sizes is None

    def test_sharded_tabular_samples_keep_per_sample_shard_sizes(self) -> None:
        sizes = [[1, 1], [1, 1]]
        sample = replace(tabular_payload(), shard_sizes=sizes)
        batch = Batch.collate([{"obs": sample}])
        assert batch["obs"].shard_sizes == [sizes]

    def test_mixed_sharded_and_unsharded_samples_are_rejected(self) -> None:
        sharded = replace(tabular_payload(), shard_sizes=[[1, 1], [1, 1]])
        with pytest.raises(ValueError, match="mixes sharded and unsharded"):
            Batch.collate([{"obs": sharded}, {"obs": tabular_payload()}])

    @pytest.mark.parametrize(
        ("payload", "expected_type"),
        [(gridded_payload, GriddedSource), (tabular_payload, TabularSource)],
    )
    def test_sample_class_decides_the_source_type(self, payload, expected_type) -> None:
        assert isinstance(Batch.collate([{"src": payload()}])["src"], expected_type)

    def test_mixed_sample_kinds_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="single SourceSample subclass"):
            Batch.collate([{"src": gridded_payload()}, {"src": tabular_payload()}])

    def test_plain_dicts_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="single SourceSample subclass"):
            Batch.collate([{"src": {"data": torch.zeros(1)}}])

    @pytest.mark.parametrize(
        ("sample_cls", "layout", "extra"),
        [
            (GriddedSourceSample, TABULAR_LAYOUT, {}),
            (TabularSourceSample, GRIDDED_LAYOUT, {"timedeltas": torch.zeros(4), "boundaries": (slice(0, 4),)}),
        ],
    )
    def test_sample_rejects_a_layout_of_the_other_kind(self, sample_cls, layout, extra) -> None:
        with pytest.raises(ValueError, match="requires a layout"):
            sample_cls(
                data=torch.zeros(1, 4, 2), variables=["a", "b"], layout=layout, coordinates=torch.zeros(4, 2), **extra
            )


class TestBatchTransformations:
    def test_replace_swaps_one_source(self) -> None:
        batch = gridded_batch()
        renamed = batch.replace("grid", batch["grid"].clone(variables=["x", "y", "z"]))
        assert renamed["grid"].variables == ["x", "y", "z"]
        assert batch["grid"].variables == ["a", "b", "c"]

    def test_select_narrows_variables_and_their_lookup(self) -> None:
        batch = gridded_batch()
        selected = batch.select(variables=[1])
        assert selected["grid"].variables == ["b"]
        assert selected["grid"].name_to_index == {"b": 0}
        assert batch["grid"].variables == ["a", "b", "c"]

    def test_device_transfer_preserves_the_metadata(self) -> None:
        batch = gridded_batch()
        moved = batch.to("cpu")
        assert moved["grid"].variables is batch["grid"].variables
        assert moved["grid"].statistics is batch["grid"].statistics
