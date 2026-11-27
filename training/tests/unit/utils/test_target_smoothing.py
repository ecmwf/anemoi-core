# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from pydantic import ValidationError
from scipy import sparse
from torch_geometric.data import HeteroData

from anemoi.models.data import GriddedSource
from anemoi.models.data import TabularSource
from anemoi.models.data import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.utils.target_smoothing import TargetSmoothing
from anemoi.training.utils.target_smoothing import TargetSmoothingSchema
from anemoi.training.utils.target_smoothing import build_target_smoothers
from tests.batch_builders import build_batch

if TYPE_CHECKING:
    from pathlib import Path

GRID = 12
SIGMA = 1.5
CUTOFF = 4
# variables: 0 = prognostic, 1 = forcing, 2 = diagnostic mask, 3 = another diagnostic
NAME_TO_INDEX = {"t": 0, "orog": 1, "mask": 2, "other": 3}


def ring_matrix(scale_rows: np.ndarray | None = None) -> sparse.csr_matrix:
    """Gaussian weights on a periodic ring: symmetric, self weight 1, identical rows."""
    rows, cols, data = [], [], []
    for i in range(GRID):
        for j in range(GRID):
            d = min(abs(i - j), GRID - abs(i - j))
            if d <= CUTOFF:
                rows.append(i)
                cols.append(j)
                data.append(np.exp(-(d**2) / (2 * SIGMA**2)))
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(GRID, GRID), dtype=np.float32).tocsr()
    if scale_rows is not None:
        matrix = sparse.diags(scale_rows.astype(np.float32)).dot(matrix).tocsr()
    return matrix


@pytest.fixture
def matrix_path(tmp_path: Path) -> Path:
    path = tmp_path / "smoothing.npz"
    sparse.save_npz(path, ring_matrix())
    return path


@pytest.fixture
def data_indices() -> IndexCollection:
    config = DictConfig({"forcing": ["orog"], "diagnostic": ["mask", "other"]})
    return IndexCollection(config, NAME_TO_INDEX)


LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


def make_source(
    values: torch.Tensor,
    layout: TensorLayout = LAYOUT,
    shard_sizes: list[int] | None = None,
) -> GriddedSource:
    return GriddedSource(
        name="data",
        values=values,
        variables=list(NAME_TO_INDEX),
        statistics={},
        coordinates=torch.zeros(values.shape[layout.grid], 2),
        layout=layout,
        shard_sizes=shard_sizes,
    )


class OnTensor:
    """Apply a smoother to a plain ``(batch, time, ensemble, grid, variables)`` tensor, returning a tensor.

    The input tensor is never modified (the smoother works on a copy of the source).
    """

    def __init__(self, smoother: TargetSmoothing) -> None:
        self.smoother = smoother

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return self.smoother(make_source(values)).values

    def __getattr__(self, name: str):
        return getattr(self.smoother, name)


def make_smoother(
    path: Path | None,
    indices: IndexCollection,
    variables: list[str] | None = None,
    **kwargs,
) -> OnTensor:
    smoother = TargetSmoothing(variables or ["mask"], path, **kwargs)
    smoother.resolve_variables(indices)
    return OnTensor(smoother)


def one_storm_batch(cell: int = 5) -> torch.Tensor:
    batch = torch.rand(2, 3, 1, GRID, len(NAME_TO_INDEX))
    batch[..., 2] = 0.0
    batch[..., cell, 2] = 1.0
    return batch


def ring_graph(attribute: str = "gauss_weight") -> HeteroData:
    """The ring matrix as a ``data -> data`` edge set (edge_index = [source, target], matrix[target, source])."""
    coo = ring_matrix().tocoo()
    graph = HeteroData()
    graph["data"].num_nodes = GRID
    graph["data"].node_scale = torch.full((GRID,), 2.0)
    graph[("data", "to", "data")].edge_index = torch.from_numpy(np.stack([coo.col, coo.row]).astype(np.int64))
    graph[("data", "to", "data")][attribute] = torch.from_numpy(coo.data.astype(np.float32))
    return graph


def dense_reference(batch: torch.Tensor, variable: int) -> torch.Tensor:
    """The ring matrix applied as a dense matmul to one variable."""
    matrix = torch.from_numpy(ring_matrix().toarray())
    return torch.einsum("ij,btejv->bteiv", matrix, batch[..., [variable]])[..., 0]


def test_none_mode_gives_unit_peak_and_gaussian_blob(matrix_path: Path, data_indices: IndexCollection) -> None:
    smoother = make_smoother(matrix_path, data_indices)
    out = smoother(one_storm_batch(5).clone())

    blob = out[0, 0, 0, :, 2]
    assert blob[5] == pytest.approx(1.0)
    assert blob.argmax() == 5
    distance = np.array([min(abs(i - 5), GRID - abs(i - 5)) for i in range(GRID)])
    expected = np.where(distance <= CUTOFF, np.exp(-(distance**2) / (2 * SIGMA**2)), 0.0)
    np.testing.assert_allclose(blob.numpy(), expected, atol=1e-6)


def test_none_mode_matches_a_dense_matmul(matrix_path: Path, data_indices: IndexCollection) -> None:
    batch = torch.rand(2, 3, 1, GRID, len(NAME_TO_INDEX))
    expected = dense_reference(batch, 2)

    out = make_smoother(matrix_path, data_indices)(batch.clone())

    torch.testing.assert_close(out[..., 2], expected, rtol=1e-6, atol=1e-6)


def test_none_mode_uses_the_matrix_as_given(tmp_path: Path, data_indices: IndexCollection) -> None:
    path = tmp_path / "scaled.npz"
    sparse.save_npz(path, ring_matrix(scale_rows=np.full(GRID, 0.5)))

    out = make_smoother(path, data_indices)(one_storm_batch(5).clone())

    assert out[..., 5, 2].max() == pytest.approx(0.5)


def test_none_mode_warns_when_the_diagonal_is_not_one(
    tmp_path: Path,
    data_indices: IndexCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "scaled.npz"
    scale = np.linspace(0.5, 2.0, GRID)
    sparse.save_npz(path, ring_matrix(scale_rows=scale))

    with caplog.at_level("WARNING"):
        make_smoother(path, data_indices)

    assert "diagonal of the matrix is not 1 (min=0.5, max=2)" in caplog.text


def test_none_mode_warns_when_a_diagonal_entry_is_missing(
    tmp_path: Path,
    data_indices: IndexCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "noself.npz"
    sparse.save_npz(path, ring_matrix() - sparse.eye(GRID, dtype=np.float32))

    with caplog.at_level("WARNING"):
        make_smoother(path, data_indices)

    assert "diagonal of the matrix is not 1 (min=0, max=0)" in caplog.text


def test_no_warnings_for_a_diagonal_one_matrix(
    matrix_path: Path,
    data_indices: IndexCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        make_smoother(matrix_path, data_indices)

    assert caplog.text == ""


def test_row_mode_does_not_warn_about_row_sums_or_diagonal(
    matrix_path: Path,
    data_indices: IndexCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        make_smoother(matrix_path, data_indices, normalisation="row")

    assert caplog.text == ""


def test_none_mode_does_not_warn_about_row_sums(
    matrix_path: Path,
    data_indices: IndexCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A smoothing kernel is not row-normalised on purpose, so the provider warning is switched off."""
    with caplog.at_level("WARNING", logger="anemoi.models.layers.graph_provider"):
        make_smoother(matrix_path, data_indices)

    assert "rows do not sum to 1" not in caplog.text


def test_matrix_is_applied_in_float32_and_the_batch_dtype_is_kept(
    matrix_path: Path,
    data_indices: IndexCollection,
) -> None:
    smoother = make_smoother(matrix_path, data_indices)

    out = smoother(one_storm_batch(5).to(torch.float64))

    assert out.dtype == torch.float64
    assert smoother._provider.get_edges().dtype == torch.float32
    assert out[..., 5, 2].min() == pytest.approx(1.0)


def test_row_mode_preserves_constants_and_sums(matrix_path: Path, data_indices: IndexCollection) -> None:
    smoother = make_smoother(matrix_path, data_indices, normalisation="row")

    constant = torch.full((1, 1, 1, GRID, len(NAME_TO_INDEX)), 3.0)
    torch.testing.assert_close(smoother(constant.clone())[..., 2], constant[..., 2])

    # Rows and columns have equal sums on the ring, so the total is preserved too.
    out = smoother(one_storm_batch(5).clone())
    assert out[0, 0, 0, :, 2].sum() == pytest.approx(1.0, abs=1e-5)
    assert out[0, 0, 0, 5, 2] < 1.0


def test_other_variables_are_untouched_and_the_input_is_not_modified(
    matrix_path: Path,
    data_indices: IndexCollection,
) -> None:
    batch = one_storm_batch()
    reference = batch.clone()

    out = make_smoother(matrix_path, data_indices)(batch)

    assert out is not batch
    assert torch.equal(batch, reference)
    for name, index in NAME_TO_INDEX.items():
        if name == "mask":
            assert not torch.equal(out[..., index], reference[..., index])
        else:
            assert torch.equal(out[..., index], reference[..., index])


def test_all_batch_elements_and_times_are_smoothed(matrix_path: Path, data_indices: IndexCollection) -> None:
    out = make_smoother(matrix_path, data_indices)(one_storm_batch(7).clone())

    torch.testing.assert_close(out[..., 7, 2], torch.ones_like(out[..., 7, 2]))
    assert (out[..., 6, 2] > 0).all()


def test_whole_field_nan_is_restored_and_does_not_spread(matrix_path: Path, data_indices: IndexCollection) -> None:
    batch = one_storm_batch(5)
    batch[0, 1, 0, :, 2] = torch.nan  # one missing field (e.g. a missing year)
    batch[1, 2, 0, :, 3] = torch.nan  # NaN in a variable that is not smoothed

    out = make_smoother(matrix_path, data_indices, variables=["mask", "other"])(batch.clone())

    assert torch.isnan(out[0, 1, 0, :, 2]).all()
    assert torch.isnan(out[1, 2, 0, :, 3]).all()
    # All other fields are finite, NaN did not leak across batch, time or grid.
    finite = torch.ones(out.shape, dtype=torch.bool)
    finite[0, 1, :, :, 2] = False
    finite[1, 2, :, :, 3] = False
    assert torch.isfinite(out[finite]).all()
    assert out[0, 0, 0, 5, 2] == pytest.approx(1.0)


def test_point_nan_stays_nan_without_spreading(matrix_path: Path, data_indices: IndexCollection) -> None:
    batch = one_storm_batch(5)
    batch[0, 0, 0, 8, 2] = torch.nan

    out = make_smoother(matrix_path, data_indices)(batch.clone())

    assert torch.isnan(out[0, 0, 0, 8, 2])
    assert torch.isnan(out[0, 0, 0, :, 2]).sum() == 1


def test_clamp_max_caps_overlapping_blobs(matrix_path: Path, data_indices: IndexCollection) -> None:
    batch = one_storm_batch(5)
    batch[..., 6, 2] = 1.0

    uncapped = make_smoother(matrix_path, data_indices)(batch.clone())
    capped = make_smoother(matrix_path, data_indices, clamp_max=1.0)(batch.clone())

    assert uncapped[..., 2].max() > 1.0
    assert capped[..., 2].max() == pytest.approx(1.0)


def test_grid_size_mismatch_is_rejected(matrix_path: Path, data_indices: IndexCollection) -> None:
    smoother = make_smoother(matrix_path, data_indices)

    with pytest.raises(ValueError, match="grid points"):
        smoother(torch.zeros(1, 1, 1, GRID + 1, len(NAME_TO_INDEX)))


def test_must_resolve_variables_first(matrix_path: Path) -> None:
    with pytest.raises(RuntimeError, match="resolve_variables"):
        TargetSmoothing(["mask"], matrix_path)(make_source(torch.zeros(1, 1, 1, GRID, 4)))


def test_a_variable_missing_from_the_source_is_rejected(matrix_path: Path, data_indices: IndexCollection) -> None:
    smoother = make_smoother(matrix_path, data_indices).smoother
    source = make_source(torch.zeros(1, 1, 1, GRID, 4)).select(variables=[0, 1, 3])

    with pytest.raises(ValueError, match="not in the batch"):
        smoother(source)


def test_a_tabular_source_is_rejected(matrix_path: Path, data_indices: IndexCollection) -> None:
    smoother = make_smoother(matrix_path, data_indices).smoother
    tabular = build_batch(
        values={"data": [torch.zeros(GRID, len(NAME_TO_INDEX))]},
        coordinates={"data": [torch.zeros(GRID, 2)]},
        timedeltas={"data": [torch.zeros(GRID, dtype=torch.int64)]},
        boundaries={"data": [(slice(0, GRID),)]},
        layouts={"data": TensorLayout(batch=None, time=None, ensemble=None, grid=0, variables=1)},
        variables={"data": list(NAME_TO_INDEX)},
    )["data"]
    assert isinstance(tabular, TabularSource)

    with pytest.raises(ValueError, match="tabular"):
        smoother(tabular)


def test_a_non_canonical_layout_gives_the_same_result(matrix_path: Path, data_indices: IndexCollection) -> None:
    """The grid and variables axes come from the layout, not from fixed positions."""
    batch = torch.rand(2, 3, 1, GRID, len(NAME_TO_INDEX))
    expected = make_smoother(matrix_path, data_indices)(batch.clone())

    # Axis order: variables, batch, grid, time, ensemble.
    layout = TensorLayout(batch=1, time=3, ensemble=4, grid=2, variables=0)
    permuted = batch.permute(4, 0, 3, 1, 2).contiguous()
    smoother = make_smoother(matrix_path, data_indices).smoother
    out = smoother(make_source(permuted, layout=layout)).values

    torch.testing.assert_close(out.permute(1, 3, 4, 2, 0), expected)


def test_unknown_variable_is_rejected(matrix_path: Path, data_indices: IndexCollection) -> None:
    with pytest.raises(ValueError, match="not dataset variables"):
        make_smoother(matrix_path, data_indices, variables=["mask", "nope"])


@pytest.mark.parametrize("name", ["t", "orog"])
def test_input_variables_are_rejected(matrix_path: Path, data_indices: IndexCollection, name: str) -> None:
    with pytest.raises(ValueError, match="model inputs"):
        make_smoother(matrix_path, data_indices, variables=["mask", name])


def test_matrix_must_be_square(tmp_path: Path) -> None:
    path = tmp_path / "rect.npz"
    sparse.save_npz(path, sparse.csr_matrix(np.ones((3, 4), dtype=np.float32)))

    with pytest.raises(ValueError, match="square"):
        TargetSmoothing(["mask"], path)


def test_unknown_normalisation_is_rejected(matrix_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown normalisation"):
        TargetSmoothing(["mask"], matrix_path, normalisation="l1")


# Matrix from the graph


def test_graph_route_equals_file_route(matrix_path: Path, data_indices: IndexCollection) -> None:
    batch = torch.rand(2, 3, 1, GRID, len(NAME_TO_INDEX))
    from_file = make_smoother(matrix_path, data_indices)(batch.clone())

    graph_smoother = make_smoother(
        None,
        data_indices,
        edges_name=("data", "to", "data"),
        edge_weight_attribute="gauss_weight",
        graph_data=ring_graph(),
    )
    from_graph = graph_smoother(batch.clone())

    assert graph_smoother.grid_size == GRID
    torch.testing.assert_close(from_graph, from_file)


def test_graph_route_with_row_normalisation_and_list_edges_name(
    matrix_path: Path,
    data_indices: IndexCollection,
) -> None:
    batch = one_storm_batch(5)
    from_file = make_smoother(matrix_path, data_indices, normalisation="row")(batch.clone())

    from_graph = make_smoother(
        None,
        data_indices,
        normalisation="row",
        edges_name=["data", "to", "data"],
        edge_weight_attribute="gauss_weight",
        graph_data=ring_graph(),
    )(batch.clone())

    torch.testing.assert_close(from_graph, from_file)


def test_graph_route_applies_the_source_node_weights(matrix_path: Path, data_indices: IndexCollection) -> None:
    batch = one_storm_batch(5)
    from_file = make_smoother(matrix_path, data_indices)(batch.clone())

    from_graph = make_smoother(
        None,
        data_indices,
        edges_name=("data", "to", "data"),
        edge_weight_attribute="gauss_weight",
        src_node_weight_attribute="node_scale",
        graph_data=ring_graph(),
    )(batch.clone())

    torch.testing.assert_close(from_graph[..., 2], 2.0 * from_file[..., 2])


def test_graph_route_warns_when_the_diagonal_is_not_one(
    data_indices: IndexCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        make_smoother(
            None,
            data_indices,
            edges_name=("data", "to", "data"),
            edge_weight_attribute="gauss_weight",
            src_node_weight_attribute="node_scale",
            graph_data=ring_graph(),
        )

    assert "diagonal of the matrix is not 1 (min=2, max=2)" in caplog.text


def test_graph_route_needs_the_graph(data_indices: IndexCollection) -> None:
    with pytest.raises(ValueError, match="needs the training graph"):
        make_smoother(
            None,
            data_indices,
            edges_name=("data", "to", "data"),
            edge_weight_attribute="gauss_weight",
        )


def test_graph_route_rejects_unknown_edge_set(data_indices: IndexCollection) -> None:
    with pytest.raises(ValueError, match="is not in the graph"):
        make_smoother(
            None,
            data_indices,
            edges_name=("data", "to", "hidden"),
            edge_weight_attribute="gauss_weight",
            graph_data=ring_graph(),
        )


def test_graph_route_rejects_unknown_attributes(data_indices: IndexCollection) -> None:
    with pytest.raises(ValueError, match="edge_weight_attribute 'nope'"):
        make_smoother(
            None,
            data_indices,
            edges_name=("data", "to", "data"),
            edge_weight_attribute="nope",
            graph_data=ring_graph(),
        )
    with pytest.raises(ValueError, match="src_node_weight_attribute 'nope'"):
        make_smoother(
            None,
            data_indices,
            edges_name=("data", "to", "data"),
            edge_weight_attribute="gauss_weight",
            src_node_weight_attribute="nope",
            graph_data=ring_graph(),
        )


def test_graph_route_rejects_different_source_and_target_grids(data_indices: IndexCollection) -> None:
    graph = ring_graph()
    graph["other"].num_nodes = GRID + 1
    graph[("data", "to", "other")].edge_index = torch.tensor([[0], [0]])
    graph[("data", "to", "other")].gauss_weight = torch.ones(1)

    with pytest.raises(ValueError, match="same grid"):
        make_smoother(
            None,
            data_indices,
            edges_name=("data", "to", "other"),
            edge_weight_attribute="gauss_weight",
            graph_data=graph,
        )


# Config


def test_schema_defaults(matrix_path: Path) -> None:
    schema = TargetSmoothingSchema(variables=["mask"], matrix_path=matrix_path)

    assert schema.normalisation == "none"
    assert schema.clamp_max is None
    assert schema.edges_name is None


def test_schema_accepts_edges_name_as_a_list() -> None:
    schema = TargetSmoothingSchema(
        variables=["mask"],
        edges_name=["data", "to", "data"],
        edge_weight_attribute="gauss_weight",
    )

    assert schema.edges_name == ("data", "to", "data")
    assert schema.matrix_path is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"variables": []},
        {"variables": ["mask", "mask"]},
        {"normalisation": "l1"},
        {"clamp_max": 0.0},
        {"unknown_key": 1},
        {"edges_name": ["data", "to", "data"], "edge_weight_attribute": "w"},  # both sources
        {"edge_weight_attribute": "w"},  # attribute without edges_name
        {"src_node_weight_attribute": "w"},  # attribute without edges_name
        {"matrix_path": None},  # no source
        {"matrix_path": None, "edges_name": ["data", "to"], "edge_weight_attribute": "w"},  # not a triple
        {"matrix_path": None, "edges_name": ["data", "to", "data"]},  # no weights
    ],
)
def test_schema_rejects_invalid_config(matrix_path: Path, overrides: dict) -> None:
    config = {"variables": ["mask"], "matrix_path": matrix_path} | overrides

    with pytest.raises(ValidationError):
        TargetSmoothingSchema(**config)


def test_build_target_smoothers_from_dictconfig(matrix_path: Path, data_indices: IndexCollection) -> None:
    data_config = DictConfig(
        {
            "datasets": {
                "a": {"target_smoothing": {"variables": ["mask"], "matrix_path": str(matrix_path), "clamp_max": 1.0}},
                "b": {"processors": {}},
            },
        },
    )

    smoothers = build_target_smoothers(
        data_config.datasets,
        {"a": data_indices, "b": data_indices},
        grid_sizes={"a": GRID, "b": 99},
    )

    assert list(smoothers) == ["a"]
    assert smoothers["a"].variables == ["mask"]
    assert smoothers["a"].clamp_max == 1.0


def test_build_target_smoothers_from_graph_edges(matrix_path: Path, data_indices: IndexCollection) -> None:
    data_config = DictConfig(
        {
            "a": {
                "target_smoothing": {
                    "variables": ["mask"],
                    "edges_name": ["data", "to", "data"],
                    "edge_weight_attribute": "gauss_weight",
                },
            },
        },
    )

    smoothers = build_target_smoothers(
        data_config,
        {"a": data_indices},
        grid_sizes={"a": GRID},
        graph_data=ring_graph(),
    )

    batch = one_storm_batch(5)
    expected = make_smoother(matrix_path, data_indices)(batch.clone())
    torch.testing.assert_close(OnTensor(smoothers["a"])(batch.clone()), expected)


def test_build_target_smoothers_graph_edges_check_grid_size(data_indices: IndexCollection) -> None:
    data_config = {
        "a": {
            "target_smoothing": {
                "variables": ["mask"],
                "edges_name": ["data", "to", "data"],
                "edge_weight_attribute": "gauss_weight",
            },
        },
    }

    with pytest.raises(ValueError, match="dataset grid"):
        build_target_smoothers(data_config, {"a": data_indices}, grid_sizes={"a": GRID + 1}, graph_data=ring_graph())


def test_build_target_smoothers_checks_grid_size(matrix_path: Path, data_indices: IndexCollection) -> None:
    data_config = {"a": {"target_smoothing": {"variables": ["mask"], "matrix_path": str(matrix_path)}}}

    with pytest.raises(ValueError, match="dataset grid"):
        build_target_smoothers(data_config, {"a": data_indices}, grid_sizes={"a": GRID + 1})


def test_build_target_smoothers_rejects_unknown_variable(matrix_path: Path, data_indices: IndexCollection) -> None:
    data_config = {"a": {"target_smoothing": {"variables": ["nope"], "matrix_path": str(matrix_path)}}}

    with pytest.raises(ValueError, match="not dataset variables"):
        build_target_smoothers(data_config, {"a": data_indices})
