# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from pydantic import ValidationError
from torch_geometric.data import HeteroData

from anemoi.training.losses import DTCWT2DLoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.schemas.training import DTCWT2DLossSchema
from anemoi.training.schemas.training import ImplementedLossesUsingBaseLossSchema
from anemoi.training.schemas.training import LossSchemas

pytest.importorskip("pytorch_wavelets")


@pytest.fixture(autouse=True, scope="module")
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def isolated_compile_cache() -> Iterator[None]:
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _data(
    shape: tuple[int, int] = (8, 10),
    batch: int = 1,
    time: int = 1,
    ensemble: int = 1,
    variables: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    dims = (batch, time, ensemble, shape[0] * shape[1], variables)
    return torch.randn(dims, generator=generator), torch.randn(dims, generator=generator)


def test_public_dtcwt_2d_name_and_factory_target() -> None:
    from anemoi.training import losses

    target = "anemoi.training.losses.DTCWT2DLoss"
    config = {"_target_": target, "grid_shape": [8, 10], "J": 2}
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert isinstance(validated, DTCWT2DLossSchema)
    assert validated.target_ == target
    assert ImplementedLossesUsingBaseLossSchema.dtcwt_2d.value == target
    assert losses.DTCWT2DLoss is DTCWT2DLoss
    assert "DTCWT2DLoss" in losses.__all__
    loss = get_loss_function(OmegaConf.create(config))
    assert isinstance(loss, DTCWT2DLoss)
    assert loss.name == "dtcwt_2d"


def test_field_gathering_and_fp32_casting_are_chunk_local(monkeypatch: pytest.MonkeyPatch) -> None:
    pred, target = _data((10, 9), batch=2, time=2, ensemble=2, variables=3)
    pred = pred.bfloat16().transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    target = target[:, :, :1].bfloat16()
    loss = DTCWT2DLoss([7, 9], J=2, subgrid=(5, 68), transform_field_chunk_size=5)
    chunk_inputs = loss._chunk_inputs
    chunk_sizes = []

    def record(
        values: torch.Tensor,
        indices: tuple[torch.Tensor, ...],
        projection_matrix: torch.Tensor | None,
    ) -> torch.Tensor:
        assert values.dtype == torch.bfloat16
        assert values.shape[3] == 63
        result = chunk_inputs(values, indices, projection_matrix)
        assert result.dtype == torch.float32
        assert result.shape[1:] == (1, 7, 9)
        chunk_sizes.append(result.shape[0])
        return result

    monkeypatch.setattr(loss, "_chunk_inputs", record)
    actual = loss(pred, target)
    expected = DTCWT2DLoss([7, 9], J=2)(pred[:, :, :, 5:68].float(), target[:, :, :, 5:68].float())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])
    assert chunk_sizes == [5, 5, 5, 5, 5, 5, 5, 5, 4, 4]


def test_ignore_nans_masks_affected_coefficients_and_preserves_gradients() -> None:
    pred, target = _data(variables=1)
    ordinary_loss = DTCWT2DLoss([8, 10], J=2)
    nan_aware_loss = DTCWT2DLoss([8, 10], J=2, ignore_nans=True)
    torch.testing.assert_close(nan_aware_loss(pred, target), ordinary_loss(pred, target))

    missing_location = (0, 0, 0, 35, 0)
    pred[missing_location] = float("nan")
    pred.requires_grad_()
    score = nan_aware_loss(pred, target)
    assert torch.isfinite(score)
    pred_gradient = torch.autograd.grad(score, pred)[0]
    assert torch.isfinite(pred_gradient).all()
    assert pred_gradient[missing_location] == 0


def test_ignore_nans_masks_target_and_all_missing_fields() -> None:
    pred, target = _data(variables=1)
    target[..., 0, :] = float("nan")
    loss = DTCWT2DLoss([8, 10], J=2, ignore_nans=True)
    assert torch.isfinite(loss(pred, target))
    all_missing_score = loss(torch.full_like(pred, float("nan")), torch.full_like(target, float("nan")))
    torch.testing.assert_close(all_missing_score, torch.zeros_like(all_missing_score))


def test_pooled_nan_handling_uses_fractional_support_and_preserves_gradients() -> None:
    complete_pred, complete_target = _data(variables=1)
    pooled_loss = DTCWT2DLoss([8, 10], J=2, nan_handling="pooled")
    ordinary_loss = DTCWT2DLoss([8, 10], J=2)
    torch.testing.assert_close(
        pooled_loss(complete_pred, complete_target),
        ordinary_loss(complete_pred, complete_target),
    )

    pred, target = _data(variables=1)
    pred[0, 0, 0, 0, 0] = 0.0
    missing_location = (0, 0, 0, 35, 0)
    pred[missing_location] = float("nan")
    pred.requires_grad_()

    loss = DTCWT2DLoss([8, 10], J=2, nan_handling="pooled")
    assert loss.ignore_nans
    assert loss.nan_handling == "pooled"

    field_indices = tuple(torch.tensor([0]) for _ in range(4))
    pred_chunk, _, valid_grid = loss._chunk_inputs_with_validity(pred, target, field_indices, None)
    assert valid_grid[0, 0, 0, 0]
    assert not valid_grid[0, 0, 3, 5]

    pyramid = loss.transform(pred_chunk)
    support = loss._pooled_coefficient_support(valid_grid, pyramid)
    assert any(torch.any((level > 0) & (level < 1)) for level in (*support.highpasses, support.lowpass))

    score = loss(pred, target)
    assert torch.isfinite(score)
    gradient = torch.autograd.grad(score, pred)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[missing_location] == 0


@pytest.mark.parametrize("field", ["pred", "target"])
def test_ignore_nans_does_not_mask_infinities(field: str) -> None:
    pred, target = _data()
    (pred if field == "pred" else target)[..., 1, 0] = float("inf")
    loss = DTCWT2DLoss([8, 10], J=2, ignore_nans=True)
    assert not torch.isfinite(loss(pred, target))


def test_nan_inputs_propagate_without_nan_handling() -> None:
    pred, target = _data()
    pred[..., 1, 0] = float("nan")
    loss = DTCWT2DLoss([8, 10], J=2)
    assert not torch.isfinite(loss(pred, target))


def test_subgrid_scalers_and_reductions() -> None:
    pred, target = _data((9, 10), time=2, ensemble=2, variables=3)
    loss = DTCWT2DLoss([8, 10], J=2, subgrid=(5, 85))
    reference = DTCWT2DLoss([8, 10], J=2)
    loss.add_scaler(4, torch.tensor([1.0, 2.0, 3.0]), name="variable")
    reference.add_scaler(4, torch.tensor([1.0, 2.0, 3.0]), name="variable")
    loss.add_scaler(1, torch.tensor([0.25, 0.75]), name="time")
    reference.add_scaler(1, torch.tensor([0.25, 0.75]), name="time")
    expected = reference(pred[:, :, :, 5:85], target[:, :, :, 5:85], squash=False)
    actual = loss(pred, target, squash=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(loss(pred, target), actual.mean())
    torch.testing.assert_close(loss(pred, target, squash_mode="sum"), actual.sum())
    torch.testing.assert_close(loss(pred, target, squash=False, scaler_indices=(..., [2, 0])), actual[[2, 0]])
    torch.testing.assert_close(
        loss(pred, target, without_scalers=["variable"], squash=False),
        actual / torch.tensor([1, 2, 3]),
    )
    torch.testing.assert_close(
        loss(pred, target, without_scalers=[4], squash=False),
        actual / torch.tensor([1, 2, 3]),
    )


@pytest.mark.parametrize("value", [0.0, 1.0, 1e-5])
def test_identical_and_zero_fields_have_stable_gradients(value: float) -> None:
    pred = torch.full((1, 1, 1, 80, 1), value, requires_grad=True)
    target = pred.detach().clone()
    loss = DTCWT2DLoss([8, 10], J=2)
    out = loss(pred, target)
    assert torch.isfinite(out) and out >= -1e-6
    assert torch.isfinite(torch.autograd.grad(out, pred)[0]).all()
    assert out < 1e-5


def test_anemoi_compile_configuration_selects_the_scoring_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from anemoi.models.utils.compile import mark_for_compilation

    calls = []

    def compile_kernel(kernel: Callable[..., torch.Tensor], **options: Any) -> Callable[..., torch.Tensor]:
        calls.append((kernel, options))
        return kernel

    monkeypatch.setattr("anemoi.models.utils.compile._meets_library_versions_for_compile", lambda: True)
    monkeypatch.setattr(torch, "compile", compile_kernel)
    model = torch.nn.Module()
    loss = DTCWT2DLoss([8, 10], J=2)
    model.add_module("loss", loss)
    config = OmegaConf.create(
        [{"module": "anemoi.training.losses.DTCWT2DLoss", "options": {"fullgraph": True, "dynamic": False}}],
    )
    mark_for_compilation(model, config)
    assert calls == [(loss._score_pyramid, {"fullgraph": True, "dynamic": False})]


@pytest.mark.parametrize(
    "options",
    [
        {"grid_shape": [1, 8]},
        {"J": 0},
        {"eps": 0},
        {"scale_weights": [1, -1]},
        {"subgrid": (5, 2)},
        {"transform_field_chunk_size": 0},
    ],
)
def test_invalid_configuration_is_rejected_by_schema(options: dict[str, Any]) -> None:
    config = {"_target_": "anemoi.training.losses.DTCWT2DLoss", "grid_shape": [8, 10], "J": 2, **options}
    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python(config)


def test_scale_weight_count_matches_wavelet_levels() -> None:
    with pytest.raises(ValueError, match="one weight per wavelet level"):
        DTCWT2DLoss([8, 10], J=2, scale_weights=[1])


@pytest.mark.parametrize("dimension", [3, (1, 3), (0, 2, 3, 4)])
def test_area_scalers_are_rejected_on_add_and_update(dimension: int | tuple[int, ...]) -> None:
    loss = DTCWT2DLoss([8, 10], J=2)
    with pytest.raises(ValueError, match="does not support area weights"):
        loss.add_scaler(dimension, torch.ones(80), name="area")

    with pytest.raises(ValueError, match="does not support area weights"):
        get_loss_function(
            OmegaConf.create(
                {
                    "_target_": "anemoi.training.losses.DTCWT2DLoss",
                    "grid_shape": [8, 10],
                    "J": 2,
                    "scalers": ["area"],
                },
            ),
            {"area": (dimension, torch.ones(80))},
        )

    loss.scaler.add_scaler(dimension=3, scaler=torch.ones(80), name="legacy_area")
    with pytest.raises(ValueError, match="does not support area weights"):
        loss.update_scaler("legacy_area", torch.ones(80))
    with pytest.raises(ValueError, match="does not support area weights"):
        loss(*_data(), without_scalers=["legacy_area"])

    with pytest.raises(TypeError, match="target_grid_weights"):
        DTCWT2DLoss([8, 10], J=2, target_grid_weights=[1.0] * 80)
    with pytest.raises(ValidationError, match="target_grid_weights"):
        TypeAdapter(LossSchemas).validate_python(
            {
                "_target_": "anemoi.training.losses.DTCWT2DLoss",
                "grid_shape": [8, 10],
                "target_grid_weights": [1.0] * 80,
            },
        )


def test_schema_factory_output_mask_resolution() -> None:
    from anemoi.training.train.methods.base import resolve_subgrid

    config = {
        "_target_": "anemoi.training.losses.DTCWT2DLoss",
        "grid_shape": [8, 10],
        "J": 2,
        "subgrid": "output_mask",
        "transform": {"backend": "pytorch_wavelets"},
    }
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert isinstance(validated, DTCWT2DLossSchema)
    data_config = OmegaConf.create({"regional": config})

    class Mask:
        def as_tuple(self) -> tuple[int, int]:
            return (5, 85)

    resolve_subgrid(data_config, {"regional": Mask()})
    standalone = get_loss_function(data_config.regional)
    pred, target = _data((9, 10))
    assert torch.isfinite(standalone(pred, target))


def _projection_matrix() -> torch.Tensor:
    """Return a known constant-preserving 13-source to 48-target interpolation."""
    matrix = torch.zeros(48, 13)
    rows = torch.arange(48)
    matrix[rows, rows % 13] = 0.25
    matrix[rows, (rows + 4) % 13] = 0.75
    return matrix


def _projection_graph(matrix: torch.Tensor) -> tuple[HeteroData, dict[str, Any]]:
    """Encode a small dense reference matrix as weighted source-to-target edges."""
    graph = HeteroData()
    graph["source"].num_nodes = matrix.shape[1]
    graph["wavelet_grid"].num_nodes = matrix.shape[0]
    rows, columns = matrix.nonzero(as_tuple=True)
    graph["source", "to", "wavelet_grid"].edge_index = torch.stack((columns, rows))
    graph["source", "to", "wavelet_grid"].weight = matrix[rows, columns]
    return graph, {
        "edges_name": ("source", "to", "wavelet_grid"),
        "edge_weight_attribute": "weight",
    }


def test_projection_uses_the_selected_subgrid_source_nodes() -> None:
    graph, config = _projection_graph(_projection_matrix())
    pred, target = _data((1, 17), variables=1)
    pred.requires_grad_()
    options = {"grid_shape": [6, 8], "J": 2, "projection_config": config, "graph_data": graph}
    validated = TypeAdapter(LossSchemas).validate_python(
        {
            "_target_": "anemoi.training.losses.DTCWT2DLoss",
            "grid_shape": [6, 8],
            "J": 2,
            "subgrid": [2, 15],
            "projection_config": config,
        },
    )
    assert validated.subgrid == (2, 15)
    subgrid_loss = DTCWT2DLoss(**options, subgrid=(2, 15))
    reference_loss = DTCWT2DLoss(**options)

    actual = subgrid_loss(pred, target, squash=False)
    expected = reference_loss(pred[:, :, :, 2:15], target[:, :, :, 2:15], squash=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
    )

    with pytest.raises(ValueError, match="selected input grid size"):
        DTCWT2DLoss(**options, subgrid=(2, 14))(pred, target, squash=False)


@pytest.mark.parametrize("nan_handling", ["strict", "pooled"])
def test_projection_nan_handling_propagates_source_validity(nan_handling: str) -> None:
    matrix = _projection_matrix()
    graph, config = _projection_graph(matrix)
    pred, target = _data((1, 13), variables=1)
    pred[..., 0, 0] = float("nan")
    pred.requires_grad_()
    loss = DTCWT2DLoss(
        [6, 8],
        J=2,
        projection_config=config,
        graph_data=graph,
        nan_handling=nan_handling,
        transform_field_chunk_size=2,
    )

    score = loss(pred, target, squash=False)
    assert torch.isfinite(score).all()
    pred_gradient = torch.autograd.grad(score.sum(), pred)[0]
    assert torch.isfinite(pred_gradient).all()
    assert pred_gradient[..., 0, 0].eq(0).all()


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("row_normalize", [False, True])
def test_projection_identity_and_permutation_preserve_target_order(permuted: bool, row_normalize: bool) -> None:
    matrix = torch.eye(48)
    if permuted:
        matrix = matrix.flip(0)
    graph, config = _projection_graph(matrix)
    loss = DTCWT2DLoss(
        [6, 8],
        J=2,
        projection_config={**config, "row_normalize": row_normalize},
        graph_data=graph,
    )
    pred, target = _data((6, 8))
    pred.requires_grad_()
    actual = loss(pred, target)
    expected = DTCWT2DLoss([6, 8], J=2)(
        pred.flip(3) if permuted else pred,
        target.flip(3) if permuted else target,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])


def test_projection_knn_target_builder_schema_and_factory() -> None:
    graph = HeteroData()
    graph["source"].x = torch.deg2rad(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    graph["source"].num_nodes = 4
    config = {
        "_target_": "anemoi.training.losses.DTCWT2DLoss",
        "grid_shape": [2, 3],
        "J": 1,
        "scalers": ["variable"],
        "projection_config": {
            "target_node_name": "wavelet_grid",
            "num_nearest_neighbours": 2,
            "sigma": 0.02,
            "node_builder": {
                "_target_": "anemoi.graphs.nodes.LatLonNodes",
                "latitudes": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "longitudes": [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
            },
        },
    }
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert validated.projection_config.row_normalize
    loss = get_loss_function(
        OmegaConf.create(validated.model_dump(by_alias=True)),
        {"variable": (4, torch.tensor([2.0]))},
        graph_data=graph,
        data_node_name="source",
    )
    matrix = loss.projection_provider.get_edges().to_dense()
    torch.testing.assert_close(matrix.sum(1), torch.ones(6))
    torch.testing.assert_close(
        loss.projector(torch.ones(2, 4, 1), loss.projection_provider.get_edges()),
        torch.ones(2, 6, 1),
    )
    fields = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(1, 1, 1, 4, 1).requires_grad_()
    zeros = torch.zeros_like(fields)
    actual = loss(fields, zeros, squash=False)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(torch.autograd.grad(actual.sum(), fields)[0]).all()
    assert graph.node_types == ["source"]
    with pytest.raises(ValueError, match="n_neighbors"):
        DTCWT2DLoss(
            [2, 3],
            projection_config={**config["projection_config"], "num_nearest_neighbours": 5},
            graph_data=graph,
            data_node_name="source",
        )
    with pytest.raises(ValueError, match="missing from graph_data"):
        DTCWT2DLoss(
            [2, 3],
            projection_config=config["projection_config"],
            graph_data=graph,
            data_node_name="missing",
        )


def test_projection_gathering_and_projection_stay_chunk_local(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, config = _projection_graph(_projection_matrix())
    loss = DTCWT2DLoss(
        [6, 8],
        J=2,
        projection_config=config,
        graph_data=graph,
        transform_field_chunk_size=5,
    )
    pred, target = (t.bfloat16() for t in _data((1, 13), batch=2, time=2, ensemble=2, variables=3))
    project = loss.projector.forward
    gather = loss._flat_fields
    chunks = []

    def checked_gather(values: torch.Tensor, indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        assert indices[0].numel() <= 5
        fields = gather(values, indices)
        assert fields.shape == (indices[0].numel(), values.shape[3])
        assert fields.dtype == torch.float32
        return fields

    def checked_project(fields: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        assert fields.shape[0] <= 5
        assert fields.shape[1:] == (13, 1)
        assert fields.dtype == matrix.dtype == torch.float32
        assert matrix.layout == torch.sparse_csr
        chunks.append(fields.shape[0])
        return project(fields, matrix)

    def no_dense(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Projection must not densify its sparse matrix")

    monkeypatch.setattr(loss.projector, "forward", checked_project)
    monkeypatch.setattr(loss, "_flat_fields", checked_gather)
    monkeypatch.setattr(torch.Tensor, "to_dense", no_dense)
    assert torch.isfinite(loss(pred, target))
    assert chunks == [5, 5] * 4 + [4, 4]


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("negative", "finite and nonnegative"),
        ("nan", "finite and nonnegative"),
        ("empty_row", "positive support"),
        ("destination_shape", "destination point count"),
        ("unnormalized", "rows summing to one"),
    ],
)
def test_projection_invalid_matrices(kind: str, message: str) -> None:
    matrix = _projection_matrix()
    options = {}
    if kind == "negative":
        matrix = -matrix
    elif kind == "nan":
        matrix[0, 0] = float(kind)
    elif kind == "empty_row":
        matrix[0] = 0
    elif kind == "destination_shape":
        matrix = matrix[:-1]
    elif kind == "unnormalized":
        matrix = 2 * matrix
        options["row_normalize"] = False
    graph, config = _projection_graph(matrix)
    with pytest.raises(ValueError, match=message):
        DTCWT2DLoss([6, 8], J=2, projection_config={**config, **options}, graph_data=graph)


@pytest.mark.parametrize(
    "options",
    [
        {"projection_config": {}},
        {"projection_config": {"matrix_path": "unused.npz", "edges_name": ("a", "to", "b")}},
        {"projection_config": {"num_nearest_neighbours": 0, "sigma": 1, "grid": "O16"}},
        {"projection_config": {"num_nearest_neighbours": 1, "sigma": 0, "grid": "O16"}},
    ],
)
def test_projection_schema_rejects_invalid_options(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python(
            {"_target_": "anemoi.training.losses.DTCWT2DLoss", "grid_shape": [6, 8], **options},
        )


def test_nan_handling_options_are_accepted_by_constructor_and_factory() -> None:
    strict_config = {
        "_target_": "anemoi.training.losses.DTCWT2DLoss",
        "grid_shape": [6, 8],
        "ignore_nans": True,
    }
    assert DTCWT2DLoss([6, 8], ignore_nans=True).ignore_nans
    validated = TypeAdapter(LossSchemas).validate_python(strict_config)
    assert isinstance(validated, DTCWT2DLossSchema)
    assert validated.ignore_nans

    pooled_config = {
        "_target_": "anemoi.training.losses.DTCWT2DLoss",
        "grid_shape": [6, 8],
        "nan_handling": "pooled",
    }
    validated = TypeAdapter(LossSchemas).validate_python(pooled_config)
    assert isinstance(validated, DTCWT2DLossSchema)
    assert validated.nan_handling == "pooled"

    loss = get_loss_function(OmegaConf.create(pooled_config))
    assert isinstance(loss, DTCWT2DLoss)
    assert loss.ignore_nans
    assert loss.nan_handling == "pooled"

    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python({**pooled_config, "nan_handling": "fractional"})


def _distributed_projection_worker(rank: int, rendezvous: str, variables: int, nonfinite: bool) -> None:
    """Compare sharded projection scoring, including nonfinite inputs on one rank."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=60),
    )
    try:
        matrix = _projection_matrix()
        graph, config = _projection_graph(matrix)
        loss = DTCWT2DLoss(
            [6, 8],
            J=2,
            projection_config=config,
            graph_data=graph,
            transform_field_chunk_size=1,
        )
        loss.add_scaler(4, torch.arange(1, variables + 1).float(), name="variable")
        pred, target = _data((1, 13), variables=variables, ensemble=2)
        sizes = [5, 8]
        start = sum(sizes[:rank])
        shard_slice = slice(start, start + sizes[rank])
        local_pred = pred[:, :, :, shard_slice].clone().requires_grad_()
        local_target = target[:, :, :, shard_slice].clone()
        if nonfinite and rank == 0:
            local_target[..., 0, -1] = float("nan")
        kwargs = {
            "grid_shard_slice": shard_slice,
            "grid_shard_sizes": sizes,
            "group": torch.distributed.group.WORLD,
        }
        if nonfinite:
            actual = loss(local_pred, local_target, **kwargs)
            assert not torch.isfinite(actual).all()
            torch.distributed.barrier()
        else:
            full_pred = pred.clone().requires_grad_()
            expected = loss(full_pred, target)
            expected.backward()
            actual = loss(local_pred, local_target, **kwargs)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            torch.testing.assert_close(local_pred.grad, full_pred.grad[:, :, :, shard_slice], atol=2e-6, rtol=2e-5)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize(("variables", "nonfinite"), [(3, False), (1, True)])
def test_projection_multiprocess_unequal_grids_with_nonfinite_inputs(
    tmp_path: Path,
    variables: int,
    nonfinite: bool,
) -> None:
    torch.multiprocessing.spawn(
        _distributed_projection_worker,
        args=(str(tmp_path / "projection-rendezvous"), variables, nonfinite),
        nprocs=2,
        join=True,
    )
