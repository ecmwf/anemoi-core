# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from pytest_mock import MockerFixture
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.compile import mark_for_compilation
from anemoi.training.losses import CRPS
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import GraphEdgeCRPSLoss
from anemoi.training.losses import GraphEdgeEnergyScoreLoss
from anemoi.training.losses import GraphEnergyScoreLoss
from anemoi.training.losses import GraphVariogramScoreLoss
from anemoi.training.losses import MultiscaleLossWrapper
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.utils.index_space import IndexSpace


@pytest.fixture
def graph_data() -> HeteroData:
    graph = HeteroData()
    graph["data"].num_nodes = 3
    graph["data", "to", "data"].edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 0, 1, 2, 2],
        ],
    )
    graph["data", "to", "data"].weight = torch.tensor([0.25, 0.75, 1.0, 0.4, 0.6])
    return graph


@pytest.fixture
def loss_graph() -> dict[str, object]:
    return {
        "edges_name": ["data", "to", "data"],
        "edge_weight_attribute": "weight",
        "row_normalize": True,
    }


@pytest.fixture
def score_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.tensor(
        [
            [
                [
                    [[0.0, 1.0], [1.0, 2.0], [2.0, -1.0]],
                    [[1.0, 0.0], [2.0, 3.0], [3.0, 1.0]],
                    [[2.0, 2.0], [1.0, 1.0], [4.0, 0.0]],
                ],
            ],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[[[[1.0, 0.5], [0.0, 2.0], [2.0, 0.0]]]]],
        dtype=torch.float64,
    )
    return pred, target


def _edge_metadata(graph_data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_store = graph_data["data", "to", "data"]
    return edge_store.edge_index[0], edge_store.edge_index[1], edge_store.weight


def _aggregate_edges(
    values: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    out = torch.zeros((*values.shape[:-2], num_nodes, values.shape[-1]), dtype=values.dtype)
    out.index_add_(-2, dst, values * weights.to(dtype=values.dtype).view(1, 1, -1, 1))
    return out


def _graph_energy_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    graph_data: HeteroData,
    *,
    fair: bool,
) -> torch.Tensor:
    src, dst, weights = _edge_metadata(graph_data)
    target = target.squeeze(2)
    ensemble_size = pred.shape[2]

    def neighbourhood_norm(values: torch.Tensor) -> torch.Tensor:
        squared = values[..., src, :].square()
        aggregated = _aggregate_edges(squared, dst, weights, graph_data["data"].num_nodes)
        return torch.sqrt(aggregated)

    obs = neighbourhood_norm(pred - target.unsqueeze(2)).mean(dim=2)
    pair_sum = torch.zeros_like(obs)
    for i in range(ensemble_size):
        for j in range(i + 1, ensemble_size):
            pair_sum += neighbourhood_norm(pred[:, :, i] - pred[:, :, j])
    coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if fair else 1.0 / ensemble_size**2
    return obs - coefficient * pair_sum


def _graph_variogram_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    graph_data: HeteroData,
    *,
    fair: bool,
    p: float,
) -> torch.Tensor:
    src, dst, weights = _edge_metadata(graph_data)
    target = target.squeeze(2)
    ensemble_size = pred.shape[2]

    obs = torch.abs(target[:, :, src] - target[:, :, dst]).pow(p)
    members = torch.abs(pred[:, :, :, src] - pred[:, :, :, dst]).pow(p)
    member_mean = members.mean(dim=2)
    if fair:
        cross_sum = torch.zeros_like(obs)
        for i in range(ensemble_size):
            for j in range(i):
                cross_sum += members[:, :, i] * members[:, :, j]
        edge_score = obs.square() - 2.0 * obs * member_mean
        edge_score += 2.0 * cross_sum / (ensemble_size * (ensemble_size - 1))
    else:
        edge_score = (member_mean - obs).square()
    return _aggregate_edges(edge_score, dst, weights, graph_data["data"].num_nodes)


def _graph_edge_crps_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    graph_data: HeteroData,
    *,
    alpha: float,
) -> torch.Tensor:
    src, dst, weights = _edge_metadata(graph_data)
    target = target.squeeze(2)
    ensemble_size = pred.shape[2]
    obs_edge = target[:, :, src] - target[:, :, dst]
    member_edges = pred[:, :, :, src] - pred[:, :, :, dst]
    obs_term = torch.abs(member_edges - obs_edge.unsqueeze(2)).mean(dim=2)
    pair_sum = torch.zeros_like(obs_term)
    for i in range(ensemble_size):
        for j in range(i + 1, ensemble_size):
            pair_sum += torch.abs(member_edges[:, :, i] - member_edges[:, :, j])
    coefficient = (1.0 - (1.0 - alpha) / ensemble_size) / (ensemble_size * (ensemble_size - 1))
    return _aggregate_edges(obs_term - coefficient * pair_sum, dst, weights, graph_data["data"].num_nodes)


def _graph_edge_energy_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    graph_data: HeteroData,
    *,
    fair: bool,
) -> torch.Tensor:
    src, dst, weights = _edge_metadata(graph_data)
    target = target.squeeze(2)
    ensemble_size = pred.shape[2]

    def edge_norm(values: torch.Tensor) -> torch.Tensor:
        squared = values.square()
        aggregated = _aggregate_edges(squared, dst, weights, graph_data["data"].num_nodes)
        return torch.sqrt(aggregated)

    obs_edge = target[:, :, src] - target[:, :, dst]
    member_edges = pred[:, :, :, src] - pred[:, :, :, dst]
    obs = edge_norm(member_edges - obs_edge.unsqueeze(2)).mean(dim=2)
    pair_sum = torch.zeros_like(obs)
    for i in range(ensemble_size):
        for j in range(i + 1, ensemble_size):
            pair_sum += edge_norm(member_edges[:, :, i] - member_edges[:, :, j])
    coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if fair else 1.0 / ensemble_size**2
    return obs - coefficient * pair_sum


@pytest.mark.parametrize(
    ("loss_factory", "reference"),
    [
        pytest.param(
            lambda graph, definition: GraphEnergyScoreLoss(
                graph_data=graph,
                loss_graph=definition,
                fair=True,
            ),
            lambda pred, target, graph: _graph_energy_reference(pred, target, graph, fair=True),
            id="energy",
        ),
        pytest.param(
            lambda graph, definition: GraphVariogramScoreLoss(
                graph_data=graph,
                loss_graph=definition,
                fair=True,
                p=1.3,
            ),
            lambda pred, target, graph: _graph_variogram_reference(pred, target, graph, fair=True, p=1.3),
            id="variogram",
        ),
        pytest.param(
            lambda graph, definition: GraphEdgeCRPSLoss(
                graph_data=graph,
                loss_graph=definition,
                alpha=0.7,
            ),
            lambda pred, target, graph: _graph_edge_crps_reference(pred, target, graph, alpha=0.7),
            id="edge-crps",
        ),
        pytest.param(
            lambda graph, definition: GraphEdgeEnergyScoreLoss(
                graph_data=graph,
                loss_graph=definition,
                fair=False,
            ),
            lambda pred, target, graph: _graph_edge_energy_reference(pred, target, graph, fair=False),
            id="edge-energy",
        ),
    ],
)
def test_graph_scores_match_reference(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_factory: Callable[[HeteroData, dict[str, object]], BaseLoss],
    reference: Callable[[torch.Tensor, torch.Tensor, HeteroData], torch.Tensor],
) -> None:
    pred, target = score_inputs
    loss = loss_factory(graph_data, loss_graph)

    graph_tensors = loss._graph_kernel_tensors(pred)
    actual = loss._compute_local_score_tensor(
        pred,
        target.squeeze(2),
        *graph_tensors,
    )
    expected = reference(pred, target, graph_data)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("ignore_nans", [False, True])
def test_graph_edge_energy_centering_preserves_spatial_offset_invariance(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    ignore_nans: bool,
) -> None:
    prediction, target = score_inputs
    prediction = prediction.float()
    target = target.float()
    if ignore_nans:
        prediction[:, :, :, 1, 0] = torch.nan

    member_offsets = torch.tensor([2**16, -(2**17), 2**18], dtype=prediction.dtype).view(1, 1, -1, 1, 1)
    shifted_prediction = prediction + member_offsets
    loss = GraphEdgeEnergyScoreLoss(
        graph_data=graph_data,
        loss_graph=loss_graph,
        ignore_nans=ignore_nans,
    )

    score = loss(prediction, target, squash=False)
    shifted_score = loss(shifted_prediction, target, squash=False)

    assert torch.isfinite(score).all()
    assert torch.isfinite(shifted_score).all()
    torch.testing.assert_close(shifted_score, score)


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
def test_graph_scores_expand_the_graph_across_batch(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_cls: type[BaseLoss],
) -> None:
    pred, target = score_inputs
    pred = torch.cat((pred, pred + 0.25), dim=0)
    target = torch.cat((target, target - 0.5), dim=0)
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    batched = loss(pred, target, squash=False)
    separate = torch.stack(
        [loss(pred[i : i + 1], target[i : i + 1], squash=False) for i in range(pred.shape[0])],
    ).mean(dim=0)

    torch.testing.assert_close(batched, separate)


def test_graph_provider_reuses_one_csr_matrix_across_the_batch(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
) -> None:
    loss = GraphEnergyScoreLoss(graph_data=graph_data, loss_graph=loss_graph)
    matrix_one = loss.graph_provider.get_edges(batch_size=1)
    matrix_two = loss.graph_provider.get_edges(batch_size=2)

    assert matrix_one is matrix_two
    assert matrix_two.layout == torch.sparse_csr
    assert matrix_two.shape == (3, 3)
    assert matrix_two.values().numel() == graph_data["data", "to", "data"].num_edges


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
def test_all_graph_scores_use_csr_graphs(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    loss_cls: type[BaseLoss],
) -> None:
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    assert loss.graph_provider.get_edges().layout == torch.sparse_csr


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
def test_graph_scores_have_finite_gradients(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_cls: type[BaseLoss],
) -> None:
    pred, target = score_inputs
    pred = pred.clone().requires_grad_()
    target = target.clone().requires_grad_()
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    result = loss(pred, target)
    result.backward()

    assert result.ndim == 0
    assert pred.grad is not None
    assert target.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(target.grad).all()


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
@pytest.mark.parametrize("ignore_nans", [False, True])
def test_graph_scores_align_mixed_input_dtypes(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_cls: type[BaseLoss],
    ignore_nans: bool,
) -> None:
    prediction_values, target_values = score_inputs
    prediction_values = prediction_values.to(dtype=torch.float32)
    target_values = target_values.to(dtype=torch.float64)
    if ignore_nans:
        prediction_values[0, 0, 0, 0, 0] = torch.nan
        target_values[0, 0, 0, 1, 1] = torch.inf

    mixed_prediction = prediction_values.clone().requires_grad_()
    mixed_target = target_values.clone().requires_grad_()
    mixed_loss = loss_cls(
        graph_data=graph_data,
        loss_graph=loss_graph,
        ignore_nans=ignore_nans,
    )
    mixed_output = mixed_loss(mixed_prediction, mixed_target, squash=False)
    mixed_output.sum().backward()

    reference_prediction = prediction_values.double().requires_grad_()
    reference_target = target_values.clone().requires_grad_()
    reference_loss = loss_cls(
        graph_data=graph_data,
        loss_graph=loss_graph,
        ignore_nans=ignore_nans,
    )
    reference_output = reference_loss(reference_prediction, reference_target, squash=False)
    reference_output.sum().backward()

    assert mixed_output.dtype == torch.float64
    torch.testing.assert_close(mixed_output, reference_output)
    assert mixed_prediction.grad is not None
    assert reference_prediction.grad is not None
    torch.testing.assert_close(
        mixed_prediction.grad,
        reference_prediction.grad.to(dtype=mixed_prediction.dtype),
    )
    assert mixed_target.grad is not None
    assert reference_target.grad is not None
    torch.testing.assert_close(mixed_target.grad, reference_target.grad)
    assert torch.isfinite(mixed_output).all()
    assert torch.isfinite(mixed_prediction.grad).all()
    assert torch.isfinite(mixed_target.grad).all()


@pytest.mark.parametrize(
    ("prediction_dtype", "target_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_graph_scores_reject_unsupported_input_dtypes(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    prediction_dtype: torch.dtype,
    target_dtype: torch.dtype,
) -> None:
    prediction, target = score_inputs
    loss = GraphEnergyScoreLoss(graph_data=graph_data, loss_graph=loss_graph)

    with pytest.raises(TypeError, match="Graph score inputs must be float32 or float64") as exc_info:
        loss(prediction.to(dtype=prediction_dtype), target.to(dtype=target_dtype))

    assert f"prediction dtype {prediction_dtype}" in str(exc_info.value)
    assert f"target dtype {target_dtype}" in str(exc_info.value)


def test_compile_config_uses_nested_graph_score_training_hook(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "anemoi.models.utils.compile._meets_library_versions_for_compile",
        return_value=True,
    )
    compile_mock = mocker.patch(
        "anemoi.training.losses.graph_score_base.torch.compile",
        return_value=mocker.Mock(),
    )
    graph_loss = GraphEdgeCRPSLoss(graph_data=graph_data, loss_graph=loss_graph)
    score_kernel = graph_loss._compute_local_score_tensor
    combined_loss = CombinedLoss(graph_loss)
    model = torch.nn.Module()
    model.add_module("loss", torch.nn.ModuleDict({"data": combined_loss}))
    compile_config = OmegaConf.create(
        [
            {
                "module": "anemoi.training.losses.GraphEdgeCRPSLoss",
                "options": {"dynamic": False},
            },
        ],
    )

    mark_for_compilation(model, compile_config)

    compile_mock.assert_called_once_with(score_kernel, dynamic=False)
    assert graph_loss._compute_local_score_tensor is compile_mock.return_value


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the compiled CSR check")
@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
def test_graph_scores_compile_with_checkpointing_and_nans(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_cls: type[BaseLoss],
) -> None:
    prediction_values, target_values = score_inputs
    prediction_values = prediction_values.float()
    target_values = target_values.float()
    prediction_values[0, 0, 0, 0, 0] = torch.nan
    target_values[0, 0, 0, 1, 1] = torch.inf
    results = []

    for compiled in (False, True):
        prediction = prediction_values.cuda().requires_grad_()
        target = target_values.cuda().requires_grad_()
        loss = loss_cls(
            graph_data=graph_data,
            loss_graph=loss_graph,
            ignore_nans=True,
        ).cuda()
        if compiled:
            loss.compile_for_training(dynamic=False)
        output = checkpoint(loss, prediction, target, use_reentrant=False)
        output.backward()
        assert prediction.grad is not None
        assert target.grad is not None
        assert torch.isfinite(output)
        assert torch.isfinite(prediction.grad).all()
        assert torch.isfinite(target.grad).all()
        results.append((output.detach(), prediction.grad.detach(), target.grad.detach()))

    for compiled_tensor, eager_tensor in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(compiled_tensor, eager_tensor, rtol=4.0e-4, atol=4.0e-5)


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
@pytest.mark.parametrize("num_variables", [1, 2])
def test_graph_scores_follow_standard_output_shape_contract(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_cls: type[BaseLoss],
    num_variables: int,
) -> None:
    pred, target = score_inputs
    pred = pred[..., :num_variables]
    target = target[..., :num_variables]
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    scalar_loss = loss(pred, target)
    per_variable_loss = loss(pred, target, squash=False)
    summed_loss = loss(pred, target, squash_mode="sum")

    assert scalar_loss.shape == ()
    assert per_variable_loss.shape == (num_variables,)
    torch.testing.assert_close(scalar_loss, per_variable_loss.mean())
    torch.testing.assert_close(summed_loss, per_variable_loss.sum())


@pytest.mark.parametrize(("fair", "alpha"), [(True, 1.0), (False, 0.0)])
@pytest.mark.parametrize("squash", [True, False])
def test_pointwise_graph_energy_matches_crps(
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    fair: bool,
    alpha: float,
    squash: bool,
) -> None:
    pred, target = score_inputs

    actual = GraphEnergyScoreLoss(fair=fair)(pred, target, squash=squash)
    expected = CRPS(alpha=alpha)(pred, target, squash=squash)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("alpha", [0.0, 0.7, 1.0])
def test_graph_edge_crps_matches_crps_in_edge_space(alpha: float) -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 2
    graph["data", "to", "data"].edge_index = torch.tensor([[0, 1], [1, 0]])
    definition = {
        "edges_name": ["data", "to", "data"],
        "row_normalize": True,
    }
    pred = torch.tensor(
        [[[[[0.0], [1.0]], [[2.0], [0.0]], [[1.0], [4.0]]]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[[[[1.0], [2.0]]]]], dtype=torch.float64)
    src, dst = graph["data", "to", "data"].edge_index

    actual = GraphEdgeCRPSLoss(
        loss_graph=definition,
        graph_data=graph,
        alpha=alpha,
    )(pred, target)
    edge_pred = pred[..., src, :] - pred[..., dst, :]
    edge_target = target[..., src, :] - target[..., dst, :]
    expected = CRPS(alpha=alpha)(edge_pred, edge_target)

    torch.testing.assert_close(actual, expected)


def test_fair_graph_variogram_matches_hand_calculation() -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 2
    graph["data", "to", "data"].edge_index = torch.tensor([[0, 1], [1, 0]])
    pred = torch.tensor(
        [[[[[0.0], [1.0]], [[0.0], [3.0]]]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[[[[0.0], [2.0]]]]], dtype=torch.float64)
    loss = GraphVariogramScoreLoss(
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "row_normalize": True,
        },
        graph_data=graph,
        p=1.0,
        fair=True,
    )

    # Each directed edge has observed variogram 2 and member variograms 1 and
    # 3. Its fair score is 2**2 - 2*2*((1+3)/2) + 1*3 = -1. Two reciprocal
    # edges therefore give a total score of -2.
    torch.testing.assert_close(loss(pred, target), torch.tensor(-2.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
def test_graph_scores_ignore_invalid_edges(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    loss_cls: type[BaseLoss],
) -> None:
    pred, target = score_inputs
    pred = pred.clone()
    pred[:, :, 0, 0, 0] = torch.nan
    pred.requires_grad_()
    loss = loss_cls(
        graph_data=graph_data,
        loss_graph=loss_graph,
        ignore_nans=True,
    )

    result = loss(pred, target)
    result.backward()

    assert torch.isfinite(result)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_graph_definition_applies_and_normalizes_weights(graph_data: HeteroData) -> None:
    graph_data["data", "to", "data"].weight = torch.tensor([2.0, 6.0, 3.0, 4.0, 6.0])
    loss = GraphEnergyScoreLoss(
        graph_data=graph_data,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "edge_weight_attribute": "weight",
            "row_normalize": True,
        },
    )

    matrix = loss.graph_provider.get_edges()
    expected = torch.tensor(
        [
            [0.25, 0.75, 0.0],
            [0.0, 0.0, 1.0],
            [0.4, 0.6, 0.0],
        ],
    )
    torch.testing.assert_close(matrix.to_dense(), expected)


def test_graph_definition_validation_and_provider_normalize_weights_consistently(
    graph_data: HeteroData,
) -> None:
    edge_store = graph_data["data", "to", "data"]
    edge_store.weight = torch.tensor([2.0, 6.0, 3.0, 4.0, 6.0])
    source, destination = edge_store.edge_index
    validation_weights = GraphScoreGraph._row_normalize_weights(
        destination,
        edge_store.weight,
        graph_data["data"].num_nodes,
    )

    loss = GraphEnergyScoreLoss(
        graph_data=graph_data,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "edge_weight_attribute": "weight",
            "row_normalize": True,
        },
    )
    provider_matrix = loss.graph_provider.get_edges().to_dense()

    torch.testing.assert_close(provider_matrix[destination, source], validation_weights)


def test_graph_definition_defaults_to_unnormalized_unit_weights(graph_data: HeteroData) -> None:
    loss = GraphEnergyScoreLoss(
        graph_data=graph_data,
        loss_graph={"edges_name": ["data", "to", "data"]},
    )

    torch.testing.assert_close(loss.graph_provider.get_edges().values(), torch.ones(5))
    assert not loss.graph.row_normalize


def test_graph_definition_applies_source_node_weights(graph_data: HeteroData) -> None:
    graph_data["data"].area = torch.tensor([2.0, 3.0, 4.0])
    loss = GraphVariogramScoreLoss(
        graph_data=graph_data,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "edge_weight_attribute": "weight",
            "src_node_weight_attribute": "area",
        },
    )
    src = graph_data["data", "to", "data"].edge_index[0]
    expected = graph_data["data", "to", "data"].weight * graph_data["data"].area[src]

    expected_matrix = torch.zeros(3, 3)
    dst = graph_data["data", "to", "data"].edge_index[1]
    expected_matrix.index_put_((dst, src), expected, accumulate=True)
    torch.testing.assert_close(loss.graph_provider.get_edges().to_dense(), expected_matrix)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (torch.zeros(5), "positive total weight"),
        (torch.tensor([-0.25, 1.25, 1.0, 0.4, 0.6]), "non-negative"),
        (torch.tensor([torch.nan, 1.0, 1.0, 0.4, 0.6]), "finite real values"),
    ],
)
def test_graph_definition_rejects_invalid_weights(
    graph_data: HeteroData,
    weights: torch.Tensor,
    match: str,
) -> None:
    graph_data["data", "to", "data"].weight = weights

    with pytest.raises(ValueError, match=match):
        GraphEnergyScoreLoss(
            graph_data=graph_data,
            loss_graph={
                "edges_name": ["data", "to", "data"],
                "edge_weight_attribute": "weight",
            },
        )


@pytest.mark.parametrize("loss_cls", [GraphEnergyScoreLoss, GraphEdgeEnergyScoreLoss])
def test_energy_scores_ignore_zero_weight_edges_without_nan_gradients(loss_cls: type[BaseLoss]) -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 2
    graph["data", "to", "data"].edge_index = torch.tensor(
        [
            [0, 1, 1],
            [0, 0, 1],
        ],
    )
    graph["data", "to", "data"].weight = torch.tensor([1.0, 0.0, 1.0])
    pred = torch.tensor(
        [[[[[0.0], [1.0]], [[0.0], [2.0]]]]],
        requires_grad=True,
    )
    target = torch.zeros(1, 1, 1, 2, 1)
    loss = loss_cls(
        graph_data=graph,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "edge_weight_attribute": "weight",
        },
    )

    result = loss(pred, target)
    result.backward()

    assert torch.isfinite(result)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("loss_cls", [GraphEnergyScoreLoss, GraphEdgeEnergyScoreLoss])
def test_energy_scores_have_finite_zero_norm_gradients(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    loss_cls: type[BaseLoss],
) -> None:
    prediction = torch.zeros(2, 2, 4, 3, 2, requires_grad=True)
    target = torch.zeros(2, 2, 1, 3, 2, requires_grad=True)
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    result = loss(prediction, target)
    result.backward()

    torch.testing.assert_close(result, torch.tensor(0.0))
    assert prediction.grad is not None
    assert target.grad is not None
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
    torch.testing.assert_close(target.grad, torch.zeros_like(target))


def test_graph_scores_require_graph_to_match_forecast_grid() -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 2
    graph["data", "to", "data"].edge_index = torch.tensor([[0, 1], [0, 1]])
    loss = GraphEnergyScoreLoss(
        graph_data=graph,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "row_normalize": True,
        },
    )

    with pytest.raises(ValueError, match="does not match the forecast grid"):
        loss(torch.zeros(1, 1, 2, 3, 1), torch.zeros(1, 1, 1, 3, 1))


def test_graph_scores_require_one_node_index_space() -> None:
    graph = HeteroData()
    graph["source"].num_nodes = 2
    graph["destination"].num_nodes = 2
    graph["source", "to", "destination"].edge_index = torch.tensor([[0, 1], [0, 1]])

    with pytest.raises(ValueError, match="same node type"):
        GraphEdgeCRPSLoss(
            graph_data=graph,
            loss_graph={
                "edges_name": ["source", "to", "destination"],
                "row_normalize": True,
            },
        )


@pytest.mark.parametrize("loss_cls", [GraphEnergyScoreLoss, GraphEdgeEnergyScoreLoss])
def test_energy_scores_stay_finite_for_large_values(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    loss_cls: type[BaseLoss],
) -> None:
    pred = torch.tensor(
        [[[[[1.0e20], [-1.0e20], [0.5e20]], [[1.5e20], [-0.5e20], [1.0e20]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[[[0.25e20], [-0.25e20], [0.0]]]]], dtype=torch.float32)
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    result = loss(pred, target)
    result.backward()

    assert torch.isfinite(result)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_graph_scores_require_current_tensor_layout(graph_data: HeteroData, loss_graph: dict[str, object]) -> None:
    loss = GraphEnergyScoreLoss(graph_data=graph_data, loss_graph=loss_graph)

    with pytest.raises(ValueError, match="singleton target ensemble"):
        loss(torch.zeros(1, 1, 2, 3, 1), torch.zeros(1, 1, 2, 3, 1))


def test_graph_score_factory_and_nested_losses(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    combined = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": [],
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.GraphEnergyScoreLoss",
                        "scalers": [],
                        "loss_graph": loss_graph,
                    },
                    {
                        "_target_": "anemoi.training.losses.GraphEdgeCRPSLoss",
                        "scalers": [],
                        "loss_graph": loss_graph,
                    },
                ],
            },
        ),
        graph_data=graph_data,
        data_node_name="data",
    )
    multiscale = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                "weights": [1.0],
                "multiscale_config": None,
                "per_scale_loss": {
                    "_target_": "anemoi.training.losses.GraphVariogramScoreLoss",
                    "scalers": [],
                    "loss_graph": loss_graph,
                },
            },
        ),
        graph_data=graph_data,
        data_node_name="data",
    )
    pred, target = score_inputs

    assert isinstance(combined, CombinedLoss)
    assert all(loss.graph_provider is not None for loss in combined.losses)
    assert torch.isfinite(combined(pred, target))
    assert isinstance(multiscale, MultiscaleLossWrapper)
    assert multiscale.needs_shard_layout_info
    assert torch.isfinite(multiscale(pred, target)).all()


def test_combined_multiscale_crps_with_filtered_multiscale_edge_crps(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {"tp": 0, "t2m": 1, "msl": 2},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": [],
                "loss_weights": [1.0, 0.1],
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                        "weights": [0.4, 0.6],
                        "multiscale_config": {"loss_matrices": [None, None]},
                        "per_scale_loss": {
                            "_target_": "anemoi.training.losses.CRPS",
                            "scalers": [],
                            "alpha": 0.95,
                        },
                    },
                    {
                        "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                        "weights": [0.4, 0.6],
                        "multiscale_config": {"loss_matrices": [None, None]},
                        "per_scale_loss": {
                            "_target_": "anemoi.training.losses.GraphEdgeCRPSLoss",
                            "scalers": [],
                            "alpha": 1.0,
                            "loss_graph": loss_graph,
                            "predicted_variables": ["tp"],
                            "target_variables": ["tp"],
                        },
                    },
                ],
            },
        ),
        data_indices=data_indices,
        graph_data=graph_data,
        data_node_name="data",
    )
    pred, target = score_inputs
    pred = torch.cat((pred, pred[..., :1] + 0.5), dim=-1).requires_grad_()
    target = torch.cat((target, target[..., :1] - 0.5), dim=-1)
    loss_kwargs = {
        "pred_layout": IndexSpace.MODEL_OUTPUT,
        "target_layout": IndexSpace.DATA_OUTPUT,
    }

    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.losses[0], MultiscaleLossWrapper)
    assert isinstance(loss.losses[0].loss, LossVariableMapper)
    assert isinstance(loss.losses[0].loss.loss, CRPS)
    assert isinstance(loss.losses[1], MultiscaleLossWrapper)
    assert isinstance(loss.losses[1].loss, LossVariableMapper)
    assert isinstance(loss.losses[1].loss.loss, GraphEdgeCRPSLoss)

    scalar_loss = loss(pred, target, **loss_kwargs)
    per_variable_loss = loss(pred, target, squash=False, **loss_kwargs)
    primary_scalar = loss.losses[0](pred, target, **loss_kwargs)
    edge_scalar = loss.losses[1](pred, target, **loss_kwargs)
    primary_per_variable = loss.losses[0](pred, target, squash=False, **loss_kwargs)
    edge_per_variable = loss.losses[1](pred, target, squash=False, **loss_kwargs)

    torch.testing.assert_close(scalar_loss, primary_scalar + 0.1 * edge_scalar)
    torch.testing.assert_close(per_variable_loss, primary_per_variable + 0.1 * edge_per_variable)
    torch.testing.assert_close(edge_per_variable[1:], torch.zeros(2, dtype=pred.dtype))

    scalar_loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_graph_score_schemas_accept_direct_and_combined_configs(loss_graph: dict[str, object]) -> None:
    direct = TypeAdapter(LossSchemas).validate_python(
        {
            "_target_": "anemoi.training.losses.GraphVariogramScoreLoss",
            "scalers": [],
            "loss_graph": loss_graph,
            "p": 1.5,
        },
    )
    combined = CombinedLossSchema.model_validate(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "scalers": [],
            "losses": [
                {
                    "_target_": "anemoi.training.losses.GraphEdgeEnergyScoreLoss",
                    "scalers": [],
                    "loss_graph": loss_graph,
                },
            ],
        },
    )

    assert direct.target_ == "anemoi.training.losses.GraphVariogramScoreLoss"
    assert direct.loss_graph.validate_row_sums is False
    assert combined.losses[0].target_ == "anemoi.training.losses.GraphEdgeEnergyScoreLoss"
    assert combined.losses[0].loss_graph.validate_row_sums is False


def test_graph_score_uses_current_sharding_contract(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
    mocker: MockerFixture,
) -> None:
    pred, target = score_inputs
    group = object()
    all_to_all = mocker.patch(
        "anemoi.training.losses.graph_score_base.all_to_all_transpose",
        side_effect=lambda tensor, *_args: tensor,
    )
    mocker.patch(
        "anemoi.training.losses.graph_score_base.get_shard_sizes",
        return_value=[1],
    )
    mocker.patch("anemoi.training.losses.base.reduce_tensor", side_effect=lambda tensor, _group: tensor)
    loss = GraphEnergyScoreLoss(graph_data=graph_data, loss_graph=loss_graph)

    result = loss(
        pred,
        target,
        grid_shard_slice=slice(0, 3),
        grid_shard_sizes=[3],
        grid_dim=3,
        group=group,
    )

    assert result.ndim == 0
    assert all_to_all.call_count == 3
    assert loss.needs_shard_layout_info
