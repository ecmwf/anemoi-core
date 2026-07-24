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
from pydantic import TypeAdapter
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CRPS
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import GraphEdgeCRPSLoss
from anemoi.training.losses import GraphEdgeEnergyScoreLoss
from anemoi.training.losses import GraphEnergyScoreLoss
from anemoi.training.losses import GraphVariogramScoreLoss
from anemoi.training.losses import MultiscaleLossWrapper
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
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

    actual = loss._compute_local_score_tensor(pred, target.squeeze(2))
    expected = reference(pred, target, graph_data)

    torch.testing.assert_close(actual, expected)


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
    loss = loss_cls(graph_data=graph_data, loss_graph=loss_graph)

    result = loss(pred, target)
    result.backward()

    assert result.ndim == 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


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
    loss = GraphEnergyScoreLoss(
        graph_data=graph_data,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "edge_weight_attribute": "weight",
            "row_normalize": True,
        },
    )

    row_sums = torch.zeros(3).scatter_add_(0, loss.graph.edge_dst_index, loss.graph.edge_weights)
    torch.testing.assert_close(row_sums, torch.ones(3))


def test_graph_definition_applies_source_node_weights(graph_data: HeteroData) -> None:
    graph_data["data"].area = torch.tensor([2.0, 3.0, 4.0])
    loss = GraphVariogramScoreLoss(
        graph_data=graph_data,
        loss_graph={
            "edges_name": ["data", "to", "data"],
            "edge_weight_attribute": "weight",
            "src_node_weight_attribute": "area",
            "validate_row_sums": False,
        },
    )
    src = graph_data["data", "to", "data"].edge_index[0]
    expected = graph_data["data", "to", "data"].weight * graph_data["data"].area[src]

    torch.testing.assert_close(loss.graph.edge_weights, expected)


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


@pytest.mark.parametrize(
    "loss_cls",
    [GraphEnergyScoreLoss, GraphVariogramScoreLoss, GraphEdgeCRPSLoss, GraphEdgeEnergyScoreLoss],
)
def test_graph_scores_require_graph_to_match_forecast_grid(loss_cls: type[BaseLoss]) -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 2
    graph["data", "to", "data"].edge_index = torch.tensor([[0, 1], [0, 1]])
    loss = loss_cls(
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
    assert all(loss.graph is not None for loss in combined.losses)
    assert torch.isfinite(combined(pred, target))
    assert isinstance(multiscale, MultiscaleLossWrapper)
    assert multiscale.needs_shard_layout_info
    assert torch.isfinite(multiscale(pred, target)).all()


def test_filtered_graph_score_preserves_its_reduction_default(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {"a": 0, "b": 1, "c": 2},
    )
    pred, target = score_inputs
    pred = torch.cat((pred, pred[..., :1] + 0.5), dim=-1)
    target = torch.cat((target, target[..., :1] - 0.5), dim=-1)
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.GraphEnergyScoreLoss",
                "scalers": [],
                "loss_graph": loss_graph,
                "predicted_variables": ["a", "c"],
                "target_variables": ["a", "c"],
            },
        ),
        data_indices=data_indices,
        graph_data=graph_data,
    )
    loss_kwargs = {
        "pred_layout": IndexSpace.MODEL_OUTPUT,
        "target_layout": IndexSpace.DATA_OUTPUT,
    }

    assert isinstance(loss, LossVariableMapper)
    scalar_loss = loss(pred, target, **loss_kwargs)
    per_variable_loss = loss(pred, target, squash=False, **loss_kwargs)
    summed_loss = loss(pred, target, squash_mode="sum", **loss_kwargs)

    assert per_variable_loss.shape == (3,)
    torch.testing.assert_close(per_variable_loss[1], torch.tensor(0.0, dtype=pred.dtype))
    torch.testing.assert_close(scalar_loss, per_variable_loss[[0, 2]].mean())
    torch.testing.assert_close(summed_loss, per_variable_loss.sum())


def test_combined_loss_with_direct_and_multiscale_graph_scores(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    pred, target = score_inputs
    pred = pred.clone().requires_grad_()
    loss = CombinedLoss(
        GraphEnergyScoreLoss(graph_data=graph_data, loss_graph=loss_graph),
        MultiscaleLossWrapper(
            per_scale_loss=GraphEdgeCRPSLoss(graph_data=graph_data, loss_graph=loss_graph),
            weights=[0.4, 0.6],
            multiscale_config={"loss_matrices": [None, None]},
        ),
        loss_weights=(0.25, 0.75),
    )

    scalar_loss = loss(pred, target)
    per_variable_loss = loss(pred, target, squash=False)

    assert scalar_loss.shape == ()
    assert per_variable_loss.shape == (pred.shape[-1],)
    torch.testing.assert_close(scalar_loss, per_variable_loss.mean())
    assert torch.isfinite(scalar_loss)
    assert torch.isfinite(per_variable_loss).all()

    scalar_loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_filtered_combined_loss_with_direct_and_multiscale_graph_scores(
    graph_data: HeteroData,
    loss_graph: dict[str, object],
    score_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {"a": 0, "b": 1, "c": 2},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": [],
                "loss_weights": [0.25, 0.75],
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.GraphEnergyScoreLoss",
                        "scalers": [],
                        "loss_graph": loss_graph,
                        "predicted_variables": ["a", "c"],
                        "target_variables": ["a", "c"],
                    },
                    {
                        "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                        "weights": [0.4, 0.6],
                        "multiscale_config": {"loss_matrices": [None, None]},
                        "per_scale_loss": {
                            "_target_": "anemoi.training.losses.GraphEdgeCRPSLoss",
                            "scalers": [],
                            "loss_graph": loss_graph,
                            "predicted_variables": ["a", "c"],
                            "target_variables": ["a", "c"],
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

    scalar_loss = loss(pred, target, **loss_kwargs)
    per_variable_loss = loss(pred, target, squash=False, **loss_kwargs)

    assert scalar_loss.shape == ()
    assert per_variable_loss.shape == (3,)
    torch.testing.assert_close(per_variable_loss[1], torch.tensor(0.0, dtype=pred.dtype))
    torch.testing.assert_close(scalar_loss, per_variable_loss[[0, 2]].mean())
    assert torch.isfinite(scalar_loss)
    assert torch.isfinite(per_variable_loss).all()

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
    assert combined.losses[0].target_ == "anemoi.training.losses.GraphEdgeEnergyScoreLoss"


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
