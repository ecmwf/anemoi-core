# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Temporary implementation-level checks for the CSR graph-score migration.

Delete this file together with ``anemoi.training.losses.deprecated.graph_scores``
once the legacy edge-materializing implementation is removed.
"""

from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.training.losses import GraphEdgeCRPSLoss
from anemoi.training.losses import GraphEdgeEnergyScoreLoss
from anemoi.training.losses import GraphEnergyScoreLoss
from anemoi.training.losses import GraphVariogramScoreLoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.deprecated.graph_scores import LegacyGraphEdgeCRPSLoss
from anemoi.training.losses.deprecated.graph_scores import LegacyGraphEdgeEnergyScoreLoss
from anemoi.training.losses.deprecated.graph_scores import LegacyGraphEnergyScoreLoss
from anemoi.training.losses.deprecated.graph_scores import LegacyGraphVariogramScoreLoss

LossPairFactory = Callable[[HeteroData, dict[str, object], bool], tuple[BaseLoss, BaseLoss]]


def _make_graph() -> HeteroData:
    num_nodes = 7
    destinations = torch.arange(num_nodes).repeat_interleave(3)
    offsets = torch.tensor([-2, -1, 1]).repeat(num_nodes)
    sources = (destinations + offsets).remainder(num_nodes)
    graph = HeteroData()
    graph["data"].num_nodes = num_nodes
    graph["data"].area = torch.linspace(0.8, 1.4, num_nodes)
    graph["data", "to", "data"].edge_index = torch.stack((sources, destinations))
    graph["data", "to", "data"].weight = torch.tensor([0.0, 0.35, 0.65]).repeat(num_nodes)
    return graph


def _definition(row_normalize: bool) -> dict[str, object]:
    return {
        "edges_name": ["data", "to", "data"],
        "edge_weight_attribute": "weight",
        "src_node_weight_attribute": "area",
        "row_normalize": row_normalize,
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260728)
    prediction = torch.randn(2, 2, 4, 7, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 2, 1, 7, 3, generator=generator, dtype=torch.float64)
    return prediction, target


def _evaluate(
    loss: BaseLoss,
    prediction_values: torch.Tensor,
    target_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction_values.clone().requires_grad_()
    target = target_values.clone().requires_grad_()
    output = loss(prediction, target, squash=False)
    output.sum().backward()
    assert prediction.grad is not None
    assert target.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(prediction.grad).all()
    assert torch.isfinite(target.grad).all()
    return output.detach(), prediction.grad.detach(), target.grad.detach()


def _energy_pair(
    graph: HeteroData,
    definition: dict[str, object],
    ignore_nans: bool,
    *,
    fair: bool,
) -> tuple[BaseLoss, BaseLoss]:
    kwargs = {
        "graph_data": graph,
        "loss_graph": definition,
        "fair": fair,
        "ignore_nans": ignore_nans,
    }
    return GraphEnergyScoreLoss(**kwargs), LegacyGraphEnergyScoreLoss(**kwargs)


def _edge_energy_pair(
    graph: HeteroData,
    definition: dict[str, object],
    ignore_nans: bool,
    *,
    fair: bool,
) -> tuple[BaseLoss, BaseLoss]:
    kwargs = {
        "graph_data": graph,
        "loss_graph": definition,
        "fair": fair,
        "ignore_nans": ignore_nans,
    }
    return GraphEdgeEnergyScoreLoss(**kwargs), LegacyGraphEdgeEnergyScoreLoss(**kwargs)


def _edge_crps_pair(
    graph: HeteroData,
    definition: dict[str, object],
    ignore_nans: bool,
    *,
    alpha: float,
) -> tuple[BaseLoss, BaseLoss]:
    kwargs = {
        "graph_data": graph,
        "loss_graph": definition,
        "alpha": alpha,
        "ignore_nans": ignore_nans,
    }
    return GraphEdgeCRPSLoss(**kwargs), LegacyGraphEdgeCRPSLoss(**kwargs)


def _variogram_pair(
    graph: HeteroData,
    definition: dict[str, object],
    ignore_nans: bool,
    *,
    fair: bool,
) -> tuple[BaseLoss, BaseLoss]:
    kwargs = {
        "graph_data": graph,
        "loss_graph": definition,
        "p": 1.3,
        "fair": fair,
        "ignore_nans": ignore_nans,
    }
    return GraphVariogramScoreLoss(**kwargs), LegacyGraphVariogramScoreLoss(**kwargs)


LOSS_PAIR_FACTORIES = [
    pytest.param(lambda g, d, n: _energy_pair(g, d, n, fair=True), id="fair-energy"),
    pytest.param(lambda g, d, n: _energy_pair(g, d, n, fair=False), id="empirical-energy"),
    pytest.param(lambda g, d, n: _edge_energy_pair(g, d, n, fair=True), id="fair-edge-energy"),
    pytest.param(lambda g, d, n: _edge_energy_pair(g, d, n, fair=False), id="empirical-edge-energy"),
    pytest.param(lambda g, d, n: _edge_crps_pair(g, d, n, alpha=0.0), id="empirical-edge-crps"),
    pytest.param(lambda g, d, n: _edge_crps_pair(g, d, n, alpha=0.7), id="almost-fair-edge-crps"),
    pytest.param(lambda g, d, n: _edge_crps_pair(g, d, n, alpha=1.0), id="fair-edge-crps"),
    pytest.param(lambda g, d, n: _variogram_pair(g, d, n, fair=True), id="fair-variogram"),
    pytest.param(lambda g, d, n: _variogram_pair(g, d, n, fair=False), id="empirical-variogram"),
]


@pytest.mark.parametrize("loss_pair_factory", LOSS_PAIR_FACTORIES)
@pytest.mark.parametrize("row_normalize", [False, True])
def test_csr_scores_match_legacy_outputs_and_gradients(
    loss_pair_factory: LossPairFactory,
    row_normalize: bool,
) -> None:
    graph = _make_graph()
    prediction, target = _inputs()
    csr_loss, legacy_loss = loss_pair_factory(graph, _definition(row_normalize), False)

    csr_result = _evaluate(csr_loss, prediction, target)
    legacy_result = _evaluate(legacy_loss, prediction, target)

    for csr_tensor, legacy_tensor in zip(csr_result, legacy_result, strict=True):
        torch.testing.assert_close(csr_tensor, legacy_tensor, rtol=2.0e-5, atol=2.0e-6)


@pytest.mark.parametrize("loss_pair_factory", LOSS_PAIR_FACTORIES)
@pytest.mark.parametrize("row_normalize", [False, True])
def test_csr_scores_match_legacy_nan_semantics(
    loss_pair_factory: LossPairFactory,
    row_normalize: bool,
) -> None:
    graph = _make_graph()
    prediction, target = _inputs()
    prediction[0, 0, 0, 2, 1] = torch.nan
    prediction[1, 1, 3, 5, 2] = torch.inf
    target[0, 1, 0, 4, 0] = -torch.inf
    incoming_to_zero = graph["data", "to", "data"].edge_index[0][graph["data", "to", "data"].edge_index[1] == 0]
    target[1, 0, 0, incoming_to_zero, 1] = torch.nan
    csr_loss, legacy_loss = loss_pair_factory(graph, _definition(row_normalize), True)

    csr_result = _evaluate(csr_loss, prediction, target)
    legacy_result = _evaluate(legacy_loss, prediction, target)

    for csr_tensor, legacy_tensor in zip(csr_result, legacy_result, strict=True):
        torch.testing.assert_close(csr_tensor, legacy_tensor, rtol=5.0e-5, atol=5.0e-6)
