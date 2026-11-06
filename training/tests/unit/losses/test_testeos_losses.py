# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""NaNAwareMSELoss and GraphLaplacianSmoothnessLoss, against the feat/testeos formulas they port."""

import pytest
import torch
from pydantic import TypeAdapter
from torch_geometric.data import HeteroData

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.training.losses import GraphLaplacianSmoothnessLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import NaNAwareMSELoss
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.utils.enums import TensorDim

GRIDDED = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
SHAPE = (2, 1, 1, 12, 3)  # batch, time, ensemble, grid, variables


def _gridded(data: torch.Tensor) -> GriddedSource:
    return GriddedSource(
        name="data",
        data=data,
        variables=[f"v{i}" for i in range(data.shape[-1])],
        statistics={},
        coordinates=torch.zeros(data.shape[3], 2),
        layout=GRIDDED,
    )


def _node_weights(num_points: int) -> torch.Tensor:
    weights = torch.rand(num_points, generator=torch.Generator().manual_seed(5)) + 0.5
    return weights / weights.sum()


def _graph(num_points: int) -> HeteroData:
    generator = torch.Generator().manual_seed(7)
    graph = HeteroData()
    graph["data"].num_nodes = num_points
    src = torch.randint(0, num_points, (40,), generator=generator)
    dst = torch.randint(0, num_points, (40,), generator=generator)
    graph["data", "to", "data"].edge_index = torch.stack([src, dst])
    return graph


def test_nan_aware_mse_matches_testeos_density_weighting() -> None:
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(SHAPE, generator=generator)
    target = torch.randn(SHAPE, generator=generator)
    target[torch.rand(SHAPE, generator=generator) < 0.6] = float("nan")
    target[0, ..., 1] = float("nan")  # a variable missing everywhere in a sample
    weights = _node_weights(SHAPE[3])

    loss = NaNAwareMSELoss(ignore_nans=True)
    loss.add_scaler(TensorDim.GRID, weights, name="node_weights")
    result = loss(_gridded(pred), _gridded(target), squash=False)

    # testeos: mask, weight each variable by grid / valid points, node-weight, sum over the grid,
    # average over batch, time and ensemble.
    missing = torch.isnan(target)
    density = SHAPE[3] / (SHAPE[3] - missing.sum(dim=3, keepdim=True)).clamp(min=1)
    error = torch.square(pred.masked_fill(missing, 0) - target.masked_fill(missing, 0)) * density
    expected = (error * weights.view(1, 1, 1, -1, 1)).sum(dim=3).mean(dim=(0, 1, 2)).flatten()
    torch.testing.assert_close(result, expected)


def test_nan_aware_mse_differs_from_mse_by_the_valid_fraction() -> None:
    pred = torch.zeros((1, 1, 1, 4, 1))
    target = torch.tensor([1.0, float("nan"), float("nan"), float("nan")]).view(1, 1, 1, 4, 1)
    weights = torch.full((4,), 0.25)
    nan_aware, plain = NaNAwareMSELoss(ignore_nans=True), MSELoss(ignore_nans=True)
    for loss in (nan_aware, plain):
        loss.add_scaler(TensorDim.GRID, weights, name="node_weights")

    assert nan_aware(_gridded(pred), _gridded(target)).item() == pytest.approx(1.0)
    assert plain(_gridded(pred), _gridded(target)).item() == pytest.approx(0.25)


def test_graph_smoothness_matches_testeos_laplacian() -> None:
    graph = _graph(SHAPE[3])
    pred = torch.randn(SHAPE, generator=torch.Generator().manual_seed(1))
    weights = _node_weights(SHAPE[3])

    loss = GraphLaplacianSmoothnessLoss(graph_data=graph, data_node_name="data", penalty_weight=2.0)
    loss.add_scaler(TensorDim.GRID, weights, name="node_weights")
    result = loss(_gridded(pred), _gridded(torch.full(SHAPE, float("nan"))), squash=False)

    # testeos: neighbour_sum[src] += pred[dst]; degree = edge count per src (min 1).
    src, dst = graph["data", "to", "data"].edge_index
    neighbour_sum = torch.zeros_like(pred)
    neighbour_sum.scatter_add_(3, src.view(1, 1, 1, -1, 1).expand(*SHAPE[:3], -1, SHAPE[4]), pred[:, :, :, dst])
    degree = torch.zeros(SHAPE[3]).scatter_add_(0, src, torch.ones(src.shape[0])).clamp(min=1)
    laplacian_sq = (pred - neighbour_sum / degree.view(1, 1, 1, -1, 1)) ** 2
    expected = (2.0 * laplacian_sq * weights.view(1, 1, 1, -1, 1)).sum(dim=3).mean(dim=(0, 1, 2)).flatten()
    torch.testing.assert_close(result, expected)


def test_graph_smoothness_needs_data_to_data_edges() -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 3
    with pytest.raises(ValueError, match="edges of the training graph"):
        GraphLaplacianSmoothnessLoss(graph_data=graph, data_node_name="data")


def test_graph_smoothness_schema() -> None:
    config = {
        "_target_": "anemoi.training.losses.GraphLaplacianSmoothnessLoss",
        "scalers": ["node_weights"],
        "penalty_weight": 1.0,
        "predicted_variables": ["z_500"],
    }
    TypeAdapter(LossSchemas).validate_python(config)
