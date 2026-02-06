# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multiscale import MultiscaleLossWrapper


@pytest.fixture
def loss_inputs_multiscale() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 1, 2, 4, 2]  # (batch, output_steps, ens, latlon, vars)

    pred = torch.zeros(tensor_shape)
    pred[0, 0, :, 0] = torch.tensor([1.0, 0.0])
    target = torch.zeros([tensor_shape[0], tensor_shape[1], tensor_shape[3], tensor_shape[4]])  # no ensemble dim

    # With only one "grid point" differing by 1 in all
    # variables, the loss should be 1.0

    loss_result = torch.tensor([1.0])
    return pred, target, loss_result


@pytest.fixture
def multiscale_graph_data() -> HeteroData:
    graph = HeteroData()
    graph["data"].num_nodes = 4
    graph["smooth_8x"].num_nodes = 4
    graph["smooth_4x"].num_nodes = 4
    graph["smooth_2x"].num_nodes = 4
    graph["smooth_1x"].num_nodes = 4

    identity_edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    for name in ["smooth_8x", "smooth_4x", "smooth_2x", "smooth_1x"]:
        graph[(name, "to", name)].edge_index = identity_edges
        graph[(name, "to", name)].edge_weight = torch.ones(4)
    return graph


def test_multi_scale_instantiation(loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    """Test multiscale loss instantiation with single scale."""
    per_scale_loss = AlmostFairKernelCRPS()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
        keep_batch_sharded=False,
    )

    pred, target, loss_result = loss_inputs_multiscale
    loss = multiscale_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_result), "Loss should be equal to the expected result"


def test_multiscale_graph_requires_graph_data() -> None:
    per_scale_loss = MSELoss()
    with pytest.raises(AssertionError, match="graph_data must be provided"):
        MultiscaleLossWrapper(
            per_scale_loss=per_scale_loss,
            weights=[1.0, 1.0],
            keep_batch_sharded=False,
            loss_matrices_graph=[
                {"edges_name": ["smooth_8x", "to", "smooth_8x"], "edge_weight_attribute": "edge_weight"},
                None,
            ],
            graph_data=None,
        )


def test_multiscale_graph_based(
    loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    multiscale_graph_data: HeteroData,
) -> None:
    per_scale_loss = MSELoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        keep_batch_sharded=False,
        loss_matrices_graph=[
            {"edges_name": ["smooth_8x", "to", "smooth_8x"], "edge_weight_attribute": "edge_weight"},
            {"edges_name": ["smooth_4x", "to", "smooth_4x"], "edge_weight_attribute": "edge_weight"},
            {"edges_name": ["smooth_2x", "to", "smooth_2x"], "edge_weight_attribute": "edge_weight"},
            {"edges_name": ["smooth_1x", "to", "smooth_1x"], "edge_weight_attribute": "edge_weight"},
            None,
        ],
        graph_data=multiscale_graph_data,
    )

    pred, target, _ = loss_inputs_multiscale
    loss = multiscale_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert multiscale_loss.smoothing_matrices[-1] is None


@pytest.mark.parametrize("per_scale_loss", [AlmostFairKernelCRPS(), MSELoss()])
@pytest.mark.parametrize("weights", [torch.tensor([0.3, 0.7]), torch.tensor([1.0, 2.0])])
def test_multi_scale(
    loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    per_scale_loss: BaseLoss,
    weights: torch.Tensor,
    mocker: MockerFixture,
) -> None:
    """Test multiscale loss with different per-scale losses and weights."""
    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 4
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 0]])
    graph[("src", "to", "dst")].edge_weight = torch.ones(8) / 2

    smoothing_provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="edge_weight",
        row_normalize=False,
    )

    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[None, smoothing_provider],
    )

    multiscale_loss = MultiscaleLossWrapper(
        loss_matrices=[None, "fake"],
        per_scale_loss=per_scale_loss,
        weights=weights,
        keep_batch_sharded=False,
    )

    pred, target, _ = loss_inputs_multiscale
    loss = multiscale_loss(pred, target, squash=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (2,), "Loss should have shape (num_scales,) when squash=True"
    loss = multiscale_loss(pred, target, squash=False)

    assert isinstance(loss, torch.Tensor)
    # better to have a nvar > 1 because otherwise pred.shape[-1] == 1 and loss.shape == (2) which makes the test fail
    assert loss.shape == (2, pred.shape[-1]), "Loss should have shape (num_scales, num_variables) when squash=False"


def test_multiscale_loss_equivalent_to_per_scale_loss() -> None:
    """Test equivalence when only one scale is used."""
    tensor_shape = [1, 1, 2, 4, 1]  # (batch, output_steps, ens, latlon, vars)

    pred = torch.zeros(tensor_shape)
    pred[0, 0, :, 0] = torch.tensor([1.0])
    target = torch.zeros([tensor_shape[0], tensor_shape[1], tensor_shape[3], tensor_shape[4]])  # no ensemble dim

    per_scale_loss = AlmostFairKernelCRPS()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
        keep_batch_sharded=False,
    )

    loss = multiscale_loss(pred, target)
    loss_kcrps = per_scale_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_kcrps), "Loss for single/original scale should be equal to the kcrps"
