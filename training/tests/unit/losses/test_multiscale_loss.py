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
from omegaconf import DictConfig
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses import IndexSpace
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function
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


def test_multiscale_forwards_layout_kwargs_to_filtered_per_scale_loss() -> None:
    """Nested per-scale filtered losses must receive layout kwargs."""
    data_indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1})
    multiscale_loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                "weights": [1.0],
                "keep_batch_sharded": False,
                "loss_matrices": [None],
                "per_scale_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "scalers": [],
                },
            },
        ),
        scalers={},
        data_indices=data_indices,
    )

    pred = torch.ones((1, 1, 1, 4, 2))
    target = torch.zeros((1, 1, 1, 4, 2))
    loss = multiscale_loss(
        pred,
        target,
        group=None,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (1,)


def test_multiscale_filtered_per_scale_update_scaler_preserves_variable_filtering() -> None:
    """Updating VARIABLE scalers through multiscale should preserve filtered per-scale shapes."""
    data_indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1})
    multiscale_loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                "weights": [1.0],
                "keep_batch_sharded": False,
                "loss_matrices": [None],
                "per_scale_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "predicted_variables": ["a"],
                    "target_variables": ["a"],
                    "scalers": ["grid_uniform", "dynamic"],
                },
            },
        ),
        scalers={
            "grid_uniform": (3, torch.ones(4)),
            "dynamic": (4, torch.ones(2)),
        },
        data_indices=data_indices,
    )

    # Initial VARIABLE scaler is filtered to selected predicted variables.
    torch.testing.assert_close(multiscale_loss.loss.loss.scaler.tensors["dynamic"][1], torch.tensor([1.0]))

    # Update with full-width VARIABLE scaler and ensure filtering is preserved.
    multiscale_loss.update_scaler("dynamic", torch.tensor([3.0, 5.0]), override=True)
    updated = multiscale_loss.loss.loss.scaler.tensors["dynamic"][1]
    assert updated.shape == (1,)
    torch.testing.assert_close(updated, torch.tensor([3.0]))

    pred = torch.ones((1, 1, 1, 4, 2))
    target = torch.zeros((1, 1, 1, 4, 2))
    loss = multiscale_loss(
        pred,
        target,
        group=None,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
