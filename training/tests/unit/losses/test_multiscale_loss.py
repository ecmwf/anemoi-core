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
from anemoi.training.utils.enums import TensorDim


class TrackingLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: object | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del pred, target, squash
        self.calls.append(
            {
                "scaler_indices": scaler_indices,
                "without_scalers": without_scalers,
                "grid_shard_slice": grid_shard_slice,
                "group": group,
                "kwargs": kwargs,
            },
        )
        return torch.tensor(1.0)


class FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


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


def test_multiscale_loss_forwards_scaler_indices() -> None:
    pred = torch.zeros((1, 1, 1, 2, 2))
    pred[0, 0, 0, 0, 0] = 10.0
    pred[0, 0, 0, 0, 1] = 1.0
    target = torch.zeros((1, 1, 2, 2))

    per_scale_loss = MSELoss()
    per_scale_loss.add_scaler(TensorDim.GRID, torch.ones(2), name="grid_weights")
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
        keep_batch_sharded=False,
    )

    scaler_indices = (..., [1])
    loss = multiscale_loss(pred, target, scaler_indices=scaler_indices)
    expected = per_scale_loss(pred, target, scaler_indices=scaler_indices)

    assert torch.allclose(loss, expected)


def test_multiscale_loss_forwards_group_and_without_scalers() -> None:
    per_scale_loss = TrackingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
        keep_batch_sharded=False,
    )

    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 2, 1))
    sentinel_group = FakeGroup(size=1)

    multiscale_loss(
        pred,
        target,
        scaler_indices=(..., [0]),
        without_scalers=["node_weights"],
        group=sentinel_group,
    )

    assert per_scale_loss.calls == [
        {
            "scaler_indices": (..., [0]),
            "without_scalers": ["node_weights"],
            "grid_shard_slice": None,
            "group": sentinel_group,
            "kwargs": {},
        },
    ]


def test_multiscale_loss_uses_grid_shard_shapes_for_sharding(mocker: MockerFixture) -> None:
    per_scale_loss = TrackingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
        keep_batch_sharded=True,
    )
    group = FakeGroup(size=2)
    shard_shapes = [(1, 2, 1), (1, 2, 1)]
    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 2, 1))

    prepare = mocker.patch.object(
        multiscale_loss,
        "_prepare_for_smoothing",
        return_value=(pred, target, shard_shapes, shard_shapes),
    )
    gather = mocker.patch(
        "anemoi.training.losses.multiscale.gather_channels",
        side_effect=lambda x, *_args: x,
    )

    multiscale_loss(
        pred,
        target,
        group=group,
        grid_dim=-2,
        grid_shard_shapes=shard_shapes,
    )

    prepare.assert_called_once_with(pred, target, group, -2, shard_shapes)
    assert gather.call_count == 2


def test_multiscale_loss_forwards_extra_kwargs() -> None:
    per_scale_loss = TrackingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
        keep_batch_sharded=False,
    )

    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 2, 1))
    sentinel = object()

    multiscale_loss(
        pred,
        target,
        custom_kwarg=sentinel,
    )

    assert per_scale_loss.calls == [
        {
            "scaler_indices": None,
            "without_scalers": None,
            "grid_shard_slice": None,
            "group": None,
            "kwargs": {"custom_kwarg": sentinel},
        },
    ]
