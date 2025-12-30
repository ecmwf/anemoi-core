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

from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.losses.multiscale import SparseProjector


@pytest.fixture
def loss_inputs_multiscale() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 2, 4, 1]  # (bs, time, grid, vars)

    pred = torch.zeros(tensor_shape)
    pred[0, :, 0, 0] = torch.tensor([1.0, 1.0])
    target = torch.zeros(tensor_shape[1:])

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
    smoothing_matrix = SparseProjector(
        torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 0]]),
        torch.ones(8) / 2,
        src_size=4,
        dst_size=4,
        row_normalize=False,
        transpose=False,
    )
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[None, smoothing_matrix],
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
    assert loss.shape == (2, pred.shape[-1]), "Loss should have shape (num_scales, num_variables) when squash=False"


def test_multiscale_loss_equivalent_to_per_scale_loss() -> None:
    """Test equivalence when only one scale is used."""
    tensor_shape = [1, 2, 4, 5]
    pred = torch.randn(tensor_shape)
    target = torch.randn(tensor_shape[1:])

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
