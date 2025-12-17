# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses.multiscale import MultiscaleLossWrapper


@pytest.fixture
def loss_inputs_multiscale() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 2, 4, 1]

    pred = torch.zeros(tensor_shape)
    pred[0, :, 0, 0] = torch.tensor([1.0, 1.0])
    target = torch.zeros(tensor_shape[1:])

    # With only one "grid point" differing by 1 in all
    # variables, the loss should be 1.0

    loss_result = torch.tensor([1.0])
    return pred, target, loss_result


def test_multi_scale(loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
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


def test_multiscale_loss_equivalent_to_per_scale_loss() -> None:

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
