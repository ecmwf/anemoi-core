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
from omegaconf import DictConfig

from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses import HuberLoss
from anemoi.training.losses import KernelCRPS
from anemoi.training.losses import LogCoshLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import RMSELoss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim


@pytest.mark.parametrize(
    "loss_cls",
    [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss],
)
def test_manual_init(loss_cls: type[BaseLoss]) -> None:
    loss = loss_cls()
    assert isinstance(loss, BaseLoss)


@pytest.fixture
def functionalloss() -> type[FunctionalLoss]:
    class ReturnDifference(FunctionalLoss):
        def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return pred - target

    return ReturnDifference


@pytest.fixture
def loss_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 1, 4, 2]

    pred = torch.zeros(tensor_shape)
    pred[0, 0, 0] = torch.tensor([1.0, 1.0])
    target = torch.zeros(tensor_shape)

    # With only one "grid point" differing by 1 in all
    # variables, the loss should be 1.0

    loss_result = torch.tensor([1.0])
    return pred, target, loss_result


@pytest.fixture
def loss_inputs_fine(
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs with finer grid."""
    pred, target, loss_result = loss_inputs

    pred = torch.cat([pred, pred], dim=2)
    target = torch.cat([target, target], dim=2)

    return pred, target, loss_result


def test_assert_of_grid_dim(functionalloss: type[FunctionalLoss]) -> None:
    """Test that the grid dimension is set correctly."""
    loss = functionalloss()
    loss.add_scaler(TensorDim.VARIABLE, 1.0, name="variable_test")

    assert TensorDim.GRID not in loss.scaler, "Grid dimension should not be set"

    with pytest.raises(RuntimeError):
        loss.scale(torch.ones((4, 2)))


@pytest.fixture
def simple_functionalloss(functionalloss: type[FunctionalLoss]) -> FunctionalLoss:
    loss = functionalloss()
    loss.add_scaler(TensorDim.GRID, torch.ones((4,)), name="unit_scaler")
    return loss


@pytest.fixture
def functionalloss_with_scaler(simple_functionalloss: FunctionalLoss) -> FunctionalLoss:
    loss = simple_functionalloss
    loss.add_scaler(TensorDim.GRID, torch.rand((4,)), name="test")
    return loss


@pytest.fixture
def functionalloss_with_scaler_fine(functionalloss: FunctionalLoss) -> FunctionalLoss:
    loss = functionalloss()
    loss.add_scaler(TensorDim.GRID, torch.rand((8,)), name="test")
    return loss


def test_simple_functionalloss(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test a functional loss."""
    pred, target, loss_result = loss_inputs

    loss = simple_functionalloss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_result), "Loss should be equal to the expected result"


def test_batch_invariance(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, loss_result = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape, "Batch size should be different"

    loss_batch_size_1 = simple_functionalloss(pred_batch_size_1, target_batch_size_1)
    loss_batch_size_2 = simple_functionalloss(pred_batch_size_2, target_batch_size_2)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert torch.allclose(loss_batch_size_1, loss_result), "Loss should be equal to the expected result"

    assert isinstance(loss_batch_size_2, torch.Tensor)
    assert torch.allclose(loss_batch_size_2, loss_result), "Loss should be equal to the expected result"

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_batch_invariance_without_squash(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, _ = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape, "Batch size should be different"

    loss_batch_size_1 = simple_functionalloss(pred_batch_size_1, target_batch_size_1, squash=False)
    loss_batch_size_2 = simple_functionalloss(pred_batch_size_2, target_batch_size_2, squash=False)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert isinstance(loss_batch_size_2, torch.Tensor)

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_batch_invariance_with_scaler(
    functionalloss_with_scaler: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, _ = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape

    loss_batch_size_1 = functionalloss_with_scaler(pred_batch_size_1, target_batch_size_1)
    loss_batch_size_2 = functionalloss_with_scaler(pred_batch_size_2, target_batch_size_2)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert isinstance(loss_batch_size_2, torch.Tensor)

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_grid_invariance(
    functionalloss_with_scaler: FunctionalLoss,
    functionalloss_with_scaler_fine: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred_coarse, target_coarse, _ = loss_inputs
    pred_fine = torch.cat([pred_coarse, pred_coarse], dim=2)
    target_fine = torch.cat([target_coarse, target_coarse], dim=2)

    num_points_coarse = pred_coarse.shape[2]
    num_points_fine = pred_fine.shape[2]

    functionalloss_with_scaler.update_scaler("test", torch.ones((num_points_coarse,)) / num_points_coarse)
    functionalloss_with_scaler_fine.update_scaler("test", torch.ones((num_points_fine,)) / num_points_fine)

    loss_coarse = functionalloss_with_scaler(pred_coarse, target_coarse)
    loss_fine = functionalloss_with_scaler_fine(pred_fine, target_fine)

    assert isinstance(loss_coarse, torch.Tensor)
    assert isinstance(loss_fine, torch.Tensor)

    assert torch.allclose(loss_coarse, loss_fine), "Losses should be equal between grid sizes"


@pytest.mark.parametrize(
    "loss_cls",
    [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss],
)
def test_dynamic_init_include(loss_cls: type[BaseLoss]) -> None:
    loss = get_loss_function(DictConfig({"_target_": f"anemoi.training.losses.{loss_cls.__name__}"}))
    assert isinstance(loss, BaseLoss)


@pytest.mark.parametrize(
    "loss_cls",
    [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss],
)
def test_dynamic_init_scaler(loss_cls: type[BaseLoss]) -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
                "scalers": ["test"],
            },
        ),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "loss_cls",
    [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss],
)
def test_dynamic_init_add_all(loss_cls: type[BaseLoss]) -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
                "scalers": ["*"],
            },
        ),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "loss_cls",
    [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss],
)
def test_dynamic_init_scaler_not_add(loss_cls: type[BaseLoss]) -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
                "scalers": [],
            },
        ),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


@pytest.mark.parametrize(
    "loss_cls",
    [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss],
)
def test_dynamic_init_scaler_exclude(loss_cls: type[BaseLoss]) -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
                "scalers": ["*", "!test"],
            },
        ),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


def test_logfft2dist_loss() -> None:
    """Test that loss function can be instantiated."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spatial.LogFFT2Distance",
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
    )
    assert isinstance(loss, FunctionalLoss)
    assert hasattr(loss, "x_dim")
    assert hasattr(loss, "y_dim")

    right_shaped_pred_output_pair = (torch.ones((6, 1, 710 * 640, 2)), torch.zeros((6, 1, 710 * 640, 2)))
    loss_value = loss.calculate_difference(*right_shaped_pred_output_pair)
    assert loss_value.shape == torch.Size((6, 1, 710 * 640, 2)), "Loss output shape should match input shape"
    wrong_shaped_pred_output_pair = (torch.ones((6, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 710 * 640 + 1, 2)))
    with pytest.raises(AssertionError):
        loss.calculate_difference(*wrong_shaped_pred_output_pair)


def test_fcl_loss() -> None:
    """Test that loss function can be instantiated and behaves as expected."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spatial.FourierCorrelationLoss",
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
    )
    assert isinstance(loss, FunctionalLoss)
    assert hasattr(loss, "x_dim")
    assert hasattr(loss, "y_dim")

    right_shaped_pred_output_pair = (torch.ones((6, 1, 710 * 640, 2)), torch.zeros((6, 1, 710 * 640, 2)))
    loss_value = loss.calculate_difference(*right_shaped_pred_output_pair)
    assert loss_value.shape == torch.Size((6, 1, 710 * 640, 2)), "Loss output shape should match input shape"
    wrong_shaped_pred_output_pair = (torch.ones((6, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 710 * 640 + 1, 2)))
    with pytest.raises(AssertionError):
        loss.calculate_difference(*wrong_shaped_pred_output_pair)
