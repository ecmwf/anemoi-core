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
from hydra.errors import InstantiationException
from omegaconf import DictConfig

from anemoi.training.losses import HuberLoss
from anemoi.training.losses import LogCoshLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import RMSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function


@pytest.mark.parametrize("loss_cls", [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss])
def test_manual_init(loss_cls: type[BaseLoss]) -> None:
    loss = loss_cls(torch.ones(1))
    assert isinstance(loss, BaseLoss)


@pytest.mark.parametrize("loss_cls", [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss])
def test_dynamic_init_include(loss_cls: type[BaseLoss]) -> None:
    loss = get_loss_function(DictConfig({"_target_": f"anemoi.training.losses.{loss_cls.__name__}"}))
    assert isinstance(loss, BaseLoss)


@pytest.mark.parametrize("loss_cls", [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss])
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


@pytest.mark.parametrize("loss_cls", [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss])
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


@pytest.mark.parametrize("loss_cls", [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss])
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


@pytest.mark.parametrize("loss_cls", [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss])
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


def test_combined_loss() -> None:
    """Test the combined loss function."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.mse.WeightedMSELoss"},
                    {"_target_": "anemoi.training.losses.mae.WeightedMAELoss"},
                ],
                "scalars": ["test"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, CombinedLoss)
    assert "test" in loss.scalar

    assert isinstance(loss.losses[0], WeightedMSELoss)
    assert "test" in loss.losses[0].scalar

    assert isinstance(loss.losses[1], WeightedMAELoss)
    assert "test" in loss.losses[1].scalar


def test_combined_loss_invalid_loss_weights() -> None:
    """Test the combined loss function with invalid loss weights."""
    with pytest.raises(InstantiationException):
        GraphForecaster.get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.combined.CombinedLoss",
                    "losses": [
                        {"_target_": "anemoi.training.losses.mse.WeightedMSELoss"},
                        {"_target_": "anemoi.training.losses.mae.WeightedMAELoss"},
                    ],
                    "scalars": ["test"],
                    "loss_weights": [1.0, 0.5, 1],
                },
            ),
            node_weights=torch.ones(1),
            scalars={"test": (-1, torch.ones(2))},
        )


def test_combined_loss_invalid_behaviour() -> None:
    """Test the combined loss function and setting the scalrs."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.mse.WeightedMSELoss"},
                    {"_target_": "anemoi.training.losses.mae.WeightedMAELoss"},
                ],
                "scalars": ["test"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": (-1, torch.ones(2))},
    )
    with pytest.raises(AttributeError):
        loss.scalar = "test"


def test_combined_loss_equal_weighting() -> None:
    """Test equal weighting when not given."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.mse.WeightedMSELoss"},
                    {"_target_": "anemoi.training.losses.mae.WeightedMAELoss"},
                ],
            },
        ),
        node_weights=torch.ones(1),
        scalars={},
    )
    assert all(weight == 1.0 for weight in loss.loss_weights)


def test_combined_loss_seperate_scalars() -> None:
    """Test that scalars are passed to the correct loss function."""
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.mse.WeightedMSELoss", "scalars": ["test"]},
                    {"_target_": "anemoi.training.losses.mae.WeightedMAELoss", "scalars": ["test2"]},
                ],
                "scalars": ["test", "test2"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": (-1, torch.ones(2)), "test2": (-1, torch.ones(2))},
    )
    assert isinstance(loss, CombinedLoss)
    assert "test" in loss.scalar
    assert "test2" in loss.scalar

    assert isinstance(loss.losses[0], WeightedMSELoss)
    assert "test" in loss.losses[0].scalar
    assert "test2" not in loss.losses[0].scalar

    assert isinstance(loss.losses[1], WeightedMAELoss)
    assert "test" not in loss.losses[1].scalar
    assert "test2" in loss.losses[1].scalar
