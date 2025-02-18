# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
from omegaconf import DictConfig

from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function


def test_manual_init() -> None:
    loss = MSELoss(torch.ones(1))
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_include() -> None:
    loss = get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.MSELoss"}),
        node_weights=torch.ones(1),
    )
    assert isinstance(loss, BaseLoss)
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_scaler() -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "scalers": ["test"],
            },
        ),
        node_weights=torch.ones(1),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


def test_dynamic_init_add_all() -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "scalers": ["*"],
            },
        ),
        node_weights=torch.ones(1),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


def test_dynamic_init_scaler_not_add() -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "scalers": [],
            },
        ),
        node_weights=torch.ones(1),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" not in loss.scaler


def test_dynamic_init_scaler_exclude() -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "scalers": ["*", "!test"],
            },
        ),
        node_weights=torch.ones(1),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" not in loss.scaler
