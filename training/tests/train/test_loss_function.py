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

from anemoi.training.losses.mse import WeightedMSELoss
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.train.forecaster import GraphForecaster


def test_manual_init() -> None:
    loss = WeightedMSELoss(torch.ones(1))
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_include() -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.mse.WeightedMSELoss"}),
        node_weights=torch.ones(1),
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_scalar() -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.mse.WeightedMSELoss",
                "scalars": ["test"],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseWeightedLoss)

    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" in loss.scalar
    torch.testing.assert_close(loss.scalar.get_scalar(2), torch.ones((1, 2)))


def test_dynamic_init_scalar_not_add() -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.mse.WeightedMSELoss",
                "scalars": [],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseWeightedLoss)
    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" not in loss.scalar
