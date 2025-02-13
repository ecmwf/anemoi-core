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

from anemoi.training.losses.combined import CombinedLoss
from anemoi.training.losses.mae import WeightedMAELoss
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
        scalars={},
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
