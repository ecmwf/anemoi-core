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
from torch_geometric.data import HeteroData

from anemoi.training.losses.combined import CombinedLoss
from anemoi.training.losses.mae import WeightedMAELoss
from anemoi.training.losses.mse import WeightedMSELoss
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.train.forecaster import GraphForecaster


@pytest.fixture
def graph_data() -> HeteroData:
    hdata = HeteroData()
    lons = torch.tensor([1.56, 3.12, 4.68, 6.24])
    lats = torch.tensor([-3.12, -1.56, 1.56, 3.12])
    cutout_mask = torch.tensor([False, True, False, False]).unsqueeze(1)
    area_weights = torch.ones(cutout_mask.shape)
    hdata["data"]["x"] = torch.stack((lats, lons), dim=1)
    hdata["data"]["cutout"] = cutout_mask
    hdata["data"]["area_weight"] = area_weights
    return hdata


@pytest.fixture
def scalars() -> dict[str, tuple]:
    variable_scaling = torch.tensor([1.0])
    limited_area_mask = torch.tensor([1.0])
    return {
        "variable": (-1, variable_scaling),
        "loss_weights_mask": ((-2, -1), torch.ones((1, 1))),
        "limited_area_mask": (2, limited_area_mask),
    }


@pytest.fixture
def output_mask() -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 1.0, 1.0])


def test_manual_init() -> None:
    loss = WeightedMSELoss(torch.ones(1))
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_include(scalars: dict[str, tuple], graph_data: HeteroData, output_mask: torch.Tensor) -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.mse.WeightedMSELoss"}),
        node_weights=torch.ones(1),
        scalars=scalars,
        output_mask=output_mask,
        graph_data=graph_data,
    )
    assert isinstance(loss, BaseWeightedLoss)
    assert loss.node_weights == torch.ones(1)


def test_dynamic_init_scalar(graph_data: HeteroData, output_mask: torch.Tensor) -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.mse.WeightedMSELoss",
                "scalars": ["test"],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": ((0, 1), torch.ones((1, 2)))},
        output_mask=output_mask,
        graph_data=graph_data,
    )
    assert isinstance(loss, BaseWeightedLoss)

    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" in loss.scalar
    torch.testing.assert_close(loss.scalar.get_scalar(2), torch.ones((1, 2)))


def test_dynamic_init_scalar_not_add(graph_data: HeteroData, output_mask: torch.Tensor) -> None:
    loss = GraphForecaster.get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.mse.WeightedMSELoss",
                "scalars": [],
            },
        ),
        node_weights=torch.ones(1),
        scalars={"test": (-1, torch.ones(2))},
        output_mask=output_mask,
        graph_data=graph_data,
    )
    assert isinstance(loss, BaseWeightedLoss)
    torch.testing.assert_close(loss.node_weights, torch.ones(1))
    assert "test" not in loss.scalar


def test_combined_loss() -> None:
    loss1 = WeightedMSELoss(torch.ones(1))
    loss2 = WeightedMAELoss(torch.ones(1))
    cl = CombinedLoss(losses=[loss1, loss2], loss_weights=(1.0, 0.5))
    assert isinstance(cl, CombinedLoss)
    cl_class = CombinedLoss(
        losses=[WeightedMSELoss, WeightedMAELoss],
        node_weights=torch.ones(1),
        loss_weights=(1.0, 0.5),
    )
    assert isinstance(cl_class, CombinedLoss)
