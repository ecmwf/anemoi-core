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
from torch.testing import assert_close

from anemoi.training.losses.msle import WeightedMSLELoss


@pytest.fixture
def basic_inputs() -> tuple:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    node_weights = torch.ones(2)
    return pred, target, node_weights


def test_forward(basic_inputs: tuple) -> None:
    pred, target, node_weights = basic_inputs
    loss = WeightedMSLELoss(node_weights=node_weights)
    computed_loss = loss(pred, target)
    assert isinstance(computed_loss, torch.Tensor)
    assert_close(computed_loss, torch.tensor(0.0))
