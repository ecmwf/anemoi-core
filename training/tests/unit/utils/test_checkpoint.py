# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from anemoi.training.utils.checkpoint import freeze_submodule_by_name
import torch
from torch import nn


@pytest.fixture
def model() -> torch.nn.Module:
    class SubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(10, 10)
            self.sequential = nn.Sequential(
                nn.Linear(10, 10),
                nn.Linear(10, 10),
            )

    return SubModule()


def test_freeze_submodule(model: torch.nn.Module) -> None:
    """Test the freeze_submodule_by_name function."""

    freeze_submodule_by_name(model, "sequential.0")

    assert model.lin1.weight.requires_grad
    assert not model.sequential[0].weight.requires_grad
    assert model.sequential[1].weight.requires_grad


def test_freeze_module(model: torch.nn.Module) -> None:
    """Test the freeze_submodule_by_name function."""

    freeze_submodule_by_name(model, "sequential")

    assert model.lin1.weight.requires_grad
    assert not model.sequential[0].weight.requires_grad
    assert not model.sequential[1].weight.requires_grad
