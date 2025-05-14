# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch_geometric.data import HeteroData

from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask


def test_boolean1d_apply() -> None:
    num_points = 10
    x = torch.reshape(torch.arange(1, num_points + 1), (1, 1, num_points, 1))
    y = -x.clone()
    mask = torch.arange(num_points) % 2 == 0
    graph = HeteroData()
    graph["nodes"]["attr"] = mask
    bool_1d = Boolean1DMask(graph, "nodes", "attr")

    x_1 = bool_1d.apply(x, dim=2, fill_value=y)

    assert x_1.shape == x.shape
    assert torch.all(x_1[:, :, ~mask, :] < 0)
    assert torch.all(x_1[:, :, ~mask, :] == y[:, :, ~mask, :])
    assert torch.all(x_1[:, :, mask, :] > 0)
    assert torch.all(x_1[:, :, mask, :] == x[:, :, mask, :])

    x_2 = bool_1d.apply(x, dim=2, fill_value=0)

    assert x_2.shape == x.shape
    assert torch.all(x_2[:, :, ~mask, :] == 0)
    assert torch.all(x_2[:, :, mask, :] > 0)
    assert torch.all(x_2[:, :, mask, :] == x[:, :, mask, :])


def test_nooutput_apply() -> None:
    num_points = 10
    x = torch.reshape(torch.arange(1, num_points + 1), (1, 1, num_points, 1))
    y = -x.clone()

    no_mask = NoOutputMask()

    x_1 = no_mask.apply(x, dim=2, fill_value=y)

    assert x_1.shape == x.shape
    assert torch.all(x_1 == x)
