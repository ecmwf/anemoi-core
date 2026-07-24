# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

import torch

from anemoi.models.data import TensorLayout
from anemoi.models.data.views import create_source_view
from anemoi.training.losses import WeightedMSELoss


def _sparse_view(data: list[torch.Tensor]):
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    return create_source_view(
        name="obs",
        data=data,
        variables=["a", "b"],
        statistics={},
        coordinates=[torch.zeros(sample.shape[0], 2) for sample in data],
        layout=layout,
        is_static=False,
        boundaries=None,
    )


def test_weighted_mse_dispatches_sparse_sample_weights() -> None:
    pred = _sparse_view([torch.ones(2, 2), torch.full((3, 2), 2.0)])
    target = _sparse_view([torch.zeros(2, 2), torch.zeros(3, 2)])
    weights = [torch.full((2, 1), 2.0), torch.full((3, 1), 3.0)]

    loss = WeightedMSELoss()(pred, target, weights=weights, squash=False)

    torch.testing.assert_close(loss, torch.tensor([7.0, 7.0]))
