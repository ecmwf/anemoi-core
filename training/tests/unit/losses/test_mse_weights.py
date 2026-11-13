# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.training.losses import MSELoss

if TYPE_CHECKING:
    from anemoi.models.data.sources.base import Source


def _sparse_view(data: list[torch.Tensor]) -> Source:
    layout = TensorLayout(grid=0, variables=1)
    return TabularSource(
        name="obs",
        data=data,
        variables=["a", "b"],
        statistics={},
        coordinates=[torch.zeros(sample.shape[0], 2) for sample in data],
        layout=layout,
        boundaries=[(slice(0, sample.shape[0]),) for sample in data],
        timedeltas=[torch.zeros(sample.shape[0]) for sample in data],
    )


def test_mse_applies_sparse_sample_weights() -> None:
    pred = _sparse_view([torch.ones(2, 2), torch.full((3, 2), 2.0)])
    target = _sparse_view([torch.zeros(2, 2), torch.zeros(3, 2)])
    weights = [torch.full((2, 1), 2.0), torch.full((3, 1), 3.0)]

    loss = MSELoss()(pred, target, per_sample_kwargs={"weights": weights}, squash=False)

    torch.testing.assert_close(loss, torch.tensor([7.0, 7.0]))


def test_mse_applies_gridded_weights() -> None:
    """Weighting each sample's squared error by w is the same as scaling its error by sqrt(w)."""
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)

    def view(data: torch.Tensor) -> Source:
        return GriddedSource(
            name="data",
            data=data,
            variables=["a", "b"],
            statistics={},
            layout=layout,
            coordinates=torch.zeros(3, 2),
        )

    error = torch.randn(2, 1, 1, 3, 2, dtype=torch.float64)
    target = torch.zeros_like(error)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64).view(2, 1, 1, 1, 1)

    weighted = MSELoss()(view(error), view(target), weights=weights, squash=False)
    scaled = MSELoss()(view(error * weights.sqrt()), view(target), squash=False)
    unweighted = MSELoss()(view(error), view(target), squash=False)

    torch.testing.assert_close(weighted, scaled)
    assert not torch.allclose(weighted, unweighted)
