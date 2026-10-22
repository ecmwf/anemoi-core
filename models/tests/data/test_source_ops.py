# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Structure-agnostic payload operations on sources (ported from ``transport/data_helpers`` tests)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import Source
from anemoi.models.data.sources import TabularSource

TABULAR_LAYOUT = TensorLayout.from_tuple("grid", "variables")
TABULAR_ENSEMBLE_LAYOUT = TensorLayout.from_tuple("ensemble", "grid", "variables")
GRIDDED_LAYOUT = TensorLayout.from_tuple("batch", "time", "ensemble", "grid", "variables")


def _tabular(samples: list[torch.Tensor], layout: TensorLayout = TABULAR_LAYOUT) -> Source:
    grid_axis = layout.axis("grid", ndim=layout.ndim)
    nodes = [sample.shape[grid_axis] for sample in samples]
    return TabularSource(
        name="obs",
        variables=["a"],
        layout=layout,
        data=samples,
        coordinates=[torch.zeros(n, 2) for n in nodes],
        timedeltas=[torch.zeros(n) for n in nodes],
        boundaries=[(slice(0, n),) for n in nodes],
    )


def _gridded(data: torch.Tensor) -> Source:
    return GriddedSource(
        name="era5",
        variables=["a"],
        layout=GRIDDED_LAYOUT,
        data=data,
        coordinates=torch.zeros(data.shape[GRIDDED_LAYOUT.grid], 2),
    )


def test_map_with_condition_preserves_sample_and_member_alignment() -> None:
    left = _tabular([torch.ones(2, 2, 1), torch.full((2, 3, 1), 4.0)], TABULAR_ENSEMBLE_LAYOUT)
    right = _tabular([torch.full((2, 2, 1), 2.0), torch.full((2, 3, 1), 10.0)], TABULAR_ENSEMBLE_LAYOUT)
    condition = torch.tensor([2.0, 3.0, 5.0, 7.0]).reshape(2, 1, 2, 1, 1)

    result = left.map_with_condition(lambda a, b, c: a + b * c, condition, right)

    torch.testing.assert_close(result.data[0], torch.tensor([[[5.0], [5.0]], [[7.0], [7.0]]]))
    torch.testing.assert_close(result.data[1], torch.tensor([[[54.0], [54.0], [54.0]], [[74.0], [74.0], [74.0]]]))


@pytest.mark.parametrize(
    ("data", "other", "batch_size", "error", "message"),
    [
        (
            lambda: _tabular([torch.zeros(1, 1)]),
            lambda: _gridded(torch.zeros(1, 1, 1, 1, 1)),
            1,
            TypeError,
            "Cannot combine gridded and tabular",
        ),
        (
            lambda: _tabular([torch.zeros(1, 1)]),
            lambda: _tabular([torch.zeros(1, 1), torch.zeros(1, 1)]),
            1,
            ValueError,
            "same length",
        ),
        (lambda: _tabular([]), lambda: _tabular([torch.zeros(1, 1)]), 0, ValueError, "same length"),
        (
            lambda: _tabular([torch.zeros(1, 1)]),
            lambda: _tabular([torch.zeros(1, 1)]),
            2,
            ValueError,
            "Condition batch size",
        ),
    ],
)
def test_map_with_condition_validates_structure_before_computing(
    data, other, batch_size: int, error: type[Exception], message: str
) -> None:
    operation = Mock()
    with pytest.raises(error, match=message):
        data().map_with_condition(operation, torch.ones(batch_size, 1, 1, 1, 1), other())
    operation.assert_not_called()
