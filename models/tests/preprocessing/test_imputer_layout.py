# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Layout-aware imputer tests.

Validates that ``BaseImputer`` fills NaNs along the variables axis of a
:class:`~anemoi.models.data.sources.Source` for both gridded
``(B, T, E, N, V)`` and tabular ``(N, V)`` per-sample tensors, and that the
inverse transform is a no-op.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.preprocessing.imputer import ConstantImputer

VARIABLES = ["a", "b", "c"]


def _make_imputer():
    config = DictConfig({"default": "none", 1.0: ["a"], 2.0: ["b"]})
    return ConstantImputer(config=config)


def test_transform_inverse_tabular_roundtrip():
    """Imputer fills NaNs on tabular data and inverse_transform leaves them filled."""
    imputer = _make_imputer()
    x = torch.tensor(
        [
            [float("nan"), 0.0, 0.0],
            [0.0, float("nan"), 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    view = TabularSource(
        name="tabular",
        data=[x],
        variables=VARIABLES,
        coordinates=[torch.zeros(x.shape[0], 2)],
        layout=TensorLayout(grid=0, variables=1),
        timedeltas=[torch.zeros(x.shape[0])],
        boundaries=[(slice(0, x.shape[0]),)],
    )
    transformed = imputer(view, in_place=False).data[0]
    # NaNs replaced by configured constants (1 for "a", 2 for "b").
    assert transformed[0, 0].item() == pytest.approx(1.0)
    assert transformed[1, 1].item() == pytest.approx(2.0)
    assert not torch.isnan(transformed).any()

    restored = imputer(view.clone(data=[transformed]), in_place=False, inverse=True).data[0]
    assert torch.equal(restored, transformed)


def test_transform_inverse_gridded_roundtrip_with_ensemble():
    """A 5-D gridded tensor with an ensemble axis is imputed along its variables axis."""
    imputer = _make_imputer()
    base = torch.zeros(1, 1, 2, 2, 3)
    base[0, 0, :, 0, 0] = float("nan")
    base[0, 0, :, 1, 1] = float("nan")
    base[0, 0, 1, 0, 2] = float("nan")  # "c" has no replacement configured
    view = GriddedSource(
        name="gridded",
        data=base,
        variables=VARIABLES,
        coordinates=torch.zeros(2, 2),
        layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
    )

    transformed = imputer(view, in_place=False).data
    assert torch.all(transformed[0, 0, :, 0, 0] == 1.0)
    assert torch.all(transformed[0, 0, :, 1, 1] == 2.0)
    assert torch.isnan(transformed[0, 0, 1, 0, 2])
    assert torch.isnan(view.data[0, 0, 0, 0, 0]), "in_place=False must not modify the input."

    restored = imputer(view.clone(data=transformed), in_place=False, inverse=True).data
    assert torch.allclose(restored, transformed, equal_nan=True)
