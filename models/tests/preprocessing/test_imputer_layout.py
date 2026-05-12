# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Layout-aware imputer tests.

Validates that ``BaseImputer.{get_nans,transform,inverse_transform}``
correctly consume a :class:`TensorLayout` for both gridded
``(B, T, E, N, V)`` and sparse ``(E=1, N, V)`` per-sample tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data.batch import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.imputer import ConstantImputer


def _make_imputer():
    config = DictConfig(
        {
            "data": {
                "imputer": {"default": "none", 1.0: ["a"], 2.0: ["b"]},
                "forcing": [],
                "diagnostic": [],
            },
        },
    )
    name_to_index = {"a": 0, "b": 1, "c": 2}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    return ConstantImputer(config=config.data.imputer, data_indices=data_indices)


def test_get_nans_gridded_with_layout():
    imputer = _make_imputer()
    # (B=2, T=2, E=2, N=3, V=3) — NaNs identical across the ensemble axis.
    x = torch.zeros(2, 2, 2, 3, 3)
    x[0, 0, :, 1, 0] = float("nan")
    x[1, 1, :, 2, 1] = float("nan")
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)

    mask = imputer.get_nans(x, layout=layout)
    assert mask.shape == (2, 2, 3, 3)  # ensemble collapsed
    assert mask[0, 0, 1, 0].item() is True
    assert mask[1, 1, 2, 1].item() is True
    assert mask.sum().item() == 2


def test_get_nans_sparse_with_layout():
    imputer = _make_imputer()
    # (E=1, N=4, V=3) — sparse per-sample shape.
    x = torch.zeros(1, 4, 3)
    x[0, 1, 0] = float("nan")
    x[0, 2, 1] = float("nan")
    layout = TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)

    mask = imputer.get_nans(x, layout=layout)
    assert mask.shape == (4, 3)  # ensemble dropped
    assert mask[1, 0].item() is True
    assert mask[2, 1].item() is True
    assert mask.sum().item() == 2


def test_get_nans_legacy_fallback_matches_layout():
    """When no layout is supplied the behaviour matches the legacy path for 5-D inputs."""
    imputer = _make_imputer()
    x = torch.zeros(2, 2, 2, 3, 3)
    x[0, 0, 0, 1, 0] = float("nan")
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    legacy = imputer.get_nans(x)
    layouted = imputer.get_nans(x, layout=layout)
    assert torch.equal(legacy, layouted)


def test_transform_inverse_sparse_roundtrip():
    """Imputer fills NaNs and then puts them back via inverse_transform on sparse data."""
    imputer = _make_imputer()
    x = torch.tensor(
        [
            [
                [float("nan"), 0.0, 0.0],
                [0.0, float("nan"), 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    layout = TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)
    transformed = imputer.transform(x.clone(), layout=layout)
    # NaNs replaced by configured constants (1 for "a", 2 for "b").
    assert transformed[0, 0, 0].item() == pytest.approx(1.0)
    assert transformed[0, 1, 1].item() == pytest.approx(2.0)
    assert not torch.isnan(transformed).any()

    # Inverse should restore NaNs at the same positions.
    restored = imputer.inverse_transform(transformed.clone(), layout=layout)
    assert torch.isnan(restored[0, 0, 0])
    assert torch.isnan(restored[0, 1, 1])
    assert restored[0, 2, 2].item() == 0.0


def test_transform_inverse_gridded_roundtrip_with_layout():
    """Layout-aware path matches legacy behaviour for a 5-D gridded tensor."""
    imputer = _make_imputer()
    base = torch.zeros(1, 1, 1, 2, 3)
    base[0, 0, 0, 0, 0] = float("nan")
    base[0, 0, 0, 1, 1] = float("nan")
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)

    transformed = imputer.transform(base.clone(), layout=layout)
    assert transformed[0, 0, 0, 0, 0].item() == pytest.approx(1.0)
    assert transformed[0, 0, 0, 1, 1].item() == pytest.approx(2.0)

    restored = imputer.inverse_transform(transformed.clone(), layout=layout)
    assert torch.isnan(restored[0, 0, 0, 0, 0])
    assert torch.isnan(restored[0, 0, 0, 1, 1])


def test_loss_mask_training_sparse_has_singleton_batch_dim():
    """Sparse path must prepend a singleton batch dim so loss_mask_training
    matches the (BATCH, GRID, VARIABLE) scaler contract used downstream.
    """
    imputer = _make_imputer()
    # (E=1, N=4, V=3) with a NaN to ensure the mask is built.
    x = torch.zeros(1, 4, 3)
    x[0, 1, 0] = float("nan")
    layout = TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)

    imputer.transform(x.clone(), layout=layout)
    # Expect (1, N, n_outputs) — n_outputs = number of model output vars.
    n_outputs = len(imputer.data_indices.model.output.name_to_index)
    assert imputer.loss_mask_training.shape == (1, 4, n_outputs)


def test_loss_mask_training_gridded_has_batch_dim():
    """Gridded path keeps the existing 3-D shape (B, N, n_outputs)."""
    imputer = _make_imputer()
    x = torch.zeros(2, 1, 1, 5, 3)
    x[0, 0, 0, 1, 0] = float("nan")
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    imputer.transform(x.clone(), layout=layout)
    n_outputs = len(imputer.data_indices.model.output.name_to_index)
    assert imputer.loss_mask_training.shape == (2, 5, n_outputs)
