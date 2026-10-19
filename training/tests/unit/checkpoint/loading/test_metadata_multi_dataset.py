# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for multi-dataset metadata preservation.

``model._ckpt_model_name_to_index`` must be a dict keyed by dataset name.
The legacy single-IndexCollection shape raises ``TypeError``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader
from tests.helpers import FakeIndexCollection


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_multi_dataset_data_indices_builds_dataset_keyed_dict() -> None:
    """Dict-of-IndexCollections shape produces dataset-keyed name_to_index and variables metadata."""
    era5 = {"temperature": 0, "wind_u": 1}
    metno = {"temperature": 0, "humidity": 1, "pressure": 2}
    data_indices = {
        "era5": FakeIndexCollection(era5),
        "metno": FakeIndexCollection(metno),
    }
    variables_metadata = {"temperature": {"units": "K"}, "wind_u": {"units": "m s**-1"}}
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {
            "data_indices": data_indices,
            "metadata": {"dataset": {"era5": {"variables_metadata": variables_metadata}}},
        },
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    WeightsOnlyLoader().process(context)

    assert model._ckpt_model_name_to_index == {"era5": era5, "metno": metno}
    assert model._ckpt_variables_metadata["era5"] == variables_metadata


def test_single_dataset_legacy_shape_raises() -> None:
    """Legacy single-IndexCollection shape raises ``TypeError``. Migrate it first."""
    name_to_index = {"temperature": 0, "wind_u": 1, "wind_v": 2}
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": FakeIndexCollection(name_to_index)},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    with pytest.raises(TypeError, match="migration sync"):
        WeightsOnlyLoader().process(context)


def test_empty_dict_data_indices_does_not_set_attribute() -> None:
    """An empty dict is not a usable multi-dataset payload; leave attribute unset."""
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": {}},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    WeightsOnlyLoader().process(context)

    assert not hasattr(model, "_ckpt_model_name_to_index")


def test_multi_dataset_dict_entries_missing_name_to_index_skips() -> None:
    """Dict whose values lack .name_to_index is unusable; leave attribute unset."""
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": {"era5": object()}},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    WeightsOnlyLoader().process(context)

    assert not hasattr(model, "_ckpt_model_name_to_index")
