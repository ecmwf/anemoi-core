# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for _preserve_anemoi_metadata multi-dataset handling.

PR #998 review (Ana): production code at
``anemoi.training.train.tasks.base.AnemoiLightningModule.on_load_checkpoint``
builds ``_ckpt_model_name_to_index`` as a dict keyed by dataset name
because ``hyper_parameters["data_indices"]`` is now
``dict[str, IndexCollection]``. The pipeline must handle both that shape
and the legacy single-IndexCollection shape.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def _fake_index_collection(name_to_index: dict[str, int]) -> object:
    return type("IndexCollection", (), {"name_to_index": name_to_index})()


@pytest.mark.asyncio
async def test_multi_dataset_data_indices_builds_dataset_keyed_dict() -> None:
    """Dict-of-IndexCollections shape produces a dict-of-name_to_index attribute."""
    era5 = {"temperature": 0, "wind_u": 1}
    metno = {"temperature": 0, "humidity": 1, "pressure": 2}
    data_indices = {
        "era5": _fake_index_collection(era5),
        "metno": _fake_index_collection(metno),
    }
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": data_indices},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    await WeightsOnlyLoader().process(context)

    assert model._ckpt_model_name_to_index == {"era5": era5, "metno": metno}


@pytest.mark.asyncio
async def test_single_dataset_legacy_shape_still_works() -> None:
    """Legacy single-IndexCollection shape produces a flat name_to_index attribute."""
    name_to_index = {"temperature": 0, "wind_u": 1, "wind_v": 2}
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": _fake_index_collection(name_to_index)},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    await WeightsOnlyLoader().process(context)

    assert model._ckpt_model_name_to_index == name_to_index


@pytest.mark.asyncio
async def test_empty_dict_data_indices_does_not_set_attribute() -> None:
    """An empty dict is not a usable multi-dataset payload; leave attribute unset."""
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": {}},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    await WeightsOnlyLoader().process(context)

    assert not hasattr(model, "_ckpt_model_name_to_index")


@pytest.mark.asyncio
async def test_multi_dataset_dict_entries_missing_name_to_index_skips() -> None:
    """Dict whose values lack .name_to_index is unusable; leave attribute unset."""
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": {"data_indices": {"era5": object()}},
    }
    model = _Model()

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    await WeightsOnlyLoader().process(context)

    assert not hasattr(model, "_ckpt_model_name_to_index")
