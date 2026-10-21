# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for Anemoi metadata handling during checkpoint loading."""

import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class ModelWithMetadata(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self._ckpt_model_name_to_index = None


def test_preserves_ckpt_model_name_to_index() -> None:
    """All loaders must preserve _ckpt_model_name_to_index from hyper_parameters."""
    model = ModelWithMetadata()
    name_to_index = {"temperature": 0, "wind_u": 1, "wind_v": 2}
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        # Multi-dataset shape: data_indices is dict[str, IndexCollection].
        "hyper_parameters": {"data_indices": {"era5": type("obj", (), {"name_to_index": name_to_index})()}},
    }

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = loader.process(context)

    assert result.model._ckpt_model_name_to_index == {"era5": name_to_index}


def test_weights_only_sets_ckpt_variables_metadata() -> None:
    """The loader also sets ``_ckpt_variables_metadata``. The checkpoint hyper_parameters stay unchanged."""
    model = ModelWithMetadata()
    variables_metadata = {"2t": {"units": "K"}}
    hyper_parameters = {
        "data_indices": {"era5": type("obj", (), {"name_to_index": {"2t": 0}})()},
        "metadata": {"dataset": {"era5": {"variables_metadata": variables_metadata}}},
    }
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "hyper_parameters": hyper_parameters,
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    WeightsOnlyLoader().process(context)

    assert model._ckpt_variables_metadata == {"era5": variables_metadata}
    assert context.checkpoint_data["hyper_parameters"] is hyper_parameters
