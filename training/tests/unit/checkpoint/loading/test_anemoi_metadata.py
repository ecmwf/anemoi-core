# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for Anemoi metadata handling during checkpoint loading."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.base import extract_checkpoint_variables_metadata
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


class _SmallNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 2)


def test_weights_only_preserves_name_to_index_variables_metadata_and_hyper_parameters() -> None:
    """A weights-only load restores both index maps exactly and leaves hyper_parameters untouched."""
    torch.manual_seed(0)
    name_to_index = {"t2m": 0, "u10": 1, "v10": 2}
    variables_metadata = {"t2m": {"units": "K"}, "u10": {"units": "m s**-1"}, "v10": {"units": "m s**-1"}}
    hyper_parameters = {
        "data_indices": {"era5": type("IndexCollection", (), {"name_to_index": name_to_index})()},
        "metadata": {"dataset": {"era5": {"variables_metadata": variables_metadata}}},
    }
    model = _SmallNet()
    checkpoint_data = {
        "state_dict": {key: torch.randn_like(value) for key, value in model.state_dict().items()},
        "hyper_parameters": hyper_parameters,
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    WeightsOnlyLoader(strict=True).process(context)

    assert model._ckpt_model_name_to_index == {"era5": name_to_index}
    assert model._ckpt_variables_metadata == {"era5": variables_metadata}
    assert context.checkpoint_data["hyper_parameters"] is hyper_parameters
    assert (
        context.checkpoint_data["hyper_parameters"]["metadata"]["dataset"]["era5"]["variables_metadata"]
        == variables_metadata
    )


def test_extract_variables_metadata_noop_without_name_to_index() -> None:
    """Without a restored ``name_to_index`` there is nothing to key the metadata by, so nothing is set."""
    model = SimpleNamespace()
    extract_checkpoint_variables_metadata(model, {})
    assert not hasattr(model, "_ckpt_variables_metadata")


def test_extract_variables_metadata_noop_without_checkpoint_data() -> None:
    """With ``name_to_index`` set but no checkpoint, the call is a safe no-op.

    It guards on a ``None`` checkpoint instead of dereferencing it, so it stays safe to
    call unconditionally after a load that produced no checkpoint data.
    """
    model = SimpleNamespace(_ckpt_model_name_to_index={"era5": {"2t": 0}})
    extract_checkpoint_variables_metadata(model, None)
    assert not hasattr(model, "_ckpt_variables_metadata")
