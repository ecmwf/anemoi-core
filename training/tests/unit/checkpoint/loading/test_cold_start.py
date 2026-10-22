# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ColdStartLoader."""

import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import ColdStartLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_cold_start_loads_weights() -> None:
    model = SimpleModel()
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = ColdStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = loader.process(context)

    assert result.model is not None
    assert torch.equal(result.model.linear.weight, checkpoint_data["state_dict"]["linear.weight"])
    assert torch.equal(result.model.linear.bias, checkpoint_data["state_dict"]["linear.bias"])


def test_cold_start_resets_training_state() -> None:
    """Epoch and global_step must be 0 regardless of checkpoint content."""
    model = SimpleModel()
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "epoch": 42,
        "global_step": 10000,
    }

    loader = ColdStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = loader.process(context)

    assert result.metadata.get("epoch", 0) == 0
    assert result.metadata.get("global_step", 0) == 0


def test_cold_start_tracks_pretrained_source() -> None:
    model = SimpleModel()
    ckpt_path = "/path/to/pretrained.ckpt"
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = ColdStartLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data, checkpoint_path=ckpt_path)
    result = loader.process(context)

    assert result.metadata["pretrained_from"] == str(result.checkpoint_path)
    assert result.metadata["pretrained_from"] == ckpt_path


def _checkpoint_for(model: nn.Module) -> dict:
    return {"state_dict": {key: torch.full_like(value, 1.0) for key, value in model.state_dict().items()}}


def test_cold_start_inherits_the_strict_parameter() -> None:
    assert ColdStartLoader(strict=False).strict is False
    assert ColdStartLoader().strict is True


def test_cold_start_records_the_source_url_when_no_local_path_is_published() -> None:
    """A remote source deletes its download, so its URL is the recorded provenance."""
    model = SimpleModel()
    context = CheckpointContext(
        model=model,
        checkpoint_data=_checkpoint_for(model),
        metadata={"source_url": "https://example.com/pretrained.ckpt"},
    )

    result = ColdStartLoader().process(context)

    assert result.metadata["pretrained_from"] == "https://example.com/pretrained.ckpt"


def test_cold_start_records_no_provenance_without_a_path_or_url() -> None:
    model = SimpleModel()

    result = ColdStartLoader().process(CheckpointContext(model=model, checkpoint_data=_checkpoint_for(model)))

    assert result.metadata["pretrained_from"] is None
