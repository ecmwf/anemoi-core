# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for WarmStartLoader.

Warm start is a marker: selecting it means ``Trainer.fit(ckpt_path=)`` performs
the one and only load of the checkpoint (weights, optimizer, scheduler and loop
progress together) and the pipeline only resolves the source to a local file. The
builder never emits this stage, so ``process`` refuses to run.
"""

import inspect

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.base import LoadingStrategy
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_warm_start_declares_restores_training_state() -> None:
    """The attribute the builder and the trainer read to route the load through Lightning."""
    assert WarmStartLoader.restores_training_state is True
    assert LoadingStrategy.restores_training_state is False


def test_warm_start_keeps_its_constructor_signature() -> None:
    """``loading=warm_start`` configs carry no parameters; the class must keep accepting none."""
    assert "strict" not in inspect.signature(WarmStartLoader.__init__).parameters
    assert isinstance(WarmStartLoader(), LoadingStrategy)


@pytest.mark.asyncio
async def test_warm_start_process_refuses_to_run() -> None:
    """A hand-built pipeline that routes a checkpoint through the marker fails loudly, untouched."""
    model = SimpleModel()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)

    with pytest.raises(
        RuntimeError,
        match=r"WarmStartLoader is a marker: resume loads happen in Trainer\.fit\(ckpt_path=\)",
    ):
        await WarmStartLoader().process(context)

    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)
    assert not getattr(model, "weights_initialized", False)
