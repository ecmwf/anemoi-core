# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for WarmStartLoader.

Warm start loads weights (strict) + parity at model-build, exactly like the
other strategies. The optimizer / scheduler / loop-progress restore is owned by
Lightning's ``ckpt_path`` resume at fit time (the pipeline runs before those
objects exist), signalled by :attr:`WarmStartLoader.restores_training_state`.
"""

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_warm_start_declares_restores_training_state() -> None:
    """Warm start is the strategy whose optimizer/loop restore is owned by Lightning ckpt_path."""
    assert WarmStartLoader.restores_training_state is True


@pytest.mark.asyncio
async def test_warm_start_restores_model_weights() -> None:
    model = SimpleModel()
    saved_weight = torch.randn(5, 10)
    checkpoint_data = {"state_dict": {"linear.weight": saved_weight, "linear.bias": torch.randn(5)}}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert torch.equal(result.model.linear.weight, saved_weight)
    assert result.model.weights_initialized is True
    assert result.metadata["loading_strategy"] == "warm_start"


@pytest.mark.asyncio
async def test_warm_start_loads_without_optimizer_in_context() -> None:
    """At model-build the optimizer does not exist yet; warm start must still load weights.

    Optimizer / scheduler / loop-progress restore is deferred to Lightning's
    ``ckpt_path``, so the loader neither requires nor touches an optimizer — even
    when the checkpoint carries ``optimizer_states``.
    """
    model = SimpleModel()
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "optimizer_states": [{"irrelevant": True}],
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)  # no optimizer in context

    assert result.optimizer is None
    assert result.model.weights_initialized is True


@pytest.mark.asyncio
async def test_warm_start_surfaces_training_state_on_context() -> None:
    """Warm start records the extracted training progress on ``context.metadata``.

    Observational only: Lightning's ``ckpt_path`` owns the live restore, but the
    extracted epoch/global_step are surfaced on the context for inspection/tooling.
    """
    model = SimpleModel()
    checkpoint_data = {
        "state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)},
        "epoch": 42,
        "global_step": 10000,
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = await WarmStartLoader().process(context)

    assert result.metadata["epoch"] == 42
    assert result.metadata["global_step"] == 10000


@pytest.mark.asyncio
async def test_warm_start_requires_exact_match() -> None:
    """Warm start expects the same architecture: a shape/key mismatch raises."""
    model = SimpleModel()
    # Wrong shape and a missing key → strict load fails.
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(3, 3)}}

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    with pytest.raises(CheckpointIncompatibleError):
        await WarmStartLoader().process(context)
