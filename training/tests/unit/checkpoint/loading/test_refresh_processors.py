"""Regression tests for LoadingStrategy._refresh_checkpoint_processors.

Mirrors anemoi.training.train.tasks.base.AnemoiLightningModule
._update_checkpoint_state_dict_for_load so that pipeline-based loading
honours config.training.update_ds_stats_on_ckpt_load.{states,tendencies}.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class _ModelWithProcessors(nn.Module):
    """Fake top-level model wrapping pre/post processors and a body."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_processors = nn.Linear(4, 4)
        self.post_processors = nn.Linear(4, 4)
        self.pre_processors_tendencies = nn.Linear(4, 4)
        self.post_processors_tendencies = nn.Linear(4, 4)
        self.body = nn.Linear(4, 4)


def _ckpt_with_stale_processors(stale_value: float) -> dict:
    """Build a checkpoint state_dict whose processor weights are obviously stale."""
    state_dict = {}
    for prefix in (
        "model.pre_processors",
        "model.post_processors",
        "model.pre_processors_tendencies",
        "model.post_processors_tendencies",
    ):
        state_dict[f"{prefix}.weight"] = torch.full((4, 4), stale_value)
        state_dict[f"{prefix}.bias"] = torch.full((4,), stale_value)
    state_dict["model.body.weight"] = torch.randn(4, 4)
    state_dict["model.body.bias"] = torch.randn(4)
    return {"state_dict": state_dict}


def _config(*, states: bool, tendencies: bool) -> OmegaConf:
    return OmegaConf.create(
        {"training": {"update_ds_stats_on_ckpt_load": {"states": states, "tendencies": tendencies}}},
    )


def _build_context(*, states: bool, tendencies: bool, stale_value: float = 99.0) -> CheckpointContext:
    return CheckpointContext(
        model=_ModelWithProcessors(),
        checkpoint_data=_ckpt_with_stale_processors(stale_value),
        config=_config(states=states, tendencies=tendencies),
    )


def test_no_op_when_both_flags_false() -> None:
    """Default config (states=False, tendencies=False) does not touch the state dict."""
    context = _build_context(states=False, tendencies=False)
    before = {k: v.clone() for k, v in context.checkpoint_data["state_dict"].items()}

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    for key, original in before.items():
        assert torch.equal(context.checkpoint_data["state_dict"][key], original)


def test_states_flag_replaces_state_processor_weights() -> None:
    """states=True swaps model.pre_processors.* and model.post_processors.* in place."""
    context = _build_context(states=True, tendencies=False)

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    state_dict = context.checkpoint_data["state_dict"]
    model_state = context.model.state_dict()
    assert torch.equal(state_dict["model.pre_processors.weight"], model_state["pre_processors.weight"])
    assert torch.equal(state_dict["model.post_processors.bias"], model_state["post_processors.bias"])
    # Tendency processors untouched because tendencies=False
    assert torch.all(state_dict["model.pre_processors_tendencies.weight"] == 99.0)


def test_tendencies_flag_replaces_tendency_processor_weights() -> None:
    """tendencies=True swaps the *_tendencies processor entries."""
    context = _build_context(states=False, tendencies=True)

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    state_dict = context.checkpoint_data["state_dict"]
    model_state = context.model.state_dict()
    assert torch.equal(
        state_dict["model.pre_processors_tendencies.weight"],
        model_state["pre_processors_tendencies.weight"],
    )
    # State processors untouched because states=False
    assert torch.all(state_dict["model.pre_processors.weight"] == 99.0)


def test_both_flags_replaces_all_four_processor_groups() -> None:
    context = _build_context(states=True, tendencies=True)

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    state_dict = context.checkpoint_data["state_dict"]
    model_state = context.model.state_dict()
    for short_prefix in (
        "pre_processors",
        "post_processors",
        "pre_processors_tendencies",
        "post_processors_tendencies",
    ):
        assert torch.equal(
            state_dict[f"model.{short_prefix}.weight"],
            model_state[f"{short_prefix}.weight"],
        )


def test_body_weights_are_not_touched() -> None:
    """Non-processor keys (body, encoder, etc.) must survive untouched."""
    context = _build_context(states=True, tendencies=True)
    before_body = context.checkpoint_data["state_dict"]["model.body.weight"].clone()

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    assert torch.equal(context.checkpoint_data["state_dict"]["model.body.weight"], before_body)


def test_missing_config_is_no_op() -> None:
    """Context with no config (e.g. unit tests of strategies in isolation) is safe."""
    context = CheckpointContext(
        model=_ModelWithProcessors(),
        checkpoint_data=_ckpt_with_stale_processors(99.0),
        config=None,
    )
    before = {k: v.clone() for k, v in context.checkpoint_data["state_dict"].items()}

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    for key, original in before.items():
        assert torch.equal(context.checkpoint_data["state_dict"][key], original)


@pytest.mark.asyncio
async def test_full_process_call_applies_the_refresh() -> None:
    """Smoke test: the helper actually runs as part of WeightsOnlyLoader.process()."""
    context = _build_context(states=True, tendencies=True)

    await WeightsOnlyLoader().process(context)

    # After process() the model's own processor weights end up in the model
    # (state_dict was rewritten then loaded back), so picking any value != 99.0 is enough.
    loaded = context.model.state_dict()["pre_processors.weight"]
    assert not torch.all(loaded == 99.0)
