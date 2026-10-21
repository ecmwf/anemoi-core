# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for WeightsOnlyLoader."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.base import warn_on_hparams_divergence
from anemoi.training.checkpoint.loading.strategies import ColdStartLoader
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_weights_only_extends_pipeline_stage() -> None:
    assert issubclass(WeightsOnlyLoader, PipelineStage)


def test_weights_only_loads_state_dict() -> None:
    model = SimpleModel()
    original_weight = model.linear.weight.clone()

    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = loader.process(context)

    # Weights changed
    assert not torch.equal(result.model.linear.weight, original_weight)


def test_weights_only_sets_metadata() -> None:
    model = SimpleModel()
    checkpoint_data = {"state_dict": {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}}

    loader = WeightsOnlyLoader()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    result = loader.process(context)

    assert result.metadata.get("loading_strategy") == "weights_only"


@pytest.mark.parametrize(
    ("state_dict", "strict", "raises"),
    [
        ({"linear.weight": torch.randn(5, 10)}, True, True),
        ({"linear.weight": torch.randn(5, 10)}, False, False),
        ({"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5), "extra": torch.randn(2)}, True, True),
        ({"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5), "extra": torch.randn(2)}, False, False),
        ({"linear.weight": torch.randn(7, 10), "linear.bias": torch.randn(5)}, False, True),
    ],
    ids=["strict-missing", "non-strict-missing", "strict-extra", "non-strict-extra", "non-strict-shape-mismatch"],
)
def test_weights_only_strict_rejects_a_key_mismatch(state_dict: dict, strict: bool, raises: bool) -> None:
    """``strict`` (the default) rejects missing and extra keys. ``strict=False`` loads the matching keys.

    ``strict=False`` does not skip a shape mismatch. Use ``skip_mismatched`` for that.
    """
    model = SimpleModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": state_dict})

    if raises:
        with pytest.raises(CheckpointLoadError):
            WeightsOnlyLoader(strict=strict).process(context)
    else:
        WeightsOnlyLoader(strict=strict).process(context)
        assert torch.equal(model.linear.weight, state_dict["linear.weight"])


def test_weights_only_rejects_a_missing_key_by_default() -> None:
    """A loader built without ``strict`` is strict: a checkpoint missing a key is rejected."""
    model = SimpleModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": {"linear.weight": torch.randn(5, 10)}})

    with pytest.raises(CheckpointLoadError):
        WeightsOnlyLoader().process(context)


@pytest.mark.parametrize(
    ("checkpoint_dtype", "model_dtype"),
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
    ids=["fp32-into-bf16", "bf16-into-fp32"],
)
def test_weights_only_casts_to_the_model_dtype(checkpoint_dtype: torch.dtype, model_dtype: torch.dtype) -> None:
    """A checkpoint in a different precision loads. The parameters keep the model dtype."""
    checkpoint = {key: value.to(checkpoint_dtype) for key, value in SimpleModel().state_dict().items()}
    model = SimpleModel().to(model_dtype)

    WeightsOnlyLoader().process(CheckpointContext(model=model, checkpoint_data={"state_dict": checkpoint}))

    for key, value in model.state_dict().items():
        assert value.dtype == model_dtype, key
        assert torch.equal(value, checkpoint[key].to(model_dtype)), key


@pytest.mark.parametrize(("checkpoint_channels", "warns"), [(64, True), (128, False)])
def test_weights_only_warns_when_checkpoint_hparams_differ(
    caplog: pytest.LogCaptureFixture,
    checkpoint_channels: int,
    warns: bool,
) -> None:
    """A different model config in the checkpoint causes a warning."""
    model = SimpleModel()
    checkpoint = {
        "state_dict": {key: value.clone() for key, value in model.state_dict().items()},
        "hyper_parameters": {"config": {"model": {"num_channels": checkpoint_channels}}},
    }
    config = OmegaConf.create({"model": {"num_channels": 128}})
    context = CheckpointContext(model=model, checkpoint_data=checkpoint, config=config)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader().process(context)

    assert ("hparams differ" in caplog.text) is warns


@pytest.mark.parametrize(
    ("checkpoint", "run_config"),
    [
        ({"hyper_parameters": {"config": {"model": {"layers": 1}}}}, None),
        ({"hyper_parameters": [1, 2, 3]}, {"model": {"layers": 2}}),
        ({"hyper_parameters": {"config": None}}, {"model": {"layers": 2}}),
        ({"hyper_parameters": {"config": {"model": {"layers": "${missing_key}"}}}}, {"model": {"layers": 2}}),
        (None, {"model": {"layers": 2}}),
    ],
    ids=[
        "no-run-config",
        "hparams-not-a-mapping",
        "no-checkpoint-config",
        "unresolvable-checkpoint-config",
        "no-checkpoint-data",
    ],
)
def test_hparams_divergence_check_is_best_effort(
    caplog: pytest.LogCaptureFixture,
    checkpoint: dict | None,
    run_config: dict | None,
) -> None:
    """If the configs cannot be compared, the check does not raise and does not warn."""
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence(checkpoint, OmegaConf.create(run_config) if run_config else None)

    assert "hparams differ" not in caplog.text


class TwoLayerModel(nn.Module):
    """A trunk shared with the checkpoint, plus a variable-dependent head."""

    def __init__(self, head_out: int) -> None:
        super().__init__()
        self.trunk = nn.Linear(10, 10)
        self.head = nn.Linear(10, head_out)


def _checkpoint_for(model: nn.Module) -> dict:
    return {"state_dict": {key: value.clone() for key, value in model.state_dict().items()}}


def test_weights_only_skip_mismatched_loads_the_compatible_parameters() -> None:
    """Fine-tuning onto fewer variables loads the trunk and re-initialises the head.

    `strict=False` cannot express this: PyTorch records a size mismatch outside its
    `if strict:` block, so the load raised before the variable-subset check was ever
    reached. That is what made the advertised `allow_variable_subset` workflow
    unusable on this strategy.
    """
    source = TwoLayerModel(head_out=8)
    target = TwoLayerModel(head_out=4)
    head_before = target.head.weight.clone()

    context = CheckpointContext(model=target, checkpoint_data=_checkpoint_for(source))
    result = WeightsOnlyLoader(strict=False, skip_mismatched=True).process(context)

    # Trunk transferred; head left at its initialised value.
    assert torch.equal(target.trunk.weight, source.trunk.weight)
    assert torch.equal(target.head.weight, head_before)
    assert result.metadata["skipped_params"].keys() == {"head.weight", "head.bias"}
    assert result.metadata["skipped_params"]["head.weight"].startswith("Shape mismatch")
    assert getattr(target, "weights_initialized", False) is True


def test_weights_only_skip_mismatched_warns_about_untrained_parameters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The skip is loud: nothing else would reveal that part of the model is random."""
    context = CheckpointContext(
        model=TwoLayerModel(head_out=4),
        checkpoint_data=_checkpoint_for(TwoLayerModel(head_out=8)),
    )

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader(strict=False, skip_mismatched=True).process(context)

    assert "SKIPPED" in caplog.text
    assert "head.weight" in caplog.text


def test_weights_only_skip_mismatched_does_not_drop_unexpected_keys() -> None:
    """Only shapes are skipped; the key set stays ``strict``'s business.

    Dropping keys the model does not have would make ``strict=True``
    unfalsifiable, so a genuinely wrong checkpoint still fails.
    """
    target = TwoLayerModel(head_out=4)
    checkpoint = _checkpoint_for(TwoLayerModel(head_out=8))
    checkpoint["state_dict"]["not_a_real_layer.weight"] = torch.randn(3, 3)

    context = CheckpointContext(model=target, checkpoint_data=checkpoint)
    with pytest.raises(CheckpointIncompatibleError, match="unexpected="):
        WeightsOnlyLoader(strict=True, skip_mismatched=True).process(context)


def test_weights_only_skip_mismatched_refuses_an_empty_load() -> None:
    """Skipping *everything* is the wrong checkpoint, not a deliberate reduction."""
    context = CheckpointContext(
        model=nn.Linear(10, 4),
        checkpoint_data={"state_dict": {"weight": torch.randn(8, 10), "bias": torch.randn(8)}},
    )

    with pytest.raises(CheckpointIncompatibleError, match="Every parameter"):
        WeightsOnlyLoader(strict=False, skip_mismatched=True).process(context)


def test_cold_start_inherits_skip_mismatched() -> None:
    """ColdStartLoader takes WeightsOnlyLoader's constructor, so it gets this too."""
    source = TwoLayerModel(head_out=8)
    target = TwoLayerModel(head_out=4)

    context = CheckpointContext(model=target, checkpoint_data=_checkpoint_for(source))
    result = ColdStartLoader(strict=False, skip_mismatched=True).process(context)

    assert torch.equal(target.trunk.weight, source.trunk.weight)
    assert result.metadata["skipped_params"].keys() == {"head.weight", "head.bias"}
    assert result.metadata["loading_strategy"] == "cold_start"


def test_weights_only_does_not_write_training_progress_metadata() -> None:
    """The checkpoint's epoch and global step are not carried into a weights-only load."""
    model = SimpleModel()
    checkpoint_data = {
        "state_dict": {key: torch.ones_like(value) for key, value in model.state_dict().items()},
        "epoch": 42,
        "global_step": 10000,
    }

    result = WeightsOnlyLoader().process(CheckpointContext(model=model, checkpoint_data=checkpoint_data))

    assert "epoch" not in result.metadata
    assert "global_step" not in result.metadata
