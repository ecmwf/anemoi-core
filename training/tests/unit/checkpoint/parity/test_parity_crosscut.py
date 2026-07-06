# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cross-cutting parity tests for the checkpoint pipeline.

These close genuine coverage gaps that span more than one layer of the
checkpoint pipeline: cross-dtype loading, load determinism / idempotency,
end-to-end metadata preservation, the deprecated-key migration advice, and
per-preset config composition of the ``training.checkpoint`` surface.

Everything here is CPU-only and deterministic. Load-probe tests build a small
``nn.Module`` plus a synthetic checkpoint, construct a real
:class:`CheckpointContext`, and drive the real loading strategy / source /
builder — no mocking of the code under test.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import Never

import pytest
import torch
import torch.nn as nn
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.formats import extract_state_dict
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader
from anemoi.training.checkpoint.sources.local import LocalSource
from anemoi.training.schemas.base_schema import _DEPRECATED_KEYS
from anemoi.training.schemas.training import CheckpointPipelineSchema
from anemoi.training.tasks.forecaster import Forecaster
from anemoi.training.train.methods.base import BaseTrainingModule

if TYPE_CHECKING:
    from pathlib import Path

_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunSource"
_LOCAL_SOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"
_WEIGHTS_ONLY = "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"
_TRANSFER_LEARNING = "anemoi.training.checkpoint.loading.strategies.TransferLearningLoader"
_FREEZING = "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage"


class _SmallNet(nn.Module):
    """Minimal two-layer CPU model for load probes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _index_collection(name_to_index: dict[str, int]) -> object:
    """Build an object exposing ``.name_to_index`` like a real IndexCollection."""
    return type("IndexCollection", (), {"name_to_index": name_to_index})()


# --- mixed precision / dtype load (fp32 <-> bf16) ---------------------


def test_weights_only_casts_fp32_state_dict_into_bfloat16_model() -> None:
    """Loading an fp32 state dict into a bf16 model casts to the model dtype, no corruption."""
    torch.manual_seed(0)
    source = _SmallNet()
    source_state = {key: value.clone() for key, value in source.state_dict().items()}

    target = _SmallNet().to(torch.bfloat16)
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    loaded = target.state_dict()
    for key, source_value in source_state.items():
        assert loaded[key].dtype == torch.bfloat16, f"{key} lost the model dtype"
        # Values equal the fp32 source cast down to bf16 — the only lossy step is the
        # documented cast, never a silent reinterpret of the raw bytes.
        assert torch.equal(loaded[key], source_value.to(torch.bfloat16))


def test_weights_only_casts_bfloat16_state_dict_into_fp32_model() -> None:
    """Loading a bf16 state dict into an fp32 model widens to fp32 with matching values."""
    torch.manual_seed(0)
    source = _SmallNet().to(torch.bfloat16)
    source_state = {key: value.clone() for key, value in source.state_dict().items()}

    target = _SmallNet()  # fp32
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    loaded = target.state_dict()
    for key, source_value in source_state.items():
        assert loaded[key].dtype == torch.float32, f"{key} lost the model dtype"
        assert torch.equal(loaded[key], source_value.to(torch.float32))


# --- pipeline determinism (two identical load-probes match) -----------


def test_same_checkpoint_loads_identically_into_two_models(tmp_path: Path) -> None:
    """The same checkpoint loaded into two fresh models yields bit-exact state dicts."""
    torch.manual_seed(0)
    source = _SmallNet()
    ckpt_path = tmp_path / "checkpoint.ckpt"
    torch.save({"state_dict": source.state_dict()}, ckpt_path)

    # Two models built from DIFFERENT initial weights: if loading failed to overwrite
    # every slot, the post-load state dicts would diverge and the test would fail.
    torch.manual_seed(1)
    model_a = _SmallNet()
    torch.manual_seed(2)
    model_b = _SmallNet()
    assert not torch.equal(model_a.state_dict()["fc1.weight"], model_b.state_dict()["fc1.weight"])

    for model in (model_a, model_b):
        context = CheckpointContext(checkpoint_path=ckpt_path, model=model)
        asyncio.run(LocalSource().process(context))
        asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    source_state = source.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        # atol=0: identical bytes, not merely close.
        assert torch.equal(state_a[key], state_b[key]), f"{key} differs between loads"
        assert torch.equal(state_a[key], source_state[key]), f"{key} does not match the source"


# --- double-load idempotency (pipeline + Lightning ckpt_path) ---------


class _StepStub(BaseTrainingModule):
    """Concrete BaseTrainingModule whose training step is unused in these tests."""

    def __init__(self) -> None:
        pass

    def _step(self, batch: object, validation_mode: bool = False) -> Never:
        raise NotImplementedError


def _lightning_module_wrapping(model: nn.Module) -> _StepStub:
    """Build a minimal training module whose ``.model`` is ``model``, parity refresh off."""
    from types import SimpleNamespace

    module = _StepStub.__new__(_StepStub)
    nn.Module.__init__(module)
    module.task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    module.model = model
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=SimpleNamespace(states=False, tendencies=False)),
    )
    return module


def test_double_load_via_pipeline_then_lightning_is_bit_exact() -> None:
    """Pipeline load then a second Lightning ckpt_path pass leaves the model bit-exact.

    A warm-start resume applies the checkpoint twice: once by the pipeline at model
    build (which sets ``weights_initialized``), then again by Lightning's ``ckpt_path``
    at fit time. The second pass must not perturb the weights, and ``on_load_checkpoint``
    must skip its parity steps because the pipeline already ran them.
    """
    torch.manual_seed(1)
    source = _SmallNet()
    checkpoint = {"state_dict": {key: value.clone() for key, value in source.state_dict().items()}}

    torch.manual_seed(99)
    model = _SmallNet()
    context = CheckpointContext(model=model, checkpoint_data=checkpoint)
    asyncio.run(WarmStartLoader().process(context))
    after_pipeline = {key: value.clone() for key, value in model.state_dict().items()}
    assert getattr(model, "weights_initialized", False) is True

    # Second application: Lightning's ckpt_path reloads the same weights, then fires
    # the on_load_checkpoint hook (which must short-circuit on weights_initialized).
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    module = _lightning_module_wrapping(model)
    module.weights_initialized = True
    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    after_second = model.state_dict()
    for key, value in after_pipeline.items():
        assert torch.equal(after_second[key], value), f"{key} drifted across the second load"
    # The parity-skip guard held: no checkpoint metadata was re-derived onto the module.
    assert not hasattr(module, "_ckpt_model_name_to_index")


# --- metadata round-trip through a weights-only load ------------------


def test_weights_only_preserves_name_index_variables_metadata_and_hyper_parameters() -> None:
    """A weights-only load preserves name_to_index, variables_metadata and hyper_parameters."""
    torch.manual_seed(0)
    name_to_index = {"t2m": 0, "u10": 1, "v10": 2}
    variables_metadata = {"t2m": {"units": "K"}, "u10": {"units": "m s**-1"}, "v10": {"units": "m s**-1"}}
    hyper_parameters = {
        "data_indices": {"era5": _index_collection(name_to_index)},
        "metadata": {"dataset": {"era5": {"variables_metadata": variables_metadata}}},
    }
    model = _SmallNet()
    checkpoint_data = {
        "state_dict": {key: torch.randn_like(value) for key, value in model.state_dict().items()},
        "hyper_parameters": hyper_parameters,
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    assert model._ckpt_model_name_to_index == {"era5": name_to_index}
    assert model._ckpt_variables_metadata == {"era5": variables_metadata}
    # hyper_parameters must survive the load untouched (same object, equal content).
    assert context.checkpoint_data["hyper_parameters"] is hyper_parameters
    assert (
        context.checkpoint_data["hyper_parameters"]["metadata"]["dataset"]["era5"]["variables_metadata"]
        == variables_metadata
    )


# --- the deprecated-key migration advice actually builds a pipeline ---


def _replacement_checkpoint_config(deprecated_key: str) -> dict:
    """The ``training.checkpoint`` block the hint for ``deprecated_key`` recommends."""
    return {
        "training.run_id": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc", "fork": False}},
        "training.fork_run_id": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc", "fork": True}},
        "system.input.warm_start": {"source": {"_target_": _LOCAL_SOURCE, "path": "/scratch/last.ckpt"}},
        "training.load_weights_only": {
            "source": {"_target_": _RUN_SOURCE, "run_id": "abc"},
            "loading": {"_target_": _WEIGHTS_ONLY},
        },
        "training.transfer_learning": {
            "source": {"_target_": _LOCAL_SOURCE, "path": "/scratch/last.ckpt"},
            "loading": {"_target_": _TRANSFER_LEARNING, "skip_mismatched": True},
        },
        "training.submodules_to_freeze": {
            "modifiers": [{"_target_": _FREEZING, "submodules_to_freeze": ["encoder"]}],
        },
    }[deprecated_key]


def test_replacement_config_map_covers_every_deprecated_key() -> None:
    """The advice-under-test map stays in lockstep with the real _DEPRECATED_KEYS registry."""
    covered = {
        "training.run_id",
        "training.fork_run_id",
        "system.input.warm_start",
        "training.load_weights_only",
        "training.transfer_learning",
        "training.submodules_to_freeze",
    }
    assert covered == set(_DEPRECATED_KEYS)


@pytest.mark.parametrize("deprecated_key", sorted(_DEPRECATED_KEYS))
def test_deprecated_key_replacement_config_builds_and_validates(deprecated_key: str) -> None:
    """Each removed key's recommended replacement composes into a valid, buildable pipeline."""
    if deprecated_key == "training.submodules_to_freeze":
        pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing", reason="PR #442")

    checkpoint_block = _replacement_checkpoint_config(deprecated_key)

    # The schema that governs this surface must accept the advice.
    CheckpointPipelineSchema(**checkpoint_block)

    cfg = OmegaConf.create({"training": {"checkpoint": checkpoint_block}})
    pipeline = build_checkpoint_pipeline(cfg)
    stage_names = [type(stage).__name__ for stage in pipeline.stages]

    if "source" in checkpoint_block:
        assert stage_names[0].endswith("Source")
    if "loading" in checkpoint_block:
        assert any(name.endswith("Loader") for name in stage_names)
    if "modifiers" in checkpoint_block:
        assert any(name.endswith("Stage") for name in stage_names)

    # Non-tautological attribute checks: the advice configures the intended behaviour.
    if deprecated_key == "training.run_id":
        source = pipeline.stages[0]
        assert source.run_id == "abc"
        assert source.fork is False
    elif deprecated_key == "training.fork_run_id":
        source = pipeline.stages[0]
        assert source.run_id == "abc"
        assert source.fork is True
    elif deprecated_key == "system.input.warm_start":
        assert isinstance(pipeline.stages[0], LocalSource)
    elif deprecated_key == "training.transfer_learning":
        loader = pipeline.stages[1]
        assert type(loader).__name__ == "TransferLearningLoader"
        assert loader.skip_mismatched is True
    elif deprecated_key == "training.submodules_to_freeze":
        assert pipeline.stages[0].submodules_to_freeze == ["encoder"]


# --- each preset composes with a checkpoint.source overlay ------------

_PRESETS = ["config", "lam", "multi", "stretched", "ensemble_crps"]


@pytest.mark.parametrize("preset", _PRESETS)
def test_preset_composes_with_checkpoint_source_overlay(preset: str) -> None:
    """Adding a RunSource overlay to each preset composes cleanly and yields a valid pipeline.

    This exercises the composed ``training.checkpoint`` surface per preset (no defaults
    conflict from the ``training/checkpoint/source`` group) and validates it with the
    schema and builder that own it.
    """
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        cfg = compose(
            config_name=preset,
            overrides=[
                "training/checkpoint/source=run",
                "+training.checkpoint.source.run_id=abc123",
            ],
        )

    checkpoint_block = OmegaConf.to_container(cfg.training.checkpoint, resolve=True)
    CheckpointPipelineSchema(**checkpoint_block)

    pipeline = build_checkpoint_pipeline(cfg)
    assert type(pipeline.stages[0]).__name__ == "RunSource"
    assert pipeline.stages[0].run_id == "abc123"
