# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Adapter record in checkpoint metadata, and restoring adapters from it on load."""

import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Never
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from torch import nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader
from anemoi.training.checkpoint.modifiers.peft import PeftModifierStage
from anemoi.training.checkpoint.modifiers.peft import adapter_record
from anemoi.training.checkpoint.pipeline import CheckpointPipeline
from anemoi.training.checkpoint.sources.local import LocalSource
from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.train.methods.base import BaseTrainingModule

from .peft_lora_checkpoint_fixture import RECORD
from .peft_lora_checkpoint_fixture import STATE_DICT_SHAPES
from .test_peft import Interface
from .test_peft import TrainingModule
from .test_peft import adapted_paths
from .test_peft import lora

pytest.importorskip("peft")

NO_CONFIG = SimpleNamespace()


def with_lora(model: nn.Module | None = None, **options: object) -> nn.Module:
    """A training module with LoRA applied and non-zero adapter weights."""
    model = model if model is not None else TrainingModule()
    trainable_modules = options.pop("trainable", None)
    options.setdefault("target_modules", r"model\.model\.processor\..*")
    stage = PeftModifierStage(peft_config=lora(**options), trainable_modules=trainable_modules)
    stage.process(CheckpointContext(model=model))
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_B." in name:
                param.copy_(torch.randn(param.shape, generator=generator))
    return model


def save(model: nn.Module, **metadata: object) -> dict:
    """What a training checkpoint holds: the state dict and the run metadata with its record."""
    record = adapter_record(model)
    return {
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "hyper_parameters": {"metadata": {**metadata, **({"peft": record} if record else {})}},
    }


def trainable(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def assert_same_weights(a: nn.Module, b: nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        torch.testing.assert_close(sa[key], sb[key], msg=key)


# --- the record --------------------------------------------------------------------------


def test_plain_model_has_no_record() -> None:
    assert adapter_record(TrainingModule()) is None


def test_model_without_graph_model_has_no_record() -> None:
    assert adapter_record(nn.Linear(2, 2)) is None


def test_record_describes_the_live_adapters() -> None:
    model = with_lora()
    record = adapter_record(model)
    json.dumps(record)  # stored in checkpoint metadata, which is JSON
    assert record["version"] == 1
    assert record["merged"] is False
    assert record["modules"] == adapted_paths(model)
    assert record["trainable"] == []
    (adapter,) = record["adapters"].values()
    assert adapter["type"] == "lora"
    assert (adapter["config"]["r"], adapter["config"]["lora_alpha"]) == (4, 8)


def test_record_lists_trainable_parameters_outside_the_adapters() -> None:
    model = with_lora(trainable=["model.model.decoder"])
    assert adapter_record(model)["trainable"] == ["model.model.decoder.data.bias", "model.model.decoder.data.weight"]


# --- restore -----------------------------------------------------------------------------


def test_restore_then_strict_load() -> None:
    source = with_lora(trainable=["model.model.decoder"])
    checkpoint = save(source)
    target = TrainingModule()

    checkpoint = apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)
    target.load_state_dict(checkpoint["state_dict"], strict=True)

    assert adapted_paths(target) == adapted_paths(source)
    assert trainable(target) == trainable(source)
    assert_same_weights(target, source)


def test_restore_through_the_pipeline_loader(tmp_path: Path) -> None:
    source = with_lora()
    path = tmp_path / "lora.ckpt"
    torch.save(save(source), path)

    pipeline = CheckpointPipeline(stages=[LocalSource(path=str(path)), WeightsOnlyLoader()])
    result = pipeline.execute(CheckpointContext(model=TrainingModule()))

    assert adapted_paths(result.model) == adapted_paths(source)
    assert_same_weights(result.model, source)


class HookModule(BaseTrainingModule):
    """A training module built without the full anemoi setup, to call ``on_load_checkpoint``."""

    def __init__(self) -> None:
        pass

    def _step(self, batch, validation_mode: bool = False) -> Never:  # noqa: ANN001
        raise NotImplementedError


def hook_module() -> HookModule:
    module = HookModule.__new__(HookModule)
    nn.Module.__init__(module)
    module.model = Interface()
    module.task = SimpleNamespace(load_training_runtime_state_dict=lambda _state: None)
    module.config = SimpleNamespace(training=SimpleNamespace(seed=None))
    module._set_hparams({"metadata": {}})
    return module


def test_restore_via_on_load_checkpoint_before_the_strict_load() -> None:
    source = with_lora()
    checkpoint = save(source)
    module = hook_module()

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    # What Lightning does next on a resume: a strict load of the same dict.
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    assert adapted_paths(module) == adapted_paths(source)
    assert_same_weights(module, source)


def test_matching_adapters_already_present_are_left_alone() -> None:
    checkpoint = save(with_lora())
    target = with_lora()  # the run's own PEFT stage ran first, as on a resume
    before = {name: id(m) for name, m in target.named_modules()}

    apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)

    assert {name: id(m) for name, m in target.named_modules()} == before
    target.load_state_dict(checkpoint["state_dict"], strict=True)


def test_different_adapters_already_present_raise() -> None:
    checkpoint = save(with_lora())
    with pytest.raises(CheckpointIncompatibleError, match="differ"):
        apply_checkpoint_corrections(checkpoint, with_lora(r=8), NO_CONFIG)


def test_adapter_keys_without_a_record_raise() -> None:
    checkpoint = save(with_lora())
    del checkpoint["hyper_parameters"]["metadata"]["peft"]
    with pytest.raises(CheckpointIncompatibleError, match="no adapter record"):
        apply_checkpoint_corrections(checkpoint, TrainingModule(), NO_CONFIG)


def test_plain_checkpoint_into_plain_model_is_untouched() -> None:
    checkpoint = save(TrainingModule())
    target = TrainingModule()
    apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)
    assert adapted_paths(target) == []


def test_unknown_stored_option_is_dropped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    checkpoint = save(with_lora())
    (adapter,) = checkpoint["hyper_parameters"]["metadata"]["peft"]["adapters"].values()
    adapter["config"]["option_from_a_newer_peft"] = 1
    target = TrainingModule()

    with caplog.at_level(logging.WARNING):
        apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)

    assert "option_from_a_newer_peft" in caplog.text
    target.load_state_dict(checkpoint["state_dict"], strict=True)


def test_restoring_without_peft_raises_with_install_hint() -> None:
    checkpoint = save(with_lora())
    with (
        patch.dict(sys.modules, {"peft": None}),
        pytest.raises(CheckpointConfigError, match=r"anemoi-training\[peft\]"),
    ):
        apply_checkpoint_corrections(checkpoint, TrainingModule(), NO_CONFIG)


# --- the continuation chain -----------------------------------------------------------------


def test_continuing_adapters_without_a_peft_stage_survives_reload_and_resume() -> None:
    # Someone hands over L and keeps training its adapters without declaring a PEFT stage.
    checkpoint_l = save(with_lora())
    run = TrainingModule()
    run.load_state_dict(apply_checkpoint_corrections(checkpoint_l, run, NO_CONFIG)["state_dict"], strict=True)
    with torch.no_grad():
        for name, param in run.named_parameters():
            if ".lora_A." in name:
                param.add_(1.0)  # training moves the adapters on

    # Their run's own checkpoint: the record comes from the live model, not from a config.
    checkpoint_run = save(run)
    assert checkpoint_run["hyper_parameters"]["metadata"]["peft"]["modules"] == adapted_paths(run)

    reloaded = TrainingModule()
    reloaded.load_state_dict(
        apply_checkpoint_corrections(checkpoint_run, reloaded, NO_CONFIG)["state_dict"],
        strict=True,
    )
    assert_same_weights(reloaded, run)

    resumed = hook_module()
    BaseTrainingModule.on_load_checkpoint(resumed, checkpoint_run)
    resumed.load_state_dict(checkpoint_run["state_dict"], strict=True)
    assert_same_weights(resumed, run)


# --- the record in a real checkpoint ------------------------------------------------------


class CallbackModule(BoringModel):
    """A Lightning module shaped enough like anemoi's for ``AnemoiCheckpoint`` to save it."""

    def __init__(self, metadata: dict) -> None:
        super().__init__()
        self.model = Interface()
        self.model.config = {}
        self.model.metadata = metadata
        self.model.supporting_arrays = {}
        self.model.graph_data = None
        self.model.model._graph_data = None
        self.save_hyperparameters()


def test_checkpoint_callback_writes_the_record(tmp_path: Path) -> None:
    metadata = {}
    module = CallbackModule(metadata)
    # BoringModel trains its own ``layer``; the PEFT stage would freeze it otherwise.
    PeftModifierStage(
        peft_config=lora(target_modules=["model.model.decoder.data"]),
        trainable_modules=["layer"],
    ).process(
        CheckpointContext(model=module),
    )
    callback = AnemoiCheckpoint(dirpath=str(tmp_path), filename="{step}", save_last=True, monitor="step", mode="max")
    trainer = Trainer(default_root_dir=tmp_path, accelerator="cpu", callbacks=[callback], max_epochs=1, logger=False)
    trainer.fit(module)

    saved = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    record = saved["hyper_parameters"]["metadata"]["peft"]
    assert record["modules"] == ["model.model.decoder.data"]
    assert record["trainable"] == ["layer.bias", "layer.weight"]
    assert record == adapter_record(module)


# --- compatibility fixture -----------------------------------------------------------------


def test_fixture_checkpoint_still_loads() -> None:
    """A checkpoint written once with peft 0.18.1 keeps loading with the installed peft."""
    generator = torch.Generator().manual_seed(0)
    checkpoint = {
        "state_dict": {key: torch.randn(shape, generator=generator) for key, shape in STATE_DICT_SHAPES.items()},
        "hyper_parameters": {"metadata": {"peft": copy.deepcopy(RECORD)}},
    }
    target = TrainingModule()

    checkpoint = apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)
    target.load_state_dict(checkpoint["state_dict"], strict=True)

    assert adapted_paths(target) == RECORD["modules"]
    assert {name for name, p in target.named_parameters() if p.requires_grad and ".lora_" not in name} == set(
        RECORD["trainable"],
    )
