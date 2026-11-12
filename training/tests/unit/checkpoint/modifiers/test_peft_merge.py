# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Merging PEFT adapters: the inference checkpoint, and stacking adapters on adapters."""

import itertools
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Trainer
from torch import nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections
from anemoi.training.checkpoint.modifiers.peft import PeftModifierStage
from anemoi.training.checkpoint.modifiers.peft import adapter_record
from anemoi.training.checkpoint.modifiers.peft import merge_adapters
from anemoi.training.checkpoint.modifiers.peft import merged_copy_on_cpu
from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.utils.checkpoints import load_metadata

from .test_peft import METHODS
from .test_peft import TrainingModule
from .test_peft import adapted
from .test_peft import lora
from .test_peft_restore import NO_CONFIG
from .test_peft_restore import callback_module
from .test_peft_restore import save
from .test_peft_restore import with_adapters

pytest.importorskip("peft")


def output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``x`` through the first processor block and the decoder of the graph model."""
    graph_model = model.model.model
    block = graph_model.processor[0]
    return graph_model.decoder["data"](block.mlp(block.lin_value(block.lin_query(x))))


# Merged weights compute (W + BA) x where the live adapters compute W x + B (A x): equal up to
# float32 rounding, which the default tolerance of assert_close is too tight for. The rounding
# is relative to the output, so the tolerance is too; a wrong merge is off by far more.
MERGE_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}


def inputs() -> torch.Tensor:
    return torch.randn(3, 8, generator=torch.Generator().manual_seed(11))


def has_peft_reference(module: nn.Module) -> bool:
    return any(type(m).__module__.startswith("peft") or "peft_config" in m.__dict__ for m in module.modules())


def without_peft() -> dict:
    """``sys.modules`` entries that make every ``peft`` import fail, as in an environment without it."""
    return {name: None for name in list(sys.modules) if name == "peft" or name.startswith("peft.")}


# --- merge ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(METHODS))
def test_merge_restores_the_plain_keys_and_drops_peft(method: str) -> None:
    model = with_adapters(method=method, modules_to_save=["decoder.data"])
    x = inputs()
    expected = output(model, x)

    merge_adapters(model.model.model)

    assert set(model.state_dict()) == set(TrainingModule().state_dict())
    assert adapted(model) == []
    assert not has_peft_reference(model)
    torch.testing.assert_close(output(model, x), expected, **MERGE_TOLERANCE)


def test_merged_copy_leaves_the_live_model_alone() -> None:
    live = with_adapters(modules_to_save=["decoder.data"])
    before = {k: v.clone() for k, v in live.state_dict().items()}
    x = inputs()

    merged = merged_copy_on_cpu(live)

    after = live.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert adapted(live)  # still adapted, still training
    assert not has_peft_reference(merged)
    live_storage = {t.data_ptr() for t in itertools.chain(live.parameters(), live.buffers())}
    assert not any(t.data_ptr() in live_storage for t in itertools.chain(merged.parameters(), merged.buffers()))
    assert all(t.device.type == "cpu" for t in merged.parameters())
    torch.testing.assert_close(output(merged, x), output(live, x), **MERGE_TOLERANCE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_merged_copy_allocates_no_extra_gpu_memory() -> None:
    live = with_adapters().cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    merged_copy_on_cpu(live)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == baseline


def test_merged_model_unpickles_without_peft(tmp_path: Path) -> None:
    model = with_adapters(modules_to_save=["decoder.data"])
    x = inputs()
    expected = output(model, x)
    torch.save(merged_copy_on_cpu(model), tmp_path / "merged.pt")
    torch.save(model, tmp_path / "unmerged.pt")

    with patch.dict(sys.modules, without_peft()):
        merged = torch.load(tmp_path / "merged.pt", weights_only=False)
        with pytest.raises(ModuleNotFoundError, match="peft"):
            torch.load(tmp_path / "unmerged.pt", weights_only=False)

    torch.testing.assert_close(output(merged, x), expected, **MERGE_TOLERANCE)


def test_inference_checkpoint_is_merged_and_says_so(tmp_path: Path) -> None:
    module = callback_module()
    callback = AnemoiCheckpoint(dirpath=str(tmp_path), filename="{step}", save_last=True, monitor="step", mode="max")
    Trainer(default_root_dir=tmp_path, accelerator="cpu", callbacks=[callback], max_epochs=1, logger=False).fit(module)

    inference = torch.load(tmp_path / "inference-last.ckpt", map_location="cpu", weights_only=False)
    assert not has_peft_reference(inference)
    assert load_metadata(tmp_path / "inference-last.ckpt")["peft"]["merged"] is True

    training = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    assert training["hyper_parameters"]["metadata"]["peft"]["merged"] is False
    assert adapted(module) == ["decoder.data"]  # the run keeps its adapters


def test_offline_inference_conversion_merges_the_adapters(tmp_path: Path) -> None:
    """``anemoi-training checkpoint inference`` writes merged weights too, not only the callback."""
    from anemoi.training.utils.checkpoint import save_inference_checkpoint

    module = with_adapters(method="miss", modules_to_save=["decoder.data"])
    x = inputs()
    expected = output(module, x)
    # What load_and_prepare_model returns: the model interface and the run metadata.
    metadata = {"peft": adapter_record(module)}

    path = save_inference_checkpoint(module.model, metadata, tmp_path / "inference.ckpt")

    inference = torch.load(path, map_location="cpu", weights_only=False)
    assert not has_peft_reference(inference)
    assert load_metadata(path)["peft"]["merged"] is True
    block = inference.model.processor[0]
    merged_output = inference.model.decoder["data"](block.mlp(block.lin_value(block.lin_query(x))))
    torch.testing.assert_close(merged_output, expected, **MERGE_TOLERANCE)
    assert adapted(module)  # the model passed in keeps its adapters


# --- stacking ------------------------------------------------------------------------------


def test_stacking_merges_the_old_adapters_then_adds_one_level() -> None:
    model = with_adapters()  # L: LoRA on the processor
    x = inputs()
    l_output = output(model, x)
    l_adapters = adapter_record(model)["adapters"]

    PeftModifierStage(peft_config=METHODS["miss"](target_modules=["lin_query"])).process(
        CheckpointContext(model=model),
    )

    assert adapted(model) == ["processor.0.lin_query", "processor.1.lin_query"]
    assert not any(".base_layer.base_layer." in key for key in model.state_dict())
    torch.testing.assert_close(output(model, x), l_output, **MERGE_TOLERANCE)  # L lives on in the base
    assert adapter_record(model)["lineage"] == [l_adapters]


def test_stacking_after_a_restore_keeps_the_recorded_lineage() -> None:
    first = with_adapters()
    first_adapters = adapter_record(first)["adapters"]
    PeftModifierStage(peft_config=lora(target_modules=["encoder.data"])).process(CheckpointContext(model=first))
    second_adapters = adapter_record(first)["adapters"]
    checkpoint = save(first)  # one level of adapters, one lineage entry
    target = TrainingModule()
    target.load_state_dict(apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)["state_dict"], strict=True)
    assert adapter_record(target)["lineage"] == [first_adapters]

    PeftModifierStage(peft_config=lora(target_modules=["decoder.data"])).process(CheckpointContext(model=target))

    assert adapter_record(target)["lineage"] == [first_adapters, second_adapters]
