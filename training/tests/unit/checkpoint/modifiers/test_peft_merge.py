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
import subprocess
import sys
import textwrap
from pathlib import Path

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

from .test_peft import TrainingModule
from .test_peft import adapted_paths
from .test_peft import lora
from .test_peft_restore import NO_CONFIG
from .test_peft_restore import CallbackModule
from .test_peft_restore import save
from .test_peft_restore import with_lora

pytest.importorskip("peft")


def processor_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    block = model.model.model.processor[0]
    return block.mlp(block.lin_value(block.lin_query(x)))


def has_peft_reference(module: nn.Module) -> bool:
    return any(type(m).__module__.startswith("peft") or "peft_config" in m.__dict__ for m in module.modules())


# --- merge ---------------------------------------------------------------------------------


def test_merge_restores_the_plain_keys_and_drops_peft() -> None:
    model = with_lora()
    x = torch.randn(3, 8)
    expected = processor_output(model, x)

    merge_adapters(model.model.model)

    assert set(model.state_dict()) == set(TrainingModule().state_dict())
    assert adapted_paths(model) == []
    assert not has_peft_reference(model)
    torch.testing.assert_close(processor_output(model, x), expected)


def test_merged_copy_leaves_the_live_model_alone() -> None:
    live = with_lora()
    before = {k: v.clone() for k, v in live.state_dict().items()}
    x = torch.randn(3, 8)

    merged = merged_copy_on_cpu(live)

    after = live.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert adapted_paths(live)  # still adapted, still training
    assert not has_peft_reference(merged)
    live_storage = {t.data_ptr() for t in itertools.chain(live.parameters(), live.buffers())}
    assert not any(t.data_ptr() in live_storage for t in itertools.chain(merged.parameters(), merged.buffers()))
    assert all(t.device.type == "cpu" for t in merged.parameters())
    torch.testing.assert_close(processor_output(merged, x), processor_output(live, x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_merged_copy_allocates_no_extra_gpu_memory() -> None:
    live = with_lora().cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    merged_copy_on_cpu(live)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == baseline


def test_merged_inference_model_unpickles_without_peft(tmp_path: Path) -> None:
    # Only torch classes, so a process without peft can import everything the pickle names.
    graph = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    model = nn.ModuleDict({"model": nn.ModuleDict({"model": graph})})
    PeftModifierStage(peft_config=lora()).process(CheckpointContext(model=model))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_B." in name:
                param.normal_()
    x = torch.randn(2, 4)
    torch.save(x, tmp_path / "x.pt")
    torch.save(graph(x).detach(), tmp_path / "y.pt")
    torch.save(merged_copy_on_cpu(model), tmp_path / "merged.pt")
    torch.save(model, tmp_path / "unmerged.pt")

    probe = textwrap.dedent(
        """
        import sys
        import torch
        sys.modules["peft"] = None  # an inference environment without peft
        directory = sys.argv[1]
        for name in ("merged", "unmerged"):
            try:
                model = torch.load(f"{directory}/{name}.pt", weights_only=False)
            except ModuleNotFoundError:
                print(name, "needs-peft")
                continue
            y = model["model"]["model"](torch.load(f"{directory}/x.pt"))
            print(name, torch.allclose(y, torch.load(f"{directory}/y.pt"), atol=1e-5))
        """,
    )
    # A fresh interpreter is the only way to load the pickle with peft unavailable.
    result = subprocess.run(  # noqa: S603 (fixed script, sys.executable)
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split("\n")[:2] == ["merged True", "unmerged needs-peft"], result.stderr


def test_inference_checkpoint_is_merged_and_says_so(tmp_path: Path) -> None:
    module = CallbackModule({})
    PeftModifierStage(
        peft_config=lora(target_modules=["model.model.decoder.data"]),
        trainable_modules=["layer"],
    ).process(
        CheckpointContext(model=module),
    )
    callback = AnemoiCheckpoint(dirpath=str(tmp_path), filename="{step}", save_last=True, monitor="step", mode="max")
    Trainer(default_root_dir=tmp_path, accelerator="cpu", callbacks=[callback], max_epochs=1, logger=False).fit(module)

    inference = torch.load(tmp_path / "inference-last.ckpt", map_location="cpu", weights_only=False)
    assert adapted_paths(inference) == []
    assert not has_peft_reference(inference)
    assert load_metadata(tmp_path / "inference-last.ckpt")["peft"]["merged"] is True

    training = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    assert training["hyper_parameters"]["metadata"]["peft"]["merged"] is False
    assert adapted_paths(module) == ["model.model.decoder.data"]  # the run keeps its adapters


# --- stacking ------------------------------------------------------------------------------


def test_stacking_merges_the_old_adapters_then_adds_one_level() -> None:
    model = with_lora()  # L: adapters on the processor
    x = torch.randn(3, 8)
    l_output = processor_output(model, x)
    l_modules = adapted_paths(model)

    stage = PeftModifierStage(peft_config=lora(r=2, target_modules=["model.model.decoder.data"]))
    stage.process(CheckpointContext(model=model))

    assert adapted_paths(model) == ["model.model.decoder.data"]
    assert not any(".base_layer.base_layer." in key for key in model.state_dict())
    torch.testing.assert_close(processor_output(model, x), l_output)  # L's adapters live on in the base
    (entry,) = adapter_record(model)["lineage"]
    assert entry["modules"] == l_modules


def test_stacking_after_a_restore_keeps_the_recorded_lineage() -> None:
    first = with_lora()
    PeftModifierStage(peft_config=lora(target_modules=["model.model.encoder.data"])).process(
        CheckpointContext(model=first),
    )
    checkpoint = save(first)  # one level of adapters, one lineage entry
    target = TrainingModule()
    target.load_state_dict(apply_checkpoint_corrections(checkpoint, target, NO_CONFIG)["state_dict"], strict=True)
    assert len(adapter_record(target)["lineage"]) == 1

    PeftModifierStage(peft_config=lora(target_modules=["model.model.decoder.data"])).process(
        CheckpointContext(model=target),
    )

    lineage = adapter_record(target)["lineage"]
    assert [entry["modules"] for entry in lineage] == [
        adapter_record(with_lora())["modules"],
        ["model.model.encoder.data"],
    ]
