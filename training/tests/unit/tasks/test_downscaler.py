# (C) Copyright 2026 Anemoi contributors.

from types import SimpleNamespace

import torch

from anemoi.training.tasks import Downscaler


def test_downscaler_selects_named_source_and_target_datasets() -> None:
    task = Downscaler(input_datasets=["source"], output_datasets=["target"], input_offset="-6h", output_offset="0h")
    batch = {
        "source": torch.arange(2).reshape(1, 1, 2, 1),
        "target": torch.arange(2, 4).reshape(1, 1, 2, 1),
    }
    indices = {
        "source": SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(full=torch.tensor([0])))),
        "target": SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(full=torch.tensor([0])))),
    }

    inputs = task.get_inputs(batch, indices)
    targets = task.get_targets(batch)

    assert list(inputs) == ["source"]
    assert list(targets) == ["target"]
    assert inputs["source"].shape == (1, 1, 2, 1)
    assert torch.equal(targets["target"], batch["target"])

def test_downscaler_has_no_rollout_or_advance_input_contract() -> None:
    task = Downscaler(input_datasets=["source"], output_datasets=["target"])

    assert task.num_steps == 1
    assert not hasattr(task, "advance_input")
    assert task.get_input_offsets() == task.get_output_offsets()
