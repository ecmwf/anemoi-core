# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""LoRA adapters through real training runs: fine-tune, resume, reload, continue, stack."""

import os
from copy import deepcopy
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import open_dict

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline

pytest.importorskip("peft")

PEFT_STAGE = "anemoi.training.checkpoint.modifiers.peft.PeftModifierStage"
LOCAL_SOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"
RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
WEIGHTS_ONLY = "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"


def peft_stage(target_modules: str = r"model\.model\.processor\..*") -> dict:
    return {"_target_": PEFT_STAGE, "peft_config": {"r": 4, "lora_alpha": 8, "target_modules": target_modules}}


class _Record(pl.Callback):
    """Record the step a run starts at and its training losses."""

    def __init__(self) -> None:
        self.start_step: int | None = None
        self.losses: list[float] = []

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self.start_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        del trainer, pl_module, batch, batch_idx
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(float(loss.detach().cpu()))


def _train(cfg: DictConfig, checkpoint: dict, max_epochs: int = 1) -> tuple[AnemoiTrainer, _Record]:
    cfg = deepcopy(cfg)
    with open_dict(cfg):
        cfg.training.checkpoint = checkpoint
        cfg.training.max_epochs = max_epochs
    trainer = AnemoiTrainer(cfg)
    record = _Record()
    trainer.callbacks.append(record)
    trainer.train()
    return trainer, record


def _run_dir(cfg: DictConfig, trainer: AnemoiTrainer) -> Path:
    return Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root) / trainer.run_id


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _prepare(cfg: DictConfig) -> tuple[DictConfig, Path, dict]:
    """Short runs, and the staged pretrained checkpoint the fixture wrote."""
    pretrained_path = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root) / "dummy_id" / "last.ckpt"
    pretrained = _load(pretrained_path)["state_dict"]
    with open_dict(cfg):
        cfg.training.seed = 42
        cfg.dataloader.limit_batches.training = 4
        cfg.dataloader.limit_batches.validation = 1
        # Model sharding supports batch size 1 only; use it everywhere so losses compare.
        cfg.dataloader.batch_size.training = 1
        cfg.dataloader.batch_size.validation = 1
        cfg.diagnostics.plot.callbacks = []
    return cfg, pretrained_path, pretrained


def _base_changes(state_dict: dict, pretrained: dict) -> list[str]:
    """Graph-model weights that differ from the pretrained checkpoint (adapters aside)."""
    changed = []
    for key, value in state_dict.items():
        original = key.replace(".base_layer.", ".")
        in_base = key.startswith("model.model.") and ".lora_" not in key and original in pretrained
        if in_base and not torch.equal(value, pretrained[original]):
            changed.append(key)
    return changed


@skip_if_offline
@pytest.mark.slow
def test_lora_fine_tune_resume_reload_continue_and_stack(
    global_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = global_config_with_checkpoint
    get_test_archive(url)
    cfg, pretrained_path, pretrained = _prepare(cfg)

    # Fine-tune: pretrained weights, LoRA on the processor.
    fine_tune, _ = _train(
        cfg,
        {
            "source": {"_target_": LOCAL_SOURCE, "path": str(pretrained_path)},
            "loading": {"_target_": WEIGHTS_ONLY},
            "modifiers": [peft_stage()],
        },
    )
    run_dir = _run_dir(cfg, fine_tune)
    l_ckpt = _load(run_dir / "last.ckpt")
    record = l_ckpt["hyper_parameters"]["metadata"]["peft"]
    lora_keys = [key for key in l_ckpt["state_dict"] if ".lora_" in key]
    assert record["modules"] and lora_keys
    assert _base_changes(l_ckpt["state_dict"], pretrained) == []
    assert any(l_ckpt["state_dict"][key].abs().sum() > 0 for key in lora_keys if ".lora_B." in key)

    # The inference checkpoint holds merged weights and says so.
    inference_path = run_dir / "inference-last.ckpt"
    inference = _load(inference_path)
    assert not any(type(m).__module__.startswith("peft") for m in inference.modules())
    assert load_metadata(inference_path)["peft"]["merged"] is True

    # Resume the run through Lightning: the run's own stage rebuilds the adapters.
    _, resumed = _train(
        cfg,
        {"source": {"_target_": RUN_SOURCE, "run_id": run_dir.name, "fork": False}, "modifiers": [peft_stage()]},
        max_epochs=2,
    )
    assert resumed.start_step == l_ckpt["global_step"]
    assert _base_changes(_load(run_dir / "last.ckpt")["state_dict"], pretrained) == []

    # Someone else loads L with no PEFT config and keeps training its adapters.
    l_path = run_dir / "last.ckpt"
    continued, _ = _train(
        cfg,
        {"source": {"_target_": LOCAL_SOURCE, "path": str(l_path)}, "loading": {"_target_": WEIGHTS_ONLY}},
    )
    continued_ckpt = _load(_run_dir(cfg, continued) / "last.ckpt")
    assert continued_ckpt["hyper_parameters"]["metadata"]["peft"]["modules"] == record["modules"]

    # And stacks new adapters on top: L's are merged, one adapter level remains.
    stacked, _ = _train(
        cfg,
        {
            "source": {"_target_": LOCAL_SOURCE, "path": str(l_path)},
            "loading": {"_target_": WEIGHTS_ONLY},
            "modifiers": [peft_stage(target_modules="all-linear")],
        },
    )
    stacked_record = _load(_run_dir(cfg, stacked) / "last.ckpt")["hyper_parameters"]["metadata"]["peft"]
    assert [entry["modules"] for entry in stacked_record["lineage"]] == [record["modules"]]


@skip_if_offline
@pytest.mark.slow
@pytest.mark.multigpu
def test_lora_with_model_sharding_matches_one_gpu(
    global_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    """LoRA fine-tuning with the model sharded across two GPUs gives the one-GPU losses.

    Launch with ``srun --ntasks=2 python -m pytest ... --multigpu --slow`` on a 2-GPU
    allocation. Rank 0 first runs the one-GPU reference on its own (rank 1 waits for it
    when the process group forms), then both ranks train the same fine-tune with the
    model sharded across them.
    """
    cfg, url = global_config_with_checkpoint
    get_test_archive(url)
    cfg, pretrained_path, pretrained = _prepare(cfg)
    checkpoint = {
        "source": {"_target_": LOCAL_SOURCE, "path": str(pretrained_path)},
        "loading": {"_target_": WEIGHTS_ONLY},
        "modifiers": [peft_stage()],
    }
    rank = int(os.environ.get("SLURM_PROCID", "0"))

    single = _train(cfg, checkpoint)[1] if rank == 0 else None

    sharded_cfg = deepcopy(cfg)
    with open_dict(sharded_cfg):
        sharded_cfg.system.hardware.accelerator = "cuda"
        sharded_cfg.system.hardware.num_gpus_per_node = 2
        sharded_cfg.system.hardware.num_gpus_per_model = 2
        sharded_cfg.dataloader.read_group_size = 2
        # testing_modifications.yaml sets SingleDeviceStrategy: without this override each
        # rank would train alone and nothing would be sharded.
        sharded_cfg.training.strategy = {
            "_target_": "anemoi.training.distributed.strategy.DDPGroupStrategy",
            "num_gpus_per_model": 2,
            "read_group_size": 2,
        }
        # pytest's tmp_path differs per rank; every rank must agree on the output root.
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        scratch = Path(os.environ.get("SCRATCH", cfg.system.output.root))
        sharded_cfg.system.output.root = str(scratch / f"peft_sharded_{job_id}")
    trainer, sharded = _train(sharded_cfg, checkpoint)

    assert trainer.model.trainer.world_size == 2
    assert trainer.model.model_comm_group_size == 2
    if rank == 0:
        # Only rank 0 writes checkpoints, into the directory of its own run id.
        torch.testing.assert_close(torch.tensor(sharded.losses), torch.tensor(single.losses), rtol=1e-5, atol=0)
        sharded_ckpt = _load(_run_dir(sharded_cfg, trainer) / "last.ckpt")
        assert _base_changes(sharded_ckpt["state_dict"], pretrained) == []
