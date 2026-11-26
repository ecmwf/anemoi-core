# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training cycles for the disentangled forecaster, with and without latent rollout."""

import copy
from pathlib import Path

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline

DISENTANGLED = "anemoi.models.models.AnemoiModelDisentangledEncProcDec"


def _as_disentangled_forecaster(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        cfg.model.model._target_ = DISENTANGLED
        cfg.task.rollout = OmegaConf.create({"start": 2, "epoch_increment": 0, "maximum": 2})
    return cfg


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_disentangled(
    global_config: tuple[DictConfig, str, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url, _ = global_config
    get_test_archive(url)
    AnemoiTrainer(_as_disentangled_forecaster(cfg)).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_disentangled_latent_rollout(
    global_config: tuple[DictConfig, str, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url, _ = global_config
    get_test_archive(url)
    cfg = _as_disentangled_forecaster(cfg)
    with open_dict(cfg):
        cfg.training.method = OmegaConf.create({"_target_": "anemoi.training.train.methods.LatentRolloutTraining"})
        cfg.task.rollout = OmegaConf.create({"start": 3, "epoch_increment": 0, "maximum": 3})
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_disentangled_from_frozen_autoencoder(
    gnn_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    """Train an autoencoder, then a disentangled forecaster on its frozen encoder/decoder."""
    cfg, url = gnn_config
    get_test_archive(url)

    # 1) autoencoder: Autoencoder task, one input step
    ae_cfg = copy.deepcopy(cfg)
    with open_dict(ae_cfg):
        ae_cfg.task = OmegaConf.create({"_target_": "anemoi.training.tasks.Autoencoder"})
        ae_cfg.system.output.root = str(Path(cfg.system.output.root) / "autoencoder")
    AnemoiTrainer(ae_cfg).train()
    checkpoints = sorted(Path(ae_cfg.system.output.root).rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    assert checkpoints, "autoencoder run wrote no checkpoint"

    # 2) disentangled forecaster from the autoencoder weights, encoder/decoder frozen
    fc_cfg = _as_disentangled_forecaster(copy.deepcopy(cfg))
    frozen = ["encoder", "decoder"]
    with open_dict(fc_cfg):
        fc_cfg.system.output.root = str(Path(cfg.system.output.root) / "forecaster")
        fc_cfg.training.checkpoint = OmegaConf.create(
            {
                "source": {
                    "_target_": "anemoi.training.checkpoint.sources.local.LocalSource",
                    "path": str(checkpoints[-1]),
                },
                "loading": {
                    "_target_": "anemoi.training.checkpoint.loading.strategies.TransferLearningLoader",
                    "skip_mismatched": True,
                },
                "modifiers": [
                    {
                        "_target_": "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage",
                        "submodules_to_freeze": [f"model.model.{name}" for name in frozen],
                    },
                ],
            },
        )
    trainer = AnemoiTrainer(fc_cfg)
    trainer.train()

    graph_model = trainer.model.model.model
    for name in frozen:
        assert not any(p.requires_grad for p in getattr(graph_model, name).parameters()), f"{name} not frozen"
    assert all(p.requires_grad for p in graph_model.latent_blender.parameters())
    assert all(p.requires_grad for p in graph_model.processor.parameters())
