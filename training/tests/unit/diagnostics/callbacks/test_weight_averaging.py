# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for weight averaging callback functionality."""

from pathlib import Path

import omegaconf
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel

from anemoi.training.diagnostics.callbacks import _get_weight_averaging_callback
from anemoi.training.diagnostics.callbacks.weight_averaging import EMAWeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import SWAWeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import WeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import averaged_weights

default_config = """
training:
  weight_averaging: null
"""


class _ModelWithIntegerBuffer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("indices", torch.tensor([0], dtype=torch.long))


def test_weight_averaging_disabled_when_null() -> None:
    """No callback is returned when weight_averaging is null."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    callbacks = _get_weight_averaging_callback(config.training.weight_averaging)
    assert callbacks == []


def test_ema_callback_instantiates() -> None:
    """Anemoi EMA callback is instantiated from a hydra-style config."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.training.weight_averaging = {
        "_target_": "anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging",
        "decay": 0.999,
    }
    callbacks = _get_weight_averaging_callback(config.training.weight_averaging)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EMAWeightAveraging)
    assert isinstance(callbacks[0], WeightAveraging)


def test_swa_callback_instantiates() -> None:
    """Anemoi SWA callback is instantiated from a hydra-style config."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.training.weight_averaging = {
        "_target_": "anemoi.training.diagnostics.callbacks.weight_averaging.SWAWeightAveraging",
    }
    callbacks = _get_weight_averaging_callback(config.training.weight_averaging)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], SWAWeightAveraging)
    assert isinstance(callbacks[0], WeightAveraging)


def test_weight_averaging_syncs_fixed_buffers_without_averaging_them() -> None:
    model = _ModelWithIntegerBuffer()
    callback = EMAWeightAveraging()
    callback.setup(None, model, "fit")
    assert callback._average_model is not None

    callback._average_model.update_parameters(model)
    model.weight.data.fill_(2.0)
    model.indices.fill_(1)
    callback._average_model.update_parameters(model)

    assert callback._average_model.module.indices.item() == 1


def test_should_update_respects_step_schedule() -> None:
    """update_every_n_steps and update_starting_at_step gate step updates."""
    callback = EMAWeightAveraging(update_every_n_steps=5, update_starting_at_step=100)

    assert not callback.should_update(step_idx=0)
    assert not callback.should_update(step_idx=95)  # right frequency, before the start step
    assert not callback.should_update(step_idx=101)  # after the start step, wrong frequency
    assert callback.should_update(step_idx=100)
    assert callback.should_update(step_idx=105)


def test_should_update_defaults_to_every_step() -> None:
    """With no schedule configured, every step updates and epoch ends do not."""
    callback = EMAWeightAveraging()

    assert callback.should_update(step_idx=0)
    assert callback.should_update(step_idx=7)
    assert not callback.should_update(epoch_idx=0)


class _Trainer:
    def __init__(self, module: pl.LightningModule, callbacks: list) -> None:
        self.lightning_module = module
        self.callbacks = callbacks


def test_averaged_weights_swaps_and_restores() -> None:
    """The context manager exposes the averaged weights and puts the live ones back."""
    model = _ModelWithIntegerBuffer()
    callback = EMAWeightAveraging(decay=0.5)
    callback.setup(None, model, "fit")

    callback._average_model.update_parameters(model)  # average == 1.0
    model.weight.data.fill_(3.0)
    trainer = _Trainer(model, [callback])

    with averaged_weights(trainer):
        assert model.weight.item() == 1.0
    assert model.weight.item() == 3.0


def _ema_trainer(tmp_path: Path, max_steps: int, callbacks: list) -> pl.Trainer:
    return pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        callbacks=callbacks,
        max_steps=max_steps,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_ema_state_is_restored_on_resume(tmp_path: Path) -> None:
    """A run resumed from its own checkpoint continues the EMA instead of restarting it."""
    steps = 4
    checkpoint = ModelCheckpoint(dirpath=str(tmp_path), save_last=True, every_n_train_steps=steps)
    _ema_trainer(tmp_path, steps, [EMAWeightAveraging(decay=0.9), checkpoint]).fit(BoringModel())

    saved = torch.load(tmp_path / "last.ckpt", weights_only=False)
    # The averaged weights are saved as "state_dict", the training weights beside them.
    assert "current_model_state" in saved
    assert saved["averaging_state"]["n_averaged"].item() == steps
    assert saved["callbacks"]["EMAWeightAveraging"]["latest_update_step"] == steps
    assert not torch.equal(saved["state_dict"]["layer.weight"], saved["current_model_state"]["layer.weight"])

    # Resuming at the step the checkpoint stopped at restores state, then exits immediately.
    resumed_callback = EMAWeightAveraging(decay=0.9)
    resumed_model = BoringModel()
    _ema_trainer(tmp_path, steps, [resumed_callback]).fit(resumed_model, ckpt_path=str(tmp_path / "last.ckpt"))

    assert resumed_callback._latest_update_step == steps, "EMA schedule restarted instead of resuming"
    assert resumed_callback._average_model.n_averaged.item() == steps
    assert torch.equal(resumed_callback._average_model.module.layer.weight, saved["state_dict"]["layer.weight"])
    # Training continues from the raw weights, not from the averaged ones.
    assert torch.equal(resumed_model.layer.weight, saved["current_model_state"]["layer.weight"])
