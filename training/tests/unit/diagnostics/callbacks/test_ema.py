# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn

from anemoi.training.diagnostics.callbacks.ema import EMA_CHECKPOINT_KEY
from anemoi.training.diagnostics.callbacks.ema import ExponentialMovingAverage
from anemoi.training.utils.checkpoint import load_and_prepare_model


def test_ema_callback_updates_and_restores_state():
    config = OmegaConf.create(
        {"training": {"ema": {"enabled": True, "decay": 0.5, "update_after_step": 0}}},
    )
    callback = ExponentialMovingAverage(config)
    model = nn.Linear(1, 1, bias=False)
    trainer = SimpleNamespace(global_step=0, model=SimpleNamespace(model=model))
    pl_module = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(load_weights_only=False)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.weight.data.fill_(2.0)
    callback.on_before_zero_grad(trainer, pl_module, optimizer)

    model.weight.data.fill_(4.0)
    trainer.global_step = 1
    callback.on_before_zero_grad(trainer, pl_module, optimizer)

    checkpoint = {}
    callback.on_save_checkpoint(trainer, pl_module, checkpoint)

    restored = ExponentialMovingAverage(config)
    restored.on_load_checkpoint(trainer, pl_module, checkpoint)

    assert torch.equal(
        restored.ema_state_dict["weight"],
        torch.full_like(model.weight, 3.0),
    )
    assert checkpoint[EMA_CHECKPOINT_KEY]["decay"] == 0.5


def test_load_and_prepare_model_uses_ema_weights(monkeypatch):
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.metadata = {"uuid": "1234"}
            self.config = {"config": True}

    model = DummyModel()
    model.linear.weight.data.fill_(1.0)
    model.linear.bias.data.fill_(0.0)
    ema_state_dict = {
        "linear.weight": torch.full_like(model.linear.weight, 5.0),
        "linear.bias": torch.full_like(model.linear.bias, 1.0),
    }
    module = SimpleNamespace(model=model)

    monkeypatch.setattr(
        "anemoi.training.utils.checkpoint.BaseGraphModule.load_from_checkpoint",
        lambda _: module,
    )
    monkeypatch.setattr(
        "anemoi.training.utils.checkpoint.torch.load",
        lambda *args, **kwargs: {EMA_CHECKPOINT_KEY: {"state_dict": ema_state_dict}},
    )

    loaded_model, metadata = load_and_prepare_model("dummy.ckpt", use_ema=True)

    assert metadata == {"uuid": "1234"}
    assert torch.equal(loaded_model.linear.weight, ema_state_dict["linear.weight"])
    assert torch.equal(loaded_model.linear.bias, ema_state_dict["linear.bias"])
    assert loaded_model.metadata is None
    assert loaded_model.config is None
