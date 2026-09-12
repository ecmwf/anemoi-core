# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch

from anemoi.training.diagnostics.callbacks.loss_weight_scheduler import LossWeightScheduler
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss


def _module() -> SimpleNamespace:
    combined = CombinedLoss(MSELoss(), MAELoss(), loss_weights=(1.0, 0.5))
    return SimpleNamespace(loss={"data": combined}, logger_enabled=False, log=lambda *_a, **_k: None)


def test_schedule_values() -> None:
    sched = LossWeightScheduler(loss_index=1, start_weight=0.0, end_weight=0.1, start_step=100, ramp_steps=200)
    assert sched.weight_at(0) == 0.0
    assert sched.weight_at(100) == 0.0
    assert sched.weight_at(200) == pytest.approx(0.05)
    assert sched.weight_at(300) == pytest.approx(0.1)
    assert sched.weight_at(10_000) == pytest.approx(0.1)
    cosine = LossWeightScheduler(loss_index=1, end_weight=0.1, start_step=100, ramp_steps=200, schedule="cosine")
    assert cosine.weight_at(200) == pytest.approx(0.05)
    assert cosine.weight_at(150) < 0.025


def test_apply_updates_only_the_selected_component_and_forward_uses_it() -> None:
    module = _module()
    sched = LossWeightScheduler(loss_index=1, start_weight=0.0, end_weight=2.0, start_step=0, ramp_steps=10)
    trainer = SimpleNamespace(global_step=5)
    sched.on_train_batch_start(trainer, module, None, 0)
    assert module.loss["data"].loss_weights == (1.0, 1.0)

    pred = torch.zeros(1, 1, 1, 3, 2)
    target = torch.ones(1, 1, 1, 3, 2)
    for loss in module.loss["data"].losses:
        loss.add_scaler(3, torch.full((3,), 1 / 3), name="node_weights")
    value = module.loss["data"](pred, target)
    torch.testing.assert_close(value, torch.tensor(2.0))  # mse 1 * 1 + mae 1 * 1

    sched.on_train_batch_start(SimpleNamespace(global_step=50), module, None, 0)
    assert module.loss["data"].loss_weights == (1.0, 2.0)


def test_rejects_non_combined_loss_and_bad_index() -> None:
    sched = LossWeightScheduler(loss_index=1)
    with pytest.raises(TypeError):
        sched.apply(SimpleNamespace(loss={"data": MSELoss()}), 0)
    with pytest.raises(IndexError):
        LossWeightScheduler(loss_index=5).apply(_module(), 0)
    with pytest.raises(ValueError, match="schedule"):
        LossWeightScheduler(loss_index=0, schedule="step")
