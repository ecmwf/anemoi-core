# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import torch

from anemoi.training.diagnostics.callbacks.refractivity import RefractivityLevelLogger
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses.refractivity import RefractivityOperatorLoss


def _fake_refrac_loss() -> RefractivityOperatorLoss:
    loss = RefractivityOperatorLoss.__new__(RefractivityOperatorLoss)
    torch.nn.Module.__init__(loss)
    loss.observation_variables = ["refrac_10400", "refrac_13000"]
    loss.last_level_losses = torch.tensor([1.5, 0.5])
    loss.last_level_counts = torch.tensor([120, 80])
    loss.last_unbracketed_fraction = torch.tensor(0.05)
    return loss


def test_logger_reads_every_refractivity_leaf() -> None:
    refrac = _fake_refrac_loss()
    combined = CombinedLoss(MSELoss(), refrac, loss_weights=(1.0, 1.0))
    logged: dict[str, float] = {}
    module = SimpleNamespace(
        loss={"data": combined},
        logger_enabled=True,
        log=lambda name, value, **_kw: logged.__setitem__(name, value),
    )
    callback = RefractivityLevelLogger(every_n_batches=10)

    callback.on_train_batch_end(None, module, None, None, batch_idx=5)
    assert logged == {}
    callback.on_train_batch_end(None, module, None, None, batch_idx=10)
    assert logged["train_refrac/data/refrac_10400"] == 1.5
    assert logged["train_refrac_count/data/refrac_13000"] == 80.0
    assert abs(logged["train_refrac_unbracketed_fraction/data"] - 0.05) < 1e-6

    logged.clear()
    callback.on_validation_batch_end(None, module, None, None, batch_idx=0)
    assert logged["val_refrac/data/refrac_13000"] == 0.5


def test_logger_ignores_modules_without_refractivity_loss() -> None:
    logged: dict[str, float] = {}
    module = SimpleNamespace(loss={"data": MSELoss()}, log=lambda name, value, **_kw: logged.__setitem__(name, value))
    RefractivityLevelLogger(every_n_batches=1).on_train_batch_end(None, module, None, None, batch_idx=0)
    assert logged == {}
