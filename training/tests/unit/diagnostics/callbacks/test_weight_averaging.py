# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for weight averaging callback functionality."""

import omegaconf
import pytorch_lightning as pl
import torch
import yaml

from anemoi.training.diagnostics.callbacks import _get_weight_averaging_callback
from anemoi.training.diagnostics.callbacks.weight_averaging import EMAWeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import SWAWeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import WeightAveraging

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
