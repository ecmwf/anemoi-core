# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for weight averaging callback functionality."""

import logging

import omegaconf
import pytest
import yaml

from anemoi.training.diagnostics.callbacks import _get_weight_averaging_callback
from anemoi.training.diagnostics.callbacks.weight_averaging import EMAWeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import SWAWeightAveraging
from anemoi.training.diagnostics.callbacks.weight_averaging import WeightAveraging

default_config = """
training:
  weight_averaging: null
"""


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


def test_pl_callback_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Targeting the stock PL class instantiates but logs a warning about anemoi-specific hazards."""
    try:
        from pytorch_lightning.callbacks import EMAWeightAveraging as PLEMAWeightAveraging
    except ImportError:
        pytest.skip("EMAWeightAveraging not available in this PyTorch Lightning version")

    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.training.weight_averaging = {
        "_target_": "pytorch_lightning.callbacks.EMAWeightAveraging",
        "decay": 0.999,
    }
    with caplog.at_level(logging.WARNING, logger="anemoi.training.diagnostics.callbacks.weight_averaging"):
        callbacks = _get_weight_averaging_callback(config.training.weight_averaging)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], PLEMAWeightAveraging)
    assert not isinstance(callbacks[0], WeightAveraging)
    assert any("is from stock pytorch_lightning" in rec.message for rec in caplog.records)
