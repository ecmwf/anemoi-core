# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for weight averaging callback functionality."""

import omegaconf
import pytest
import yaml

from anemoi.training.diagnostics.callbacks import _get_weight_averaging_callback

default_config = """
training:
  weight_averaging: null
"""


def test_weight_averaging_disabled_when_null() -> None:
    """Test that weight averaging is disabled when set to null."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    callbacks = _get_weight_averaging_callback(config)
    assert callbacks == []


def test_ema_callback_available() -> None:
    """Test that EMA weight averaging callback can be instantiated."""
    pytest.importorskip("pytorch_lightning.callbacks", reason="EMA requires PyTorch Lightning 2.6+")

    try:
        from pytorch_lightning.callbacks import EMAWeightAveraging
    except ImportError:
        pytest.skip("EMAWeightAveraging not available in this PyTorch Lightning version")

    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.training.weight_averaging = {
        "_target_": "pytorch_lightning.callbacks.EMAWeightAveraging",
        "decay": 0.999,
    }
    callbacks = _get_weight_averaging_callback(config)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EMAWeightAveraging)
