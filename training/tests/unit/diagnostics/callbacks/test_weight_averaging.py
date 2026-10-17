# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for weight averaging callback functionality."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import omegaconf
import pytest
import pytorch_lightning as pl
import torch
import yaml
from packaging.version import Version

import anemoi.training.diagnostics.callbacks as callbacks_module
from anemoi.training.diagnostics.callbacks import _get_weight_averaging_callback

if TYPE_CHECKING:
    from types import ModuleType

default_config = """
training:
  weight_averaging: null
"""


@pytest.fixture
def weight_averaging_module() -> ModuleType:
    """Load the optional callback classes only on supported Lightning versions."""
    if Version(pl.__version__) < Version("2.6.0"):
        pytest.skip("Weight averaging requires PyTorch Lightning >= 2.6.0")
    return importlib.import_module("anemoi.training.diagnostics.callbacks.weight_averaging")


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


def test_callback_package_does_not_import_weight_averaging(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable optional implementation must not prevent ordinary callback imports."""
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "anemoi.training.diagnostics.callbacks.weight_averaging", None)
        importlib.reload(callbacks_module)
        assert callbacks_module._get_weight_averaging_callback(None) == []


@pytest.mark.parametrize("callback_name", ["EMAWeightAveraging", "SWAWeightAveraging"])
def test_weight_averaging_requires_supported_lightning(
    monkeypatch: pytest.MonkeyPatch,
    callback_name: str,
) -> None:
    """Reject an enabled callback before Hydra imports it on older Lightning versions."""
    monkeypatch.setattr(pl, "__version__", "2.5.4")
    config = omegaconf.OmegaConf.create(
        {"_target_": f"anemoi.training.diagnostics.callbacks.weight_averaging.{callback_name}"},
    )
    with pytest.raises(RuntimeError, match=r"requires pytorch_lightning>=2\.6\.0, but found 2\.5\.4"):
        _get_weight_averaging_callback(config)


def test_ema_callback_instantiates(weight_averaging_module: ModuleType) -> None:
    """Anemoi EMA callback is instantiated from a hydra-style config."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.training.weight_averaging = {
        "_target_": "anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging",
        "decay": 0.999,
    }
    callbacks = _get_weight_averaging_callback(config.training.weight_averaging)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], weight_averaging_module.EMAWeightAveraging)
    assert isinstance(callbacks[0], weight_averaging_module.WeightAveraging)


def test_swa_callback_instantiates(weight_averaging_module: ModuleType) -> None:
    """Anemoi SWA callback is instantiated from a hydra-style config."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.training.weight_averaging = {
        "_target_": "anemoi.training.diagnostics.callbacks.weight_averaging.SWAWeightAveraging",
    }
    callbacks = _get_weight_averaging_callback(config.training.weight_averaging)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], weight_averaging_module.SWAWeightAveraging)
    assert isinstance(callbacks[0], weight_averaging_module.WeightAveraging)


def test_weight_averaging_syncs_fixed_buffers_without_averaging_them(weight_averaging_module: ModuleType) -> None:
    model = _ModelWithIntegerBuffer()
    callback = weight_averaging_module.EMAWeightAveraging()
    callback.setup(None, model, "fit")
    assert callback._average_model is not None

    callback._average_model.update_parameters(model)
    model.weight.data.fill_(2.0)
    model.indices.fill_(1)
    callback._average_model.update_parameters(model)

    assert callback._average_model.module.indices.item() == 1
