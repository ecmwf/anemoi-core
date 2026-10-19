# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Edge-case tests for the shared checkpoint-load helpers in ``loading.base``.

The tests cover guard and error branches, and the call order inside two loaders.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading import strategies
from anemoi.training.checkpoint.loading.base import apply_checkpoint_format_migrations
from anemoi.training.checkpoint.loading.base import apply_trainable_edge_perm_migration
from anemoi.training.checkpoint.loading.base import extract_checkpoint_variables_metadata
from anemoi.training.checkpoint.loading.base import refresh_checkpoint_processors
from anemoi.training.checkpoint.loading.base import warn_on_hparams_divergence
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

class _MiniModel(nn.Module):
    """Single-layer model whose state dict round-trips through a strict load."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)


def _install_fake_migration_module(module_name: str, migrate: object) -> types.ModuleType:
    """Make a fake module with a ``migrate`` attribute for ``sys.modules``."""
    module = types.ModuleType(module_name)
    module.migrate = migrate
    return module


def _hparams_warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the records that report an hparams divergence."""
    return [record for record in caplog.records if "hparams differ" in record.getMessage()]


# --- apply_checkpoint_format_migrations --------------------------


def test_format_migration_swallows_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``AttributeError`` from the migration is a no-op."""

    def _migrate(_checkpoint: dict) -> dict:
        msg = "config"
        raise AttributeError(msg)

    module_name = loading_base._CHUNKING_FIX_PATHS[0]
    monkeypatch.setitem(sys.modules, module_name, _install_fake_migration_module(module_name, _migrate))

    checkpoint = {"state_dict": {}}
    assert apply_checkpoint_format_migrations(checkpoint) is checkpoint


# --- apply_trainable_edge_perm_migration -------


def test_edge_perm_migration_none_checkpoint_returns_none() -> None:
    """A ``None`` checkpoint returns ``None``."""
    model = _MiniModel()
    assert apply_trainable_edge_perm_migration(None, model) is None


def test_edge_perm_migration_noop_when_module_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no edge-perm module imports, the checkpoint is returned unchanged."""
    for module_name in loading_base._TRAINABLE_EDGE_PERM_PATHS:
        monkeypatch.setitem(sys.modules, module_name, None)

    model = _MiniModel()
    checkpoint = {"state_dict": {}}
    assert apply_trainable_edge_perm_migration(checkpoint, model) is checkpoint


def test_edge_perm_migration_swallows_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``KeyError`` from the edge-perm migration is a no-op."""

    def _migrate(_checkpoint: dict, _model: nn.Module) -> dict:
        msg = "hyper_parameters"
        raise KeyError(msg)

    module_name = loading_base._TRAINABLE_EDGE_PERM_PATHS[0]
    monkeypatch.setitem(sys.modules, module_name, _install_fake_migration_module(module_name, _migrate))

    model = _MiniModel()
    checkpoint = {"state_dict": {}}
    assert apply_trainable_edge_perm_migration(checkpoint, model) is checkpoint


# --- refresh_checkpoint_processors ----------------------


def test_refresh_processors_none_model_keeps_the_entries() -> None:
    """With ``model=None``, the processor entries stay in the state dict.

    A drop without re-injection causes missing keys in a strict load.
    """
    checkpoint = {
        "state_dict": {
            "model.pre_processors.w": torch.zeros(2),
            "model.post_processors.b": torch.zeros(1),
            "model.encoder.x": torch.ones(3),
        },
    }

    refresh_checkpoint_processors(checkpoint, None, update_states=True, update_tendencies=False)

    state_dict = checkpoint["state_dict"]
    assert torch.equal(state_dict["model.pre_processors.w"], torch.zeros(2))
    assert torch.equal(state_dict["model.post_processors.b"], torch.zeros(1))
    assert torch.equal(state_dict["model.encoder.x"], torch.ones(3))


def test_refresh_processors_none_checkpoint_is_noop() -> None:
    """A ``None`` checkpoint is a no-op."""
    model = _MiniModel()
    assert refresh_checkpoint_processors(None, model, update_states=True, update_tendencies=False) is None


# --- warn_on_hparams_divergence -------


def test_warn_noop_when_checkpoint_none(caplog: pytest.LogCaptureFixture) -> None:
    """A ``None`` checkpoint gives no warning."""
    run_config = OmegaConf.create({"model": {"layers": 1}})
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence(None, run_config)
    assert _hparams_warning_records(caplog) == []


def test_warn_noop_when_hyper_parameters_not_mapping(caplog: pytest.LogCaptureFixture) -> None:
    """A non-dict ``hyper_parameters`` gives no warning."""
    run_config = OmegaConf.create({"model": {"layers": 1}})
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence({"hyper_parameters": [1, 2, 3]}, run_config)
    assert _hparams_warning_records(caplog) == []


def test_warn_noop_when_checkpoint_config_none(caplog: pytest.LogCaptureFixture) -> None:
    """A ``None`` ``hyper_parameters.config`` gives no warning."""
    run_config = OmegaConf.create({"model": {"layers": 1}})
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence({"hyper_parameters": {"config": None}}, run_config)
    assert _hparams_warning_records(caplog) == []


def test_warn_skips_when_to_container_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An error in ``OmegaConf.to_container`` gives no warning."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        msg = "cannot resolve"
        raise ValueError(msg)

    monkeypatch.setattr("omegaconf.OmegaConf.to_container", _raise)

    checkpoint = {"hyper_parameters": {"config": {"model": {"layers": 1}}}}
    run_config = OmegaConf.create({"model": {"layers": 1}})
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence(checkpoint, run_config)
    assert _hparams_warning_records(caplog) == []


# --- extract_checkpoint_variables_metadata -----------------------


def test_extract_variables_metadata_none_checkpoint_is_a_safe_noop() -> None:
    """A ``None`` checkpoint does not set ``_ckpt_variables_metadata``."""
    model = SimpleNamespace(_ckpt_model_name_to_index={"era5": {"2t": 0}})
    extract_checkpoint_variables_metadata(model, None)
    assert not hasattr(model, "_ckpt_variables_metadata")


# --- shared-function ordering inside concrete loaders ---


def _install_order_recorders(
    model: nn.Module,
    calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the corrections, metadata steps and model load with call recorders.

    Each function is patched where it is looked up: corrections in ``loading.base``,
    metadata steps in ``loading.strategies``. The real ``load_state_dict`` still runs.
    """
    corrections = {
        "apply_checkpoint_format_migrations": "format",
        "apply_trainable_edge_perm_migration": "edge_perm",
        "refresh_checkpoint_processors": "refresh",
    }
    for function_name, label in corrections.items():

        def _correction_recorder(checkpoint: object, *_args: object, _label: str = label, **_kwargs: object) -> object:
            calls.append(_label)
            return checkpoint

        monkeypatch.setattr(loading_base, function_name, _correction_recorder)

    metadata_steps = {
        "warn_on_hparams_divergence": "warn",
        "preserve_anemoi_metadata": "preserve",
        "extract_checkpoint_variables_metadata": "extract",
    }
    for function_name, label in metadata_steps.items():

        def _recorder(*_args: object, _label: str = label, **_kwargs: object) -> None:
            calls.append(_label)

        monkeypatch.setattr(strategies, function_name, _recorder)

    original_load = model.load_state_dict

    def _load_recorder(*args: object, **kwargs: object) -> object:
        calls.append("load")
        return original_load(*args, **kwargs)

    model.load_state_dict = _load_recorder


@pytest.mark.asyncio
async def test_weights_only_invokes_shared_parity_functions_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """WeightsOnlyLoader runs the corrections, the load and the metadata steps in order.

    Edge-perm uses the live model and can rebuild the dict, thus it runs before refresh.
    """
    model = _MiniModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": model.state_dict()})
    loader = WeightsOnlyLoader()
    calls: list[str] = []
    _install_order_recorders(model, calls, monkeypatch)

    await loader.process(context)

    assert calls == ["format", "edge_perm", "refresh", "load", "warn", "preserve", "extract"]


@pytest.mark.asyncio
async def test_transfer_learning_invokes_shared_parity_functions_without_hparams_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TransferLearningLoader runs the same steps, but not the hparams warning."""
    model = _MiniModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": model.state_dict()})
    loader = TransferLearningLoader(skip_mismatched=True)
    calls: list[str] = []
    _install_order_recorders(model, calls, monkeypatch)

    await loader.process(context)

    assert calls == ["format", "edge_perm", "refresh", "load", "preserve", "extract"]
    assert "warn" not in calls
