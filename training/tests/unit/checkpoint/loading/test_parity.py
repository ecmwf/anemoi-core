# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the shared Lightning-parity functions in ``checkpoint.loading.base``.

These functions are the single home for the checkpoint-load parity steps, called
both by the pipeline loading strategies (via their ``_*`` wrappers) and by the
trainer's ``AnemoiLightningModule.on_load_checkpoint``. They are tested here
directly, independently of either caller.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from anemoi.training.checkpoint.loading.base import apply_checkpoint_format_migrations
from anemoi.training.checkpoint.loading.base import apply_trainable_edge_perm_migration
from anemoi.training.checkpoint.loading.base import extract_checkpoint_variables_metadata
from anemoi.training.checkpoint.loading.base import preserve_anemoi_metadata
from anemoi.training.checkpoint.loading.base import refresh_checkpoint_processors
from anemoi.training.checkpoint.loading.base import warn_on_hparams_divergence


class _ProcessorModel(torch.nn.Module):
    """Minimal model exposing a ``pre_processors`` buffer for the refresh tests."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.pre_processors = torch.nn.Module()
        self.pre_processors.register_buffer("w", torch.full((2,), value))


# --- refresh_checkpoint_processors -----------------------------------------


def test_refresh_replaces_processor_weights_in_place() -> None:
    model = _ProcessorModel(value=1.0)
    checkpoint = {
        "state_dict": {
            "model.pre_processors.w": torch.zeros(2),  # stale -> replaced from model
            "model.encoder.x": torch.zeros(1),  # not a processor -> untouched
        },
    }
    state_dict = checkpoint["state_dict"]

    refresh_checkpoint_processors(checkpoint, model, update_states=True, update_tendencies=False)

    assert torch.equal(state_dict["model.pre_processors.w"], torch.ones(2))
    assert "model.encoder.x" in state_dict


def test_refresh_noop_when_flags_off() -> None:
    checkpoint = {"state_dict": {"model.pre_processors.w": torch.zeros(2)}}
    refresh_checkpoint_processors(checkpoint, _ProcessorModel(1.0), update_states=False, update_tendencies=False)
    assert torch.equal(checkpoint["state_dict"]["model.pre_processors.w"], torch.zeros(2))


def test_refresh_noop_when_no_state_dict() -> None:
    checkpoint: dict = {}
    refresh_checkpoint_processors(checkpoint, _ProcessorModel(1.0), update_states=True, update_tendencies=True)
    assert checkpoint == {}


# --- preserve_anemoi_metadata ----------------------------------------------


def test_preserve_metadata_multi_dataset() -> None:
    model = SimpleNamespace()
    checkpoint = {
        "hyper_parameters": {
            "data_indices": {
                "era5": SimpleNamespace(name_to_index={"t2m": 0, "u10": 1}),
                "cerra": SimpleNamespace(name_to_index={"t2m": 0}),
            },
        },
    }
    preserve_anemoi_metadata(model, checkpoint)
    assert model._ckpt_model_name_to_index == {"era5": {"t2m": 0, "u10": 1}, "cerra": {"t2m": 0}}


def test_preserve_metadata_single_dataset_raises() -> None:
    checkpoint = {"hyper_parameters": {"data_indices": SimpleNamespace(name_to_index={"t2m": 0})}}
    with pytest.raises(TypeError, match="multi-dataset"):
        preserve_anemoi_metadata(SimpleNamespace(), checkpoint)


def test_preserve_metadata_missing_is_noop() -> None:
    model = SimpleNamespace()
    preserve_anemoi_metadata(model, {"hyper_parameters": {}})
    assert not hasattr(model, "_ckpt_model_name_to_index")


# --- extract_checkpoint_variables_metadata ---------------------------------


def test_extract_variables_metadata_noop_without_name_to_index() -> None:
    model = SimpleNamespace()
    extract_checkpoint_variables_metadata(model, {})
    assert not hasattr(model, "_ckpt_variables_metadata")


# --- warn_on_hparams_divergence --------------------------------------------


def test_warn_on_divergent_hparams(caplog: pytest.LogCaptureFixture) -> None:
    checkpoint = {"hyper_parameters": {"config": {"model": {"layers": 1}}}}
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence(checkpoint, {"model": {"layers": 2}})
    assert any("hparams differ" in record.getMessage() for record in caplog.records)


def test_no_warn_when_hparams_match(caplog: pytest.LogCaptureFixture) -> None:
    checkpoint = {"hyper_parameters": {"config": {"model": {"layers": 1}}}}
    with caplog.at_level(logging.WARNING):
        warn_on_hparams_divergence(checkpoint, {"model": {"layers": 1}})
    assert not [r for r in caplog.records if "hparams differ" in r.getMessage()]


def test_warn_noop_when_config_none() -> None:
    warn_on_hparams_divergence({"hyper_parameters": {"config": {"model": {}}}}, None)  # must not raise


# --- migration wrappers: tolerance / no-op ---------------------------------


def test_apply_format_migrations_none_passthrough() -> None:
    assert apply_checkpoint_format_migrations(None) is None


def test_apply_format_migrations_tolerates_incomplete_shape() -> None:
    # A raw state-dict checkpoint lacks hyper_parameters.config; the migration is
    # skipped (no raise) and the same object is returned.
    checkpoint = {"state_dict": {}}
    assert apply_checkpoint_format_migrations(checkpoint) is checkpoint


def test_apply_edge_perm_migration_none_model_passthrough() -> None:
    checkpoint = {"state_dict": {}}
    assert apply_trainable_edge_perm_migration(checkpoint, None) is checkpoint
