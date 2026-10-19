# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the migration steps of ``apply_checkpoint_corrections``.

``chunking_fix`` runs only if the processor config supports it. The edge-permutation migration always runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


@pytest.fixture
def spy_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    spy = MagicMock(side_effect=lambda ckpt: ckpt)
    monkeypatch.setattr(loading_base, "_chunking_fix", spy)
    return spy


@pytest.fixture
def spy_edge_perm_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    spy = MagicMock(side_effect=lambda ckpt, _model: ckpt)
    monkeypatch.setattr(loading_base, "_trainable_edge_perm_fix", spy)
    return spy


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


def _processor(**geometry: object) -> SimpleNamespace:
    """Return a processor config with the given ``num_layers`` and ``num_chunks``."""
    return SimpleNamespace(**geometry)


def _ckpt(processor: object | None = None) -> dict:
    """Return a checkpoint with a processor config. The default has a chunking geometry."""
    if processor is None:
        processor = _processor(num_layers=4, num_chunks=2)
    return {
        "state_dict": {"linear.weight": torch.randn(4, 4), "linear.bias": torch.randn(4)},
        "hyper_parameters": {"config": SimpleNamespace(model=SimpleNamespace(processor=processor))},
    }


def test_process_applies_each_migration_once(
    spy_chunking_migration: MagicMock,
    spy_edge_perm_migration: MagicMock,
) -> None:
    """``process`` applies both migrations once. The edge-permutation migration gets the model."""
    model = _Model()
    checkpoint = _ckpt()

    WeightsOnlyLoader().process(CheckpointContext(model=model, checkpoint_data=checkpoint))

    spy_chunking_migration.assert_called_once_with(checkpoint)
    spy_edge_perm_migration.assert_called_once_with(checkpoint, model)


@pytest.mark.parametrize(
    "checkpoint",
    [
        pytest.param(_ckpt(_processor()), id="NoOpProcessor: neither"),
        pytest.param(_ckpt(_processor(num_layers=4)), id="no num_chunks"),
        pytest.param(_ckpt(_processor(num_layers=4, num_chunks=0)), id="num_chunks=0 divides by zero"),
        pytest.param(_ckpt(_processor(num_layers=4, num_chunks=None)), id="num_chunks=None"),
        pytest.param({"state_dict": {"linear.weight": torch.randn(4, 4)}}, id="no hyper_parameters"),
    ],
)
def test_chunking_fix_is_skipped_without_a_usable_processor_config(
    spy_chunking_migration: MagicMock,
    checkpoint: dict,
) -> None:
    apply_checkpoint_corrections(checkpoint, _Model(), None)

    spy_chunking_migration.assert_not_called()


def test_autoencoder_checkpoint_loads_as_saved() -> None:
    """A NoOpProcessor checkpoint loads as saved with the real migrations."""
    checkpoint = _ckpt(_processor())
    saved = {key: value.clone() for key, value in checkpoint["state_dict"].items()}
    model = _Model()

    WeightsOnlyLoader().process(CheckpointContext(model=model, checkpoint_data=checkpoint))

    for key, value in saved.items():
        assert torch.equal(model.state_dict()[key], value), f"{key} was not loaded as saved"
