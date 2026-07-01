# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for LoadingStrategy._apply_format_migrations.

Applies the ``chunking_fix`` migration (via
``checkpoint.loading.base.apply_checkpoint_format_migrations``) so old
checkpoints with the pre-chunking attention head layout get rewritten
before any ``load_state_dict`` attempt.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


@pytest.fixture
def fake_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake anemoi.models.migrations.scripts.chunking_fix.migrate."""
    fake_migrate = MagicMock(side_effect=lambda ckpt: {**ckpt, "_migration_applied": True})

    pkg_path = "anemoi.models.migrations.scripts.chunking_fix"
    module = types.ModuleType(pkg_path)
    module.migrate = fake_migrate

    parent_path = "anemoi.models.migrations.scripts"
    parent_module = sys.modules.get(parent_path) or types.ModuleType(parent_path)
    parent_module.chunking_fix = module

    monkeypatch.setitem(sys.modules, parent_path, parent_module)
    monkeypatch.setitem(sys.modules, pkg_path, module)
    return fake_migrate


@pytest.fixture
def fake_edge_perm_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake anemoi.models.migrations.scripts.trainable_edge_perm_fix.migrate."""
    fake_migrate = MagicMock(side_effect=lambda ckpt, model: {**ckpt, "_edge_perm_applied": True, "_model": model})

    pkg_path = "anemoi.models.migrations.scripts.trainable_edge_perm_fix"
    module = types.ModuleType(pkg_path)
    module.migrate = fake_migrate

    parent_path = "anemoi.models.migrations.scripts"
    parent_module = sys.modules.get(parent_path) or types.ModuleType(parent_path)
    parent_module.trainable_edge_perm_fix = module

    monkeypatch.setitem(sys.modules, parent_path, parent_module)
    monkeypatch.setitem(sys.modules, pkg_path, module)
    return fake_migrate


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


def _ckpt() -> dict:
    return {
        "state_dict": {
            "linear.weight": torch.randn(4, 4),
            "linear.bias": torch.randn(4),
        },
    }


@pytest.mark.asyncio
async def test_migration_invoked_exactly_once_per_process(
    fake_chunking_migration: MagicMock,
) -> None:
    """WeightsOnlyLoader.process() calls the migration helper exactly once."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    await WeightsOnlyLoader().process(context)

    assert fake_chunking_migration.call_count == 1
    assert context.checkpoint_data["_migration_applied"] is True


def test_apply_format_migrations_replaces_checkpoint_data(
    fake_chunking_migration: MagicMock,
) -> None:
    """Helper reassigns context.checkpoint_data to the migrated dict."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    WeightsOnlyLoader()._apply_format_migrations(context)

    fake_chunking_migration.assert_called_once()
    assert context.checkpoint_data["_migration_applied"] is True


def test_no_checkpoint_data_is_noop() -> None:
    """No checkpoint_data → silent no-op (no ImportError, no crash)."""
    context = CheckpointContext(model=_Model(), checkpoint_data=None)

    WeightsOnlyLoader()._apply_format_migrations(context)

    assert context.checkpoint_data is None


def test_apply_trainable_edge_perm_migration_runs_model_dependent_migration(
    fake_edge_perm_migration: MagicMock,
) -> None:
    """The helper runs the model-dependent edge-perm migration and reassigns the result."""
    model = _Model()
    context = CheckpointContext(model=model, checkpoint_data=_ckpt())

    WeightsOnlyLoader()._apply_trainable_edge_perm_migration(context)

    fake_edge_perm_migration.assert_called_once()
    _, called_model = fake_edge_perm_migration.call_args.args
    assert called_model is model  # migration is model-dependent
    assert context.checkpoint_data["_edge_perm_applied"] is True
    assert context.checkpoint_data["_model"] is model


@pytest.mark.asyncio
async def test_edge_perm_migration_invoked_during_process(
    fake_edge_perm_migration: MagicMock,
) -> None:
    """WeightsOnlyLoader.process() applies the runtime edge-perm migration exactly once."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    await WeightsOnlyLoader().process(context)

    assert fake_edge_perm_migration.call_count == 1
    assert context.checkpoint_data["_edge_perm_applied"] is True


def test_edge_perm_migration_noop_without_model() -> None:
    """No model → the model-dependent migration is skipped (no crash)."""
    ckpt = _ckpt()
    context = CheckpointContext(model=None, checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_trainable_edge_perm_migration(context)

    assert context.checkpoint_data is ckpt


def test_missing_migration_module_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older anemoi-models without the migration module → no-op (not a crash)."""
    # Ensure neither candidate path resolves
    for name in (
        "anemoi.models.migrations.scripts.chunking_fix",
        "anemoi.models.migrations.scripts.1762857428_chunking_fix",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    ckpt = _ckpt()
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_format_migrations(context)

    assert context.checkpoint_data is ckpt


def test_incomplete_checkpoint_shape_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration that raises KeyError on minimal test checkpoints → no-op."""

    def raising_migrate(_ckpt: dict) -> dict:
        msg = "hyper_parameters"
        raise KeyError(msg)

    module = types.ModuleType("anemoi.models.migrations.scripts.chunking_fix")
    module.migrate = raising_migrate
    monkeypatch.setitem(sys.modules, "anemoi.models.migrations.scripts.chunking_fix", module)

    ckpt = _ckpt()
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    # Should not propagate the KeyError
    WeightsOnlyLoader()._apply_format_migrations(context)

    # And should leave the checkpoint untouched
    assert context.checkpoint_data is ckpt


def test_unexpected_migration_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the migration raises something other than KeyError/AttributeError, surface it.

    We deliberately narrowed the except clause: TypeError (signature mismatch in
    the migration itself, for instance) should not be silently swallowed.
    """

    def buggy_migrate(_ckpt: dict, _extra: int) -> dict:  # wrong signature
        msg = "called with extra positional arg"
        raise RuntimeError(msg)

    module = types.ModuleType("anemoi.models.migrations.scripts.chunking_fix")
    module.migrate = buggy_migrate
    monkeypatch.setitem(sys.modules, "anemoi.models.migrations.scripts.chunking_fix", module)

    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    with pytest.raises(TypeError):
        # buggy_migrate(_ckpt) is missing _extra → TypeError from the call itself
        WeightsOnlyLoader()._apply_format_migrations(context)
