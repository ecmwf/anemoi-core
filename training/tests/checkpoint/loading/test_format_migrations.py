"""Regression tests for LoadingStrategy._apply_format_migrations.

Mirrors the legacy ``chunking_fix_migration(checkpoint)`` call in
``anemoi.training.utils.checkpoint.transfer_learning_loading`` so old
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
