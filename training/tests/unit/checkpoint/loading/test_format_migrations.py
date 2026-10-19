# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for the format-migration step of ``apply_checkpoint_corrections``.

Old checkpoints get the chunking migration before ``load_state_dict``. Up-to-date ones do not.
The migration ledger and the processor geometry decide which checkpoints to migrate.

The ``fake_*`` fixtures test the in-memory fallback. ``spy_chunking_migration`` keeps the real ledger.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace
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


def _processor(**geometry: object) -> SimpleNamespace:
    """Return a processor config with the given ``num_layers`` and ``num_chunks``."""
    return SimpleNamespace(**geometry)


def _state_dict() -> dict:
    return {
        "linear.weight": torch.randn(4, 4),
        "linear.bias": torch.randn(4),
    }


def _ckpt(processor: object | None = None) -> dict:
    """Return a checkpoint with a processor config. The default has a chunking geometry."""
    if processor is None:
        processor = _processor(num_layers=4, num_chunks=2)
    return {
        "state_dict": _state_dict(),
        "pytorch-lightning_version": "2.6.5",
        "hyper_parameters": {"config": SimpleNamespace(model=SimpleNamespace(processor=processor))},
    }


def _ckpt_without_hparams() -> dict:
    """Return a checkpoint without ``hyper_parameters``."""
    return {"state_dict": _state_dict()}


def test_migration_invoked_exactly_once_per_process(
    fake_chunking_migration: MagicMock,
) -> None:
    """WeightsOnlyLoader.process() calls the migration helper exactly once."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    WeightsOnlyLoader().process(context)

    assert fake_chunking_migration.call_count == 1
    assert context.checkpoint_data["_migration_applied"] is True


def test_apply_format_migrations_replaces_checkpoint_data(
    fake_chunking_migration: MagicMock,
) -> None:
    """Helper reassigns context.checkpoint_data to the migrated dict."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    WeightsOnlyLoader()._apply_corrections(context)

    fake_chunking_migration.assert_called_once()
    assert context.checkpoint_data["_migration_applied"] is True


def test_no_checkpoint_data_is_noop() -> None:
    """No checkpoint_data → silent no-op (no ImportError, no crash)."""
    context = CheckpointContext(model=_Model(), checkpoint_data=None)

    WeightsOnlyLoader()._apply_corrections(context)

    assert context.checkpoint_data is None


def test_apply_trainable_edge_perm_migration_runs_model_dependent_migration(
    fake_edge_perm_migration: MagicMock,
) -> None:
    """The edge-permutation migration gets the model and its result replaces the checkpoint."""
    model = _Model()
    context = CheckpointContext(model=model, checkpoint_data=_ckpt())

    WeightsOnlyLoader()._apply_corrections(context)

    fake_edge_perm_migration.assert_called_once()
    _, called_model = fake_edge_perm_migration.call_args.args
    assert called_model is model  # migration is model-dependent
    assert context.checkpoint_data["_edge_perm_applied"] is True
    assert context.checkpoint_data["_model"] is model


def test_edge_perm_migration_invoked_during_process(
    fake_edge_perm_migration: MagicMock,
) -> None:
    """``process`` applies the edge-permutation migration once."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    WeightsOnlyLoader().process(context)

    assert fake_edge_perm_migration.call_count == 1
    assert context.checkpoint_data["_edge_perm_applied"] is True


def test_edge_perm_migration_noop_without_model() -> None:
    """Without a model, the edge-permutation migration is skipped."""
    ckpt = _ckpt()
    context = CheckpointContext(model=None, checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

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

    WeightsOnlyLoader()._apply_corrections(context)

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
    WeightsOnlyLoader()._apply_corrections(context)

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
        WeightsOnlyLoader()._apply_corrections(context)


# ---------------------------------------------------------------------------
# The migration ledger and processor geometry decide if the migration applies.
# ---------------------------------------------------------------------------


@pytest.fixture
def spy_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Spy on ``chunking_fix`` and keep ``sys.modules`` unchanged.

    The real ``Migrator`` stays available, so the ledger path runs.
    """
    spy = MagicMock(side_effect=lambda ckpt: {**ckpt, "_migration_applied": True})
    monkeypatch.setattr(
        "anemoi.training.checkpoint.loading.base._load_chunking_fix_migration",
        lambda: spy,
    )
    return spy


def _ledger(*names: str) -> list[dict]:
    """Return a migration ledger that records ``names`` as applied."""
    from anemoi.models.migrations.migrator import MigrationMetadata

    return [
        {
            "name": name,
            "metadata": MigrationMetadata(versions={"migration": "1.0.0", "anemoi-models": "0.11.0"}),
            "signature": f"signature-of-{name}",
        }
        for name in names
    ]


def _all_shipped_migration_names() -> list[str]:
    """Return all migrations of the installed anemoi-models, in ledger order.

    Not hardcoded, so the list stays complete when a new migration is added.
    """
    from anemoi.models.migrations import Migrator

    return [migration.name for migration in Migrator()._grouped_migrations[-1]]


def test_ledger_recorded_migration_is_not_reapplied(spy_chunking_migration: MagicMock) -> None:
    """A checkpoint with a complete ledger is not migrated again."""
    ckpt = _ckpt()
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names())
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    spy_chunking_migration.assert_not_called()
    assert context.checkpoint_data is ckpt
    assert "_migration_applied" not in context.checkpoint_data


def test_ledger_recording_chunking_fix_is_not_treated_as_up_to_date(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """A ledger with ``chunking_fix`` but without newer migrations causes a warning.

    Almost all real checkpoints record ``chunking_fix``, so that name alone is not sufficient.
    """
    ckpt = _ckpt()
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names()[:3])
    assert any(
        "chunking_fix" in name for name in _all_shipped_migration_names()[:3]
    ), "fixture assumes chunking_fix is within the first three migrations"
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)

    assert "behind the installed anemoi-models" in caplog.text
    # The load continues.
    spy_chunking_migration.assert_called_once()


def test_empty_ledger_still_migrates(spy_chunking_migration: MagicMock) -> None:
    """Positive control: an empty ledger with a usable geometry is migrated."""
    ckpt = _ckpt()
    ckpt["migrations"] = []
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    spy_chunking_migration.assert_called_once()
    assert context.checkpoint_data["_migration_applied"] is True


@pytest.mark.parametrize(
    ("processor", "shape"),
    [
        (_processor(), "NoOpProcessor (autoencoder) — declares neither"),
        (_processor(num_layers=4), "PointWiseMLPProcessor — no num_chunks"),
        (_processor(num_layers=4, num_chunks=0), "num_chunks=0 — schema-legal, divides by zero"),
        (_processor(num_layers=4, num_chunks=None), "num_chunks=None"),
    ],
)
def test_processor_without_usable_geometry_is_not_migrated(
    spy_chunking_migration: MagicMock,
    processor: object,
    shape: str,
) -> None:
    """A processor without a usable ``num_layers``/``num_chunks`` is not migrated."""
    ckpt = _ckpt(processor)
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    assert spy_chunking_migration.call_count == 0, f"migration ran for {shape}"
    assert context.checkpoint_data is ckpt


def test_raw_state_dict_save_is_not_migrated(spy_chunking_migration: MagicMock) -> None:
    """A checkpoint without ``hyper_parameters`` is not migrated."""
    ckpt = _ckpt_without_hparams()
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    spy_chunking_migration.assert_not_called()
    assert context.checkpoint_data is ckpt


def test_real_chunking_fix_leaves_autoencoder_checkpoint_intact() -> None:
    """The real ``chunking_fix`` does not change a NoOpProcessor checkpoint."""
    ckpt = _ckpt(_processor())  # NoOpProcessor: no num_layers, no num_chunks
    original = {k: v.clone() for k, v in ckpt["state_dict"].items()}
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader(strict=False).process(context)

    for key, value in original.items():
        assert torch.equal(context.checkpoint_data["state_dict"][key], value), f"{key} was rewritten"


# ---------------------------------------------------------------------------
# An incomplete ledger causes a warning. The load continues.
# ---------------------------------------------------------------------------


def test_incomplete_ledger_warns_naming_the_migrations_and_still_loads(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """The warning gives the missing migrations and the fix command. The load continues.

    The load does not fail because the fix changes a local file. That is not always possible.
    """
    ckpt = _ckpt()
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names()[:3])
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt, checkpoint_path="/scratch/last.ckpt")

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    for name in _all_shipped_migration_names()[3:]:
        assert name in caplog.text
    assert "anemoi-models migration sync /scratch/last.ckpt" in caplog.text
    assert "rewrites the checkpoint in place" in caplog.text
    spy_chunking_migration.assert_called_once()
    assert context.checkpoint_data is not None


def test_remote_checkpoint_warning_does_not_tell_the_user_to_migrate_a_url(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """Without a local file, the warning does not give the CLI command."""
    ckpt = _ckpt()
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names()[:3])
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    assert "downloading it" in caplog.text
    assert "migration sync None" not in caplog.text
    spy_chunking_migration.assert_called_once()


def test_non_training_checkpoint_is_not_warned_about_missing_migrations(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """A checkpoint without ``pytorch-lightning_version`` gets no missing-migration warning."""
    ckpt = _ckpt()
    del ckpt["pytorch-lightning_version"]
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names()[:3])
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    assert "behind the installed anemoi-models" not in caplog.text
    # The in-memory path still runs.
    spy_chunking_migration.assert_called_once()


def test_unreadable_ledger_is_treated_as_unknown_not_current(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """An unreadable ledger does not stop the load and is not "up to date"."""
    ckpt = _ckpt()
    ckpt["migrations"] = [{"name": "not-a-real-ledger-entry"}]
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    spy_chunking_migration.assert_called_once()


# ---------------------------------------------------------------------------
# Autoencoder checkpoints (no processor) never get chunking_fix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recorded", "expect_warning"),
    [
        pytest.param(0, False, id="empty ledger"),
        pytest.param(2, True, id="ledger from before chunking_fix"),
    ],
)
def test_autoencoder_checkpoint_behind_the_ledger_is_not_migrated_on_a_pipeline_load(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
    recorded: int,
    expect_warning: bool,
) -> None:
    """An old autoencoder checkpoint is not migrated and loads as saved.

    The ledger can cause a warning. The fallback does nothing because ``NoOpProcessor`` has no geometry.
    """
    names = _all_shipped_migration_names()[:recorded]
    assert not any("chunking_fix" in name for name in names), "fixture assumes chunking_fix is not recorded"
    ckpt = _ckpt(_processor())  # NoOpProcessor: no num_layers, no num_chunks
    ckpt["migrations"] = _ledger(*names)
    saved = {key: value.clone() for key, value in ckpt["state_dict"].items()}
    model = _Model()
    context = CheckpointContext(model=model, checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader().process(context)

    spy_chunking_migration.assert_not_called()
    assert ("behind the installed anemoi-models" in caplog.text) is expect_warning
    assert "_migration_applied" not in context.checkpoint_data
    for key, value in saved.items():
        assert torch.equal(model.state_dict()[key], value), f"{key} was not loaded as saved"
