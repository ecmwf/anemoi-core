# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for the format-migration step of ``apply_checkpoint_corrections``.

Old checkpoints carrying the pre-chunking attention-head layout must be
rewritten before any ``load_state_dict`` attempt, and up-to-date ones must be
left alone. ``checkpoint.loading.base.apply_checkpoint_format_migrations``
decides which is which from the migration ledger the checkpoint carries and from
the processor geometry the migration reads, rather than by running the migration
and interpreting the exception it throws.

The tests split accordingly: the ``fake_*`` fixtures inject a migration through
``sys.modules`` to exercise the in-memory fallback, while ``spy_chunking_migration``
patches only the resolver so the real ledger stays reachable.
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
    """A processor config node declaring (or omitting) the chunking geometry.

    ``chunking_fix`` reads ``num_layers`` and ``num_chunks`` off this node. A
    ``NoOpProcessor`` (autoencoders) declares neither; ``PointWiseMLPProcessor``
    declares only ``num_layers``.
    """
    return SimpleNamespace(**geometry)


def _state_dict() -> dict:
    return {
        "linear.weight": torch.randn(4, 4),
        "linear.bias": torch.randn(4),
    }


def _ckpt(processor: object | None = None) -> dict:
    """A checkpoint whose processor declares the geometry ``chunking_fix`` reads.

    Pass ``processor`` to model a different processor shape.
    """
    if processor is None:
        processor = _processor(num_layers=4, num_chunks=2)
    return {
        "state_dict": _state_dict(),
        "pytorch-lightning_version": "2.6.5",
        "hyper_parameters": {"config": SimpleNamespace(model=SimpleNamespace(processor=processor))},
    }


def _ckpt_without_hparams() -> dict:
    """A raw ``state_dict`` save — no ``hyper_parameters`` tree at all."""
    return {"state_dict": _state_dict()}


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
    """The helper runs the model-dependent edge-perm migration and reassigns the result."""
    model = _Model()
    context = CheckpointContext(model=model, checkpoint_data=_ckpt())

    WeightsOnlyLoader()._apply_corrections(context)

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
# Applicability is decided from the checkpoint's own migration ledger and its
# processor geometry — not by running the migration and catching the fallout.
# ---------------------------------------------------------------------------


@pytest.fixture
def spy_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Observe ``chunking_fix`` without disturbing ``sys.modules``.

    The ``fake_chunking_migration`` fixture replaces the ``scripts`` package with a
    plain module, which stops ``Migrator`` constructing at all. Patching the
    resolver instead leaves the real migration ledger reachable, so these tests
    exercise the ledger path rather than the fallback.
    """
    spy = MagicMock(side_effect=lambda ckpt: {**ckpt, "_migration_applied": True})
    monkeypatch.setattr(
        "anemoi.training.checkpoint.loading.base._load_chunking_fix_migration",
        lambda: spy,
    )
    return spy


def _ledger(*names: str) -> list[dict]:
    """A checkpoint migration ledger recording ``names`` as already applied."""
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
    """Every migration the installed anemoi-models ships, in ledger order.

    Read from the migrator rather than hardcoded, so a new migration does not
    quietly turn the up-to-date test below into a partial-ledger test.
    """
    from anemoi.models.migrations import Migrator

    return [migration.name for migration in Migrator()._grouped_migrations[-1]]


def test_ledger_recorded_migration_is_not_reapplied(spy_chunking_migration: MagicMock) -> None:
    """A checkpoint missing no migration is left alone.

    Every checkpoint anemoi writes carries this ledger, so this is the common
    case: loading an up-to-date checkpoint must not re-run a migration it has
    already had, regardless of which loading strategy is used.
    """
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
    """A ledger through chunking_fix but missing newer migrations is NOT current.

    ``register_migrations`` stamps every migration the writing version knew, not the
    ones it applied, so essentially every real checkpoint records ``chunking_fix``.
    Gating on that one name declared almost every checkpoint current and skipped all
    the newer migrations in silence. It is no longer silent.
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
    # And it is a warning, not a refusal: the load carried on to the in-memory path.
    spy_chunking_migration.assert_called_once()


def test_empty_ledger_still_migrates(spy_chunking_migration: MagicMock) -> None:
    """Positive control: an empty ledger with usable geometry still migrates.

    Guards the test above against passing vacuously: the skip must come from the
    ledger, not from the spy never being reachable.
    """
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
    """Processors that do not declare a usable chunking geometry are not candidates.

    ``chunking_fix`` reads ``num_layers``/``num_chunks`` and divides by the latter.
    Screening for that up front means an autoencoder load neither runs the
    migration nor relies on an exception handler to undo it, and the
    ``num_chunks`` 0/None shapes cannot raise ZeroDivisionError/TypeError out of
    a checkpoint load.
    """
    ckpt = _ckpt(processor)
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    assert spy_chunking_migration.call_count == 0, f"migration ran for {shape}"
    assert context.checkpoint_data is ckpt


def test_raw_state_dict_save_is_not_migrated(spy_chunking_migration: MagicMock) -> None:
    """A checkpoint with no ``hyper_parameters`` tree is not a migration candidate."""
    ckpt = _ckpt_without_hparams()
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    spy_chunking_migration.assert_not_called()
    assert context.checkpoint_data is ckpt


@pytest.mark.asyncio
async def test_real_chunking_fix_leaves_autoencoder_checkpoint_intact() -> None:
    """The real migration, through a real strategy, on a NoOpProcessor checkpoint.

    Every other test here drives a fake migration. This one runs the migration
    anemoi-models actually ships, so it would catch the migration reaching into a
    processor an autoencoder does not have. The state dict must come through
    byte-identical and the load must not raise.
    """
    ckpt = _ckpt(_processor())  # NoOpProcessor: no num_layers, no num_chunks
    original = {k: v.clone() for k, v in ckpt["state_dict"].items()}
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    await WeightsOnlyLoader(strict=False).process(context)

    for key, value in original.items():
        assert torch.equal(context.checkpoint_data["state_dict"][key], value), f"{key} was rewritten"


# ---------------------------------------------------------------------------
# An incomplete ledger is reported precisely and the load proceeds.
# ---------------------------------------------------------------------------


def test_incomplete_ledger_warns_naming_the_migrations_and_still_loads(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """An out-of-date checkpoint is reported precisely and still loads.

    Refusing would block work that has always succeeded: most of what a ledger can
    be missing does not affect this load, and the remedy is a local file operation
    that a read-only ``anemoi-training evaluate`` or a remote source cannot perform.
    Migrating in-process (``Migrator.sync``) re-reads and deep-copies the file and
    runs setup hooks that mutate ``sys.modules`` of the live process. The defect
    being fixed is that the old gate reported these checkpoints as *up to date*;
    saying precisely what is missing is the fix.
    """
    ckpt = _ckpt()
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names()[:3])
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt, checkpoint_path="/scratch/last.ckpt")

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    for name in _all_shipped_migration_names()[3:]:
        assert name in caplog.text
    assert "anemoi-models migration sync /scratch/last.ckpt" in caplog.text
    # The remedy rewrites the user's checkpoint, so the warning says so.
    assert "rewrites the checkpoint in place" in caplog.text
    # And the load went ahead rather than being refused.
    spy_chunking_migration.assert_called_once()
    assert context.checkpoint_data is not None


def test_remote_checkpoint_warning_does_not_tell_the_user_to_migrate_a_url(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """With no local file there is nothing to run the CLI against; say so."""
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
    """A raw state_dict / inference checkpoint has no ledger to be behind on.

    It carries no ``pytorch-lightning_version``, which is what ``Migrator._load_ckpt``
    itself gates on. Warning about "missing migrations" would be noise on every
    S3/HTTP weights-only load of a non-Lightning checkpoint.
    """
    ckpt = _ckpt()
    del ckpt["pytorch-lightning_version"]
    ckpt["migrations"] = _ledger(*_all_shipped_migration_names()[:3])
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    assert "behind the installed anemoi-models" not in caplog.text
    # Falls through to the narrow in-memory path, exactly as before.
    spy_chunking_migration.assert_called_once()


def test_unreadable_ledger_is_treated_as_unknown_not_current(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """A ledger the migrator cannot resolve neither blocks the load nor claims it is current."""
    ckpt = _ckpt()
    ckpt["migrations"] = [{"name": "not-a-real-ledger-entry"}]
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    with caplog.at_level(logging.WARNING):
        WeightsOnlyLoader()._apply_corrections(context)  # must not raise

    # "Cannot tell" falls through to the in-memory path rather than reporting up to date.
    spy_chunking_migration.assert_called_once()
