# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for the migration steps of ``apply_checkpoint_corrections``.

Old checkpoints carrying the pre-chunking attention-head layout must be
rewritten before any ``load_state_dict`` attempt. ``chunking_fix`` runs only on
a checkpoint whose processor declares the geometry it reads, decided up front
rather than by running the migration and interpreting the exception it throws.
Whatever a migration returns is the checkpoint the load uses.

The ``fake_*`` fixtures replace the migration callables ``checkpoint.loading.base``
imported from anemoi-models; the ``real`` tests run the shipped migrations.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


@pytest.fixture
def fake_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``chunking_fix.migrate`` with a fake that returns a new dict."""
    fake_migrate = MagicMock(side_effect=lambda ckpt: {**ckpt, "_migration_applied": True})
    monkeypatch.setattr(loading_base, "_chunking_fix", fake_migrate)
    return fake_migrate


@pytest.fixture
def fake_edge_perm_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``trainable_edge_perm_fix.migrate`` with a fake that returns a new dict."""
    fake_migrate = MagicMock(side_effect=lambda ckpt, model: {**ckpt, "_edge_perm_applied": True, "_model": model})
    monkeypatch.setattr(loading_base, "_trainable_edge_perm_fix", fake_migrate)
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


def test_corrections_run_the_model_dependent_edge_perm_migration(
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


def test_edge_perm_migration_invoked_during_process(
    fake_edge_perm_migration: MagicMock,
) -> None:
    """WeightsOnlyLoader.process() applies the runtime edge-perm migration exactly once."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    WeightsOnlyLoader().process(context)

    assert fake_edge_perm_migration.call_count == 1
    assert context.checkpoint_data["_edge_perm_applied"] is True


def test_edge_perm_migration_noop_without_model() -> None:
    """No model → the model-dependent migration is skipped (no crash)."""
    ckpt = _ckpt()
    context = CheckpointContext(model=None, checkpoint_data=ckpt)

    WeightsOnlyLoader()._apply_corrections(context)

    assert context.checkpoint_data is ckpt


def test_migration_returning_a_new_dict_is_what_the_model_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration that builds a new checkpoint, rather than editing the old one, is honoured.

    The corrections run before the state dict is extracted, and the strategy loads
    the dict the migration returned. The weights in the original object must not
    reach the model.
    """
    original = _ckpt()
    migrated_state = {key: torch.full_like(value, 7.0) for key, value in original["state_dict"].items()}

    def rebuilding_migrate(ckpt: dict) -> dict:
        return {**ckpt, "state_dict": migrated_state}

    monkeypatch.setattr(loading_base, "_chunking_fix", rebuilding_migrate)
    model = _Model()
    context = CheckpointContext(model=model, checkpoint_data=original)

    WeightsOnlyLoader().process(context)

    assert context.checkpoint_data is not original
    assert context.checkpoint_data["state_dict"] is migrated_state
    for key, value in model.state_dict().items():
        assert torch.all(value == 7.0), f"{key} came from the pre-migration dict"


def test_migration_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration that fails on a candidate checkpoint surfaces its error.

    The geometry screen decides applicability, so there is no ``except`` to absorb
    a real fault.
    """

    def buggy_migrate(_ckpt: dict) -> dict:
        msg = "processor.proc"
        raise KeyError(msg)

    monkeypatch.setattr(loading_base, "_chunking_fix", buggy_migrate)
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

    with pytest.raises(KeyError, match=r"processor\.proc"):
        WeightsOnlyLoader()._apply_corrections(context)


# ---------------------------------------------------------------------------
# Applicability is decided from the processor geometry, not by running the
# migration and catching the fallout.
# ---------------------------------------------------------------------------


@pytest.fixture
def spy_chunking_migration(fake_chunking_migration: MagicMock) -> MagicMock:
    """Observe ``chunking_fix`` calls."""
    return fake_chunking_migration


def test_usable_geometry_is_migrated(spy_chunking_migration: MagicMock) -> None:
    """Positive control for the screen tests below: a chunked processor is migrated."""
    context = CheckpointContext(model=_Model(), checkpoint_data=_ckpt())

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


def test_real_chunking_fix_leaves_autoencoder_checkpoint_intact() -> None:
    """The real migration, through a real strategy, on a NoOpProcessor checkpoint.

    Every other test here drives a fake migration. This one runs the migration
    anemoi-models actually ships, so it would catch the migration reaching into a
    processor an autoencoder does not have. The state dict must come through
    byte-identical and the load must not raise.
    """
    ckpt = _ckpt(_processor())  # NoOpProcessor: no num_layers, no num_chunks
    original = {k: v.clone() for k, v in ckpt["state_dict"].items()}
    context = CheckpointContext(model=_Model(), checkpoint_data=ckpt)

    WeightsOnlyLoader(strict=False).process(context)

    for key, value in original.items():
        assert torch.equal(context.checkpoint_data["state_dict"][key], value), f"{key} was rewritten"


def test_autoencoder_checkpoint_is_not_migrated_on_a_pipeline_load(spy_chunking_migration: MagicMock) -> None:
    """An autoencoder checkpoint on a weights-only load.

    A ``NoOpProcessor`` declares no geometry to rewrite, so the migration is never
    called and the weights load exactly as saved.
    """
    ckpt = _ckpt(_processor())  # NoOpProcessor: no num_layers, no num_chunks
    saved = {key: value.clone() for key, value in ckpt["state_dict"].items()}
    model = _Model()
    context = CheckpointContext(model=model, checkpoint_data=ckpt)

    WeightsOnlyLoader().process(context)

    spy_chunking_migration.assert_not_called()
    assert "_migration_applied" not in context.checkpoint_data
    for key, value in saved.items():
        assert torch.equal(model.state_dict()[key], value), f"{key} was not loaded as saved"


def test_real_chunking_fix_is_idempotent() -> None:
    """Running the shipped ``chunking_fix`` on an already-migrated checkpoint changes nothing.

    With no ledger deciding whether it already ran, every load of a chunked
    checkpoint applies it, including loads of checkpoints it has rewritten before.
    """
    weight = torch.randn(2)
    ckpt = _ckpt()  # num_layers=4, num_chunks=2: two blocks per chunk
    ckpt["state_dict"] = {"model.model.processor.proc.1.blocks.0.mlp.weight": weight}

    once = loading_base.apply_checkpoint_corrections(ckpt, None, None)
    assert list(once["state_dict"]) == ["model.model.processor.proc.2.mlp.weight"]

    twice = loading_base.apply_checkpoint_corrections(once, None, None)
    assert list(twice["state_dict"]) == ["model.model.processor.proc.2.mlp.weight"]
    assert twice["state_dict"]["model.model.processor.proc.2.mlp.weight"] is weight
