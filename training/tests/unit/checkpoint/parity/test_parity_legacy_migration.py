# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare the legacy transfer-learning load with ``TransferLearningLoader``.

Both paths load the same checkpoint into the same model, and the results must match.
One difference is intentional: the pipeline skips ``chunking_fix`` if the
checkpoint ledger records it. The legacy code always ran it.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Legacy implementation (copy from main).
# ---------------------------------------------------------------------------


def _legacy_filter_state_dict_size_mismatches(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> None:
    """Copy of ``utils.checkpoint._filter_state_dict_size_mismatches`` from main."""
    for key in list(state_dict):
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            del state_dict[key]


def _legacy_transfer_learning_loading(
    model: nn.Module,
    checkpoint: dict[str, Any],
    ckpt_path: Path | str,
    chunking_fix_migration: Any,
    trainable_edge_perm_fix_migration: Any,
) -> nn.Module:
    """Copy of ``utils.checkpoint.transfer_learning_loading`` from main, without ``torch.load``.

    The migrations are arguments, so both paths can use the same fakes.
    """
    checkpoint = chunking_fix_migration(checkpoint)

    model._update_checkpoint_state_dict_for_load(checkpoint)

    state_dict = checkpoint["state_dict"]
    _legacy_filter_state_dict_size_mismatches(state_dict, model.state_dict())

    checkpoint = trainable_edge_perm_fix_migration(checkpoint, model)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict, strict=False)

    data_indices = checkpoint["hyper_parameters"]["data_indices"]
    if isinstance(data_indices, dict):
        model._ckpt_model_name_to_index = {
            dataset_name: indices.name_to_index for dataset_name, indices in data_indices.items()
        }
    else:
        msg = (
            f"Checkpoint at '{ckpt_path}' was created with an older version of anemoi-core "
            "that does not support multi-dataset training. This checkpoint is incompatible "
            "with transfer learning in the current version."
        )
        raise TypeError(msg)

    model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(
        checkpoint,
        model._ckpt_model_name_to_index,
    )
    return model


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


class _Model(nn.Module):
    """Model with a no-op legacy processor-refresh hook.

    The run config also disables the refresh, so the tests compare only migration,
    filter and metadata.
    """

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.head = nn.Linear(width, 2)

    def _update_checkpoint_state_dict_for_load(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint


def _index_collection(name_to_index: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(name_to_index=name_to_index)


def _checkpoint(width: int = 4) -> dict[str, Any]:
    """Return a checkpoint with no migration ledger."""
    torch.manual_seed(0)
    source = _Model(width)
    return {
        "state_dict": dict(source.state_dict()),
        "pytorch-lightning_version": "2.6.5",
        "hyper_parameters": {
            "data_indices": {"era5": _index_collection({"2t": 0, "10u": 1})},
            "config": SimpleNamespace(model=SimpleNamespace(processor=SimpleNamespace(num_layers=4, num_chunks=2))),
        },
    }


def _run_config() -> Any:
    """Return a run config with the processor refresh disabled."""
    return OmegaConf.create({"training": {"update_ds_stats_on_ckpt_load": {"states": False, "tendencies": False}}})


@pytest.fixture
def fake_migrations(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install fake chunking_fix and trainable_edge_perm_fix migrations.

    Each fake changes one tensor, so the effect shows in the loaded weights.
    """

    def _chunking(ckpt: dict[str, Any]) -> dict[str, Any]:
        migrated = {**ckpt, "state_dict": dict(ckpt["state_dict"])}
        migrated["state_dict"]["linear.weight"] = migrated["state_dict"]["linear.weight"] * 2.0
        return migrated

    def _edge_perm(ckpt: dict[str, Any], model: nn.Module) -> dict[str, Any]:
        del model
        migrated = {**ckpt, "state_dict": dict(ckpt["state_dict"])}
        migrated["state_dict"]["linear.bias"] = migrated["state_dict"]["linear.bias"] + 1.0
        return migrated

    chunking = MagicMock(side_effect=_chunking)
    edge_perm = MagicMock(side_effect=_edge_perm)

    scripts = "anemoi.models.migrations.scripts"
    parent = sys.modules.get(scripts) or types.ModuleType(scripts)
    for name, migrate in (("chunking_fix", chunking), ("trainable_edge_perm_fix", edge_perm)):
        module = types.ModuleType(f"{scripts}.{name}")
        module.migrate = migrate
        setattr(parent, name, module)
        monkeypatch.setitem(sys.modules, f"{scripts}.{name}", module)
    monkeypatch.setitem(sys.modules, scripts, parent)

    return chunking, edge_perm


async def _new_path(model: nn.Module, checkpoint: dict[str, Any]) -> nn.Module:
    context = CheckpointContext(model=model, checkpoint_data=checkpoint, config=_run_config())
    await TransferLearningLoader(skip_mismatched=True).process(context)
    return context.model


def _assert_same_model(legacy: nn.Module, pipeline: nn.Module) -> None:
    legacy_state, pipeline_state = legacy.state_dict(), pipeline.state_dict()
    assert sorted(legacy_state) == sorted(pipeline_state)
    for key, tensor in legacy_state.items():
        assert torch.equal(tensor, pipeline_state[key]), key
    assert legacy._ckpt_model_name_to_index == pipeline._ckpt_model_name_to_index
    assert legacy._ckpt_variables_metadata == pipeline._ckpt_variables_metadata


# ---------------------------------------------------------------------------
# Equivalence tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ledgerless_checkpoint_loads_identically_to_the_legacy_path(
    fake_migrations: tuple[MagicMock, MagicMock],
) -> None:
    """A checkpoint with no ledger gives the same result on both paths."""
    chunking, edge_perm = fake_migrations
    checkpoint = _checkpoint()

    legacy = _legacy_transfer_learning_loading(
        _Model(),
        {**checkpoint, "state_dict": dict(checkpoint["state_dict"])},
        "legacy.ckpt",
        chunking,
        edge_perm,
    )
    pipeline = await _new_path(_Model(), {**checkpoint, "state_dict": dict(checkpoint["state_dict"])})

    assert chunking.call_count == 2  # once per path
    assert edge_perm.call_count == 2
    _assert_same_model(legacy, pipeline)


@pytest.mark.asyncio
async def test_shape_mismatch_is_skipped_identically_to_the_legacy_path(
    fake_migrations: tuple[MagicMock, MagicMock],
) -> None:
    """Both paths skip a resized layer and keep its initial values."""
    chunking, edge_perm = fake_migrations
    # The checkpoint head has a different shape from the model head.
    checkpoint = _checkpoint(width=4)
    checkpoint["state_dict"]["head.weight"] = torch.randn(3, 4)
    checkpoint["state_dict"]["head.bias"] = torch.randn(3)

    torch.manual_seed(7)
    legacy_model = _Model()
    torch.manual_seed(7)
    pipeline_model = _Model()

    legacy = _legacy_transfer_learning_loading(
        legacy_model,
        {**checkpoint, "state_dict": dict(checkpoint["state_dict"])},
        "legacy.ckpt",
        chunking,
        edge_perm,
    )
    pipeline = await _new_path(pipeline_model, {**checkpoint, "state_dict": dict(checkpoint["state_dict"])})

    # The head did not transfer in either path.
    assert legacy.head.weight.shape == (2, 4)
    _assert_same_model(legacy, pipeline)


@pytest.mark.asyncio
async def test_pipeline_skips_a_migration_the_legacy_path_reapplied(
    monkeypatch: pytest.MonkeyPatch,
    fake_migrations: tuple[MagicMock, MagicMock],
) -> None:
    """The pipeline does not run ``chunking_fix`` if the ledger records it."""
    chunking, _ = fake_migrations
    spy = MagicMock(side_effect=chunking.side_effect)
    monkeypatch.setattr(
        "anemoi.training.checkpoint.loading.base._load_chunking_fix_migration",
        lambda: spy,
    )

    from anemoi.models.migrations import Migrator
    from anemoi.models.migrations.migrator import MigrationMetadata

    checkpoint = _checkpoint()
    checkpoint["migrations"] = [
        {
            "name": migration.name,
            "metadata": MigrationMetadata(versions={"migration": "1.0.0", "anemoi-models": "0.11.0"}),
            "signature": f"signature-of-{migration.name}",
        }
        for migration in Migrator()._grouped_migrations[-1]
    ]

    await _new_path(_Model(), checkpoint)

    spy.assert_not_called()
