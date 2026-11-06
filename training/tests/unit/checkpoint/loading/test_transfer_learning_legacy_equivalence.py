# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Old-vs-new equivalence for the checkpoint-format migration and the load it guards.

``tests/unit/checkpoint/loading/test_format_migrations.py`` tests the new migration
logic against its own specification. This file answers the other question: does it
still do what the code it replaced did?

:func:`_legacy_transfer_learning_loading` is the pre-pipeline
``anemoi.training.utils.checkpoint.transfer_learning_loading`` frozen verbatim from
``main`` (minus the ``torch.load``, so both paths get the identical dict). Each test
runs it and the pipeline's ``TransferLearningLoader`` over the same checkpoint and
the same model, then compares the observable results tensor for tensor.

Both paths run ``chunking_fix`` on a checkpoint whose processor declares the
chunking geometry, and both use whatever each migration returns. The fakes below
return new dicts rather than editing in place, so a path that dropped a return
value would load the unmigrated weights and fail the comparison.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# The legacy implementation, frozen.
# ---------------------------------------------------------------------------


def _legacy_filter_state_dict_size_mismatches(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> None:
    """``utils.checkpoint._filter_state_dict_size_mismatches`` from main, verbatim."""
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
    """``utils.checkpoint.transfer_learning_loading`` from main, with the load hoisted out.

    The migrations are passed in rather than imported so a test can hand both paths
    the same fake.
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
# Shared material: one model shape, one checkpoint, two code paths.
# ---------------------------------------------------------------------------


class _Model(nn.Module):
    """A model with the legacy processor-refresh hook, as a no-op.

    The hook is the one legacy step with no counterpart to compare against: the
    pipeline does that work in ``refresh_checkpoint_processors``, driven by
    ``training.update_ds_stats_on_ckpt_load``, which the config below leaves off.
    Neutralising it on both sides keeps the comparison on the migration, filter and
    metadata algorithm.
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
    """A checkpoint whose processor declares the geometry ``chunking_fix`` rewrites."""
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
    """A run config with the processor refresh off, matching the no-op legacy hook."""
    return OmegaConf.create({"training": {"update_ds_stats_on_ckpt_load": {"states": False, "tendencies": False}}})


@pytest.fixture
def fake_migrations(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install fake chunking_fix / trainable_edge_perm_fix migrations with observable effects.

    Each rewrites one tensor in the state dict, so "the migration ran" is visible in
    the loaded weights rather than only in a call count. The same two callables are
    handed to the legacy reference, so any difference in result is a difference in
    the surrounding algorithm, not in the migration.
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

    monkeypatch.setattr(loading_base, "_chunking_fix", chunking)
    monkeypatch.setattr(loading_base, "_trainable_edge_perm_fix", edge_perm)

    return chunking, edge_perm


def _new_path(model: nn.Module, checkpoint: dict[str, Any]) -> nn.Module:
    context = CheckpointContext(model=model, checkpoint_data=checkpoint, config=_run_config())
    TransferLearningLoader(skip_mismatched=True).process(context)
    return context.model


def _assert_same_model(legacy: nn.Module, pipeline: nn.Module) -> None:
    legacy_state, pipeline_state = legacy.state_dict(), pipeline.state_dict()
    assert sorted(legacy_state) == sorted(pipeline_state)
    for key, tensor in legacy_state.items():
        assert torch.equal(tensor, pipeline_state[key]), key
    assert legacy._ckpt_model_name_to_index == pipeline._ckpt_model_name_to_index
    assert legacy._ckpt_variables_metadata == pipeline._ckpt_variables_metadata


# ---------------------------------------------------------------------------
# Equivalence.
# ---------------------------------------------------------------------------


def test_migrated_checkpoint_loads_identically_to_the_legacy_path(
    fake_migrations: tuple[MagicMock, MagicMock],
) -> None:
    """Both paths migrate the checkpoint and reach the same weights.

    Both run both migrations, and the loaded tensors, the restored name-to-index
    map and the extracted variables metadata all match.
    """
    chunking, edge_perm = fake_migrations
    checkpoint = _checkpoint()

    legacy = _legacy_transfer_learning_loading(
        _Model(),
        {**checkpoint, "state_dict": dict(checkpoint["state_dict"])},
        "legacy.ckpt",
        chunking,
        edge_perm,
    )
    pipeline = _new_path(_Model(), {**checkpoint, "state_dict": dict(checkpoint["state_dict"])})

    assert chunking.call_count == 2  # once per path
    assert edge_perm.call_count == 2
    _assert_same_model(legacy, pipeline)


def test_shape_mismatch_is_skipped_identically_to_the_legacy_path(
    fake_migrations: tuple[MagicMock, MagicMock],
) -> None:
    """A resized layer is skipped by both paths, leaving the model's own init in place.

    Legacy deleted the mismatched key from the state dict in place and loaded with
    ``strict=False``; the pipeline's ``filter_state_dict`` builds a new dict without
    it. Same outcome, including which parameters keep their initialised values.
    """
    chunking, edge_perm = fake_migrations
    # A checkpoint trained at width 4, loaded into a width-4 model with a resized head.
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
    pipeline = _new_path(pipeline_model, {**checkpoint, "state_dict": dict(checkpoint["state_dict"])})

    # The head did not transfer in either path.
    assert legacy.head.weight.shape == (2, 4)
    _assert_same_model(legacy, pipeline)
