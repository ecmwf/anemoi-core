# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``apply_checkpoint_corrections`` and the order of its steps.

Order: format migrations, edge-permutation migration, processor refresh.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections

if TYPE_CHECKING:
    import pytest
    from omegaconf import DictConfig


class _Inner(nn.Module):
    """Inner model (like AnemoiModelInterface). Holds the processors."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_processors = nn.Linear(2, 2)
        self.body = nn.Linear(2, 2)


class _Outer(nn.Module):
    """LightningModule-like wrapper around ``_Inner``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


class _OuterWithoutProcessors(nn.Module):
    """Wrapper whose inner model has no processors."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)


def _stale_checkpoint(model: nn.Module) -> dict[str, Any]:
    """Return a checkpoint with all values set to 99.0."""
    return {"state_dict": {key: torch.full_like(value, 99.0) for key, value in model.state_dict().items()}}


def _config(*, states: bool) -> DictConfig:
    return OmegaConf.create(
        {"training": {"update_ds_stats_on_ckpt_load": {"states": states, "tendencies": False}}},
    )


def test_edge_perm_migration_runs_before_the_processor_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The processor refresh runs after the edge-permutation migration.

    The migration can rebuild the state dict. If the refresh ran first, the rebuild would remove its entries.
    """
    model = _Outer()
    checkpoint = _stale_checkpoint(model)

    def rebuilding_migrate(ckpt: dict[str, Any], _model: nn.Module) -> dict[str, Any]:
        return {"state_dict": {k: v for k, v in ckpt["state_dict"].items() if "pre_processors" not in k}}

    monkeypatch.setattr(loading_base, "_load_trainable_edge_perm_migration", lambda: rebuilding_migrate)

    corrected = apply_checkpoint_corrections(checkpoint, model, _config(states=True))

    state_dict = corrected["state_dict"]
    live = model.model.state_dict()
    assert torch.equal(state_dict["model.pre_processors.weight"], live["pre_processors.weight"])
    assert torch.equal(state_dict["model.pre_processors.bias"], live["pre_processors.bias"])
    # Other entries come from the checkpoint.
    assert torch.all(state_dict["model.body.weight"] == 99.0)


def test_format_migrations_run_first_and_edge_perm_sees_their_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Later steps use the dict that the format migrations return."""
    model = _Outer()
    original = _stale_checkpoint(model)
    replacement = {"state_dict": dict(original["state_dict"]), "_migrated": True}
    seen_by_edge_perm: list[dict[str, Any]] = []

    monkeypatch.setattr(loading_base, "apply_checkpoint_format_migrations", lambda _ckpt, _path: replacement)

    def recording_migrate(ckpt: dict[str, Any], _model: nn.Module) -> dict[str, Any]:
        seen_by_edge_perm.append(ckpt)
        return ckpt

    monkeypatch.setattr(loading_base, "_load_trainable_edge_perm_migration", lambda: recording_migrate)

    corrected = apply_checkpoint_corrections(original, model, _config(states=True))

    assert seen_by_edge_perm == [replacement]
    assert corrected is replacement
    assert torch.equal(
        corrected["state_dict"]["model.pre_processors.weight"],
        model.model.state_dict()["pre_processors.weight"],
    )


def test_none_checkpoint_is_returned_untouched() -> None:
    assert apply_checkpoint_corrections(None, _Outer(), _config(states=True)) is None


def test_missing_config_disables_the_refresh() -> None:
    """Without a config, the processors are not refreshed."""
    model = _Outer()
    checkpoint = _stale_checkpoint(model)

    corrected = apply_checkpoint_corrections(checkpoint, model, None)

    assert torch.all(corrected["state_dict"]["model.pre_processors.weight"] == 99.0)


def test_refresh_without_an_inner_model_keeps_the_processor_entries(caplog: pytest.LogCaptureFixture) -> None:
    """Without an inner ``.model``, the refresh keeps the checkpoint processor entries.

    Removing them would make a strict load fail with missing keys.
    """
    model = nn.Linear(2, 2)
    checkpoint = {"state_dict": {"model.pre_processors.weight": torch.full((2, 2), 99.0), "model.body": torch.ones(1)}}

    with caplog.at_level(logging.WARNING):
        corrected = apply_checkpoint_corrections(checkpoint, model, _config(states=True))

    assert torch.all(corrected["state_dict"]["model.pre_processors.weight"] == 99.0)
    assert any("no model" in record.getMessage() for record in caplog.records)


def test_refresh_that_drops_but_cannot_reinject_is_logged_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """If the model has no matching processors, the refresh removes them and logs a WARNING."""
    model = _OuterWithoutProcessors()
    checkpoint = {
        "state_dict": {"model.pre_processors.weight": torch.full((2, 2), 99.0), "model.weight": torch.ones(2, 2)},
    }

    with caplog.at_level(logging.WARNING):
        corrected = apply_checkpoint_corrections(checkpoint, model, _config(states=True))

    assert "model.pre_processors.weight" not in corrected["state_dict"]
    assert any("none to re-inject" in record.getMessage() for record in caplog.records)
