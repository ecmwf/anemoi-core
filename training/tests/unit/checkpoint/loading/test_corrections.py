# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``apply_checkpoint_corrections``: one function, one order.

The load-time corrections (``chunking_fix``, the trainable-edge-permutation
migration, the processor-statistics refresh) used to be three calls repeated per
loading strategy, in an order that differed from the Lightning hook's. They are
now one function whose order is pinned here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading import base as loading_base
from anemoi.training.checkpoint.loading import strategies
from anemoi.training.checkpoint.loading.base import apply_checkpoint_corrections
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

if TYPE_CHECKING:
    import pytest
    from omegaconf import DictConfig


class _Inner(nn.Module):
    """The AnemoiModelInterface-equivalent: processors live here, keyed without ``model.``."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_processors = nn.Linear(2, 2)
        self.body = nn.Linear(2, 2)


class _Outer(nn.Module):
    """LightningModule-shaped wrapper whose ``.model`` is the inner interface."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


class _OuterWithoutProcessors(nn.Module):
    """A wrapper whose inner model has no processors to re-inject from."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)


def _stale_checkpoint(model: nn.Module) -> dict[str, Any]:
    """A checkpoint whose every entry is obviously stale (all 99.0)."""
    return {"state_dict": {key: torch.full_like(value, 99.0) for key, value in model.state_dict().items()}}


def _config(*, states: bool) -> DictConfig:
    return OmegaConf.create(
        {"training": {"update_ds_stats_on_ckpt_load": {"states": states, "tendencies": False}}},
    )


def test_edge_perm_migration_runs_before_the_processor_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The order is edge-perm, then refresh, and it is load-bearing.

    The trainable-edge-permutation migration takes the live model and may rebuild
    the state dict. Stand in for that with a migration that returns a new dict
    without the processor entries: if the refresh ran first, the buffers it injected
    from the live model would be discarded by the rebuild; run after, it puts them
    back.
    """
    model = _Outer()
    checkpoint = _stale_checkpoint(model)

    def rebuilding_migrate(ckpt: dict[str, Any], _model: nn.Module) -> dict[str, Any]:
        return {"state_dict": {k: v for k, v in ckpt["state_dict"].items() if "pre_processors" not in k}}

    monkeypatch.setattr(loading_base, "_trainable_edge_perm_fix", rebuilding_migrate)

    corrected = apply_checkpoint_corrections(checkpoint, model, _config(states=True))

    state_dict = corrected["state_dict"]
    live = model.model.state_dict()
    assert torch.equal(state_dict["model.pre_processors.weight"], live["pre_processors.weight"])
    assert torch.equal(state_dict["model.pre_processors.bias"], live["pre_processors.bias"])
    # Entries outside the refresh are the checkpoint's own.
    assert torch.all(state_dict["model.body.weight"] == 99.0)


def test_chunking_fix_runs_first_and_edge_perm_sees_its_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """``chunking_fix`` may return a new dict; every later step works on that one."""
    model = _Outer()
    original = _stale_checkpoint(model)
    original["hyper_parameters"] = {
        "config": OmegaConf.create({"model": {"processor": {"num_layers": 2, "num_chunks": 1}}}),
    }
    replacement = {"state_dict": dict(original["state_dict"]), "_migrated": True}
    seen_by_edge_perm: list[dict[str, Any]] = []

    monkeypatch.setattr(loading_base, "_chunking_fix", lambda _ckpt: replacement)

    def recording_migrate(ckpt: dict[str, Any], _model: nn.Module) -> dict[str, Any]:
        seen_by_edge_perm.append(ckpt)
        return ckpt

    monkeypatch.setattr(loading_base, "_trainable_edge_perm_fix", recording_migrate)

    corrected = apply_checkpoint_corrections(original, model, _config(states=True))

    assert seen_by_edge_perm == [replacement]
    assert corrected is replacement
    # The refresh worked on the replacement too.
    assert torch.equal(
        corrected["state_dict"]["model.pre_processors.weight"],
        model.model.state_dict()["pre_processors.weight"],
    )


def test_none_checkpoint_is_returned_untouched() -> None:
    assert apply_checkpoint_corrections(None, _Outer(), _config(states=True)) is None


def test_missing_config_disables_the_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without config, the refresh step still runs but leaves the processors alone."""
    model = _Outer()
    checkpoint = _stale_checkpoint(model)
    refresh_flags: list[dict[str, bool]] = []
    refresh = loading_base.refresh_checkpoint_processors

    def recording_refresh(
        checkpoint: dict[str, Any],
        inner_model: nn.Module,
        **flags: bool,
    ) -> None:
        refresh_flags.append(flags)
        refresh(checkpoint, inner_model, **flags)

    monkeypatch.setattr(loading_base, "refresh_checkpoint_processors", recording_refresh)

    corrected = apply_checkpoint_corrections(checkpoint, model, None)

    assert refresh_flags == [{"update_states": False, "update_tendencies": False, "update_residuals": False}]
    assert torch.all(corrected["state_dict"]["model.pre_processors.weight"] == 99.0)


def test_refresh_without_an_inner_model_keeps_the_processor_entries(caplog: pytest.LogCaptureFixture) -> None:
    """Dropping without re-injecting would leave a strict load missing keys, so nothing is dropped.

    The refresh reads the inner ``.model``; a module that has none (a bare model handed to
    a strategy directly) cannot supply replacements, and the checkpoint's own entries are
    kept rather than silently removed.
    """
    model = nn.Linear(2, 2)
    checkpoint = {"state_dict": {"model.pre_processors.weight": torch.full((2, 2), 99.0), "model.body": torch.ones(1)}}

    with caplog.at_level(logging.WARNING):
        corrected = apply_checkpoint_corrections(checkpoint, model, _config(states=True))

    assert torch.all(corrected["state_dict"]["model.pre_processors.weight"] == 99.0)
    assert any("no model" in record.getMessage() for record in caplog.records)


def test_refresh_that_drops_but_cannot_reinject_is_logged_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A model without the checkpoint's processors drops them, and says so out loud.

    That case is legitimate (a checkpoint with tendency processors loaded into a
    model without them), so it is not an error, but it must not hide at debug level.
    """
    model = _OuterWithoutProcessors()
    checkpoint = {
        "state_dict": {"model.pre_processors.weight": torch.full((2, 2), 99.0), "model.weight": torch.ones(2, 2)},
    }

    with caplog.at_level(logging.WARNING):
        corrected = apply_checkpoint_corrections(checkpoint, model, _config(states=True))

    assert "model.pre_processors.weight" not in corrected["state_dict"]
    assert any("none to re-inject" in record.getMessage() for record in caplog.records)


# --- the order of the shared steps inside the concrete loaders ---


class _MiniModel(nn.Module):
    """Single-layer model whose state dict round-trips through a strict load."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)


def _install_order_recorders(model: nn.Module, calls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the corrections, the metadata steps and the model load with order recorders.

    Both the corrections and the metadata steps are module-level functions, patched
    where each is looked up: the corrections in ``loading.base`` (``apply_checkpoint_corrections``
    calls them by name), the metadata steps in ``loading.strategies``. The real
    ``_extract_state_dict`` and ``_mark_weights_loaded`` stay live so the strategy still
    performs an actual (recorded) ``load_state_dict`` call.
    """
    corrections = {
        "_chunking_fix": "format",
        "_trainable_edge_perm_fix": "edge_perm",
        "refresh_checkpoint_processors": "refresh",
    }
    # Every checkpoint counts as a chunking_fix candidate, so the recorder is reached.
    monkeypatch.setattr(loading_base, "_chunking_fix_applicable", lambda _checkpoint: True)
    for function_name, label in corrections.items():

        def _correction_recorder(checkpoint: object, *_args: object, _label: str = label, **_kwargs: object) -> object:
            calls.append(_label)
            return checkpoint

        monkeypatch.setattr(loading_base, function_name, _correction_recorder)

    metadata_steps = {
        "warn_on_hparams_divergence": "warn",
        "preserve_anemoi_metadata": "preserve",
        "extract_checkpoint_variables_metadata": "extract",
    }
    for function_name, label in metadata_steps.items():

        def _recorder(*_args: object, _label: str = label, **_kwargs: object) -> None:
            calls.append(_label)

        monkeypatch.setattr(strategies, function_name, _recorder)

    original_load = model.load_state_dict

    def _load_recorder(*args: object, **kwargs: object) -> object:
        calls.append("load")
        return original_load(*args, **kwargs)

    model.load_state_dict = _load_recorder


def test_weights_only_runs_the_corrections_then_the_metadata_steps_around_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WeightsOnlyLoader runs the corrections in their one order, loads, then restores metadata.

    The order inside the corrections is chunking_fix, edge-perm, refresh: the order
    ``on_load_checkpoint`` has always used for ``ckpt_path`` loads, and the safe one,
    since edge-perm takes the live model and may rebuild the dict.
    """
    model = _MiniModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": model.state_dict()})
    calls: list[str] = []
    _install_order_recorders(model, calls, monkeypatch)

    WeightsOnlyLoader().process(context)

    assert calls == ["format", "edge_perm", "refresh", "load", "warn", "preserve", "extract"]
