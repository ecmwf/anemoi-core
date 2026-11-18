# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``apply_checkpoint_corrections``: one function, one order.

The load-time corrections (``chunking_fix``, then the trainable-edge-permutation
migration) are one function shared by the loading strategies and the Lightning
hook; their order is pinned here.
"""

from __future__ import annotations

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


class _Inner(nn.Module):
    """The AnemoiModelInterface-equivalent, keyed without ``model.``."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_processors = nn.Linear(2, 2)
        self.body = nn.Linear(2, 2)


class _Outer(nn.Module):
    """LightningModule-shaped wrapper whose ``.model`` is the inner interface."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


def _stale_checkpoint(model: nn.Module) -> dict[str, Any]:
    """A checkpoint whose every entry is obviously stale (all 99.0)."""
    return {"state_dict": {key: torch.full_like(value, 99.0) for key, value in model.state_dict().items()}}


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

    corrected = apply_checkpoint_corrections(original, model)

    assert seen_by_edge_perm == [replacement]
    assert corrected is replacement


def test_none_checkpoint_is_returned_untouched() -> None:
    assert apply_checkpoint_corrections(None, _Outer()) is None


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

    The order inside the corrections is chunking_fix, then edge-perm: the order
    ``on_load_checkpoint`` uses for ``ckpt_path`` loads.
    """
    model = _MiniModel()
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": model.state_dict()})
    calls: list[str] = []
    _install_order_recorders(model, calls, monkeypatch)

    WeightsOnlyLoader().process(context)

    assert calls == ["format", "edge_perm", "load", "warn", "preserve", "extract"]
