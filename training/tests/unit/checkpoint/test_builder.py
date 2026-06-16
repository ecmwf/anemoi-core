# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``build_checkpoint_pipeline`` (config wiring → pipeline)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import anemoi.training
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline

_FREEZING_TARGET = "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage"


def _checkpoint_template_dir() -> Path:
    """Absolute path to the shipped ``training/checkpoint`` config group."""
    return Path(anemoi.training.__file__).parent / "config" / "training" / "checkpoint"


def _load_template(group: str, name: str) -> DictConfig:
    """Load a shipped checkpoint template (``source/*`` or ``loading/*``)."""
    return OmegaConf.load(_checkpoint_template_dir() / group / f"{name}.yaml")


def _freezing_modifier(submodules: list[str] | None = None) -> dict:
    """Inline FreezingModifierStage config.

    The freezing template ships with the modifier layer (PR #442), not with this
    layer, so the modifier config is constructed inline for the guarded tests.
    """
    return {"_target_": _FREEZING_TARGET, "submodules_to_freeze": submodules or []}


def compose_test_config(
    *,
    source: str | None = None,
    loading: str | None = None,
    modifiers: list | None = None,
) -> DictConfig:
    """Build a minimal training config that exercises the builder's namespaces.

    ``modifiers`` accepts either template names (currently only ``"freezing"``)
    or pre-built modifier config dicts, allowing order-sensitive assertions.
    """
    checkpoint: dict = {}
    if source is not None:
        checkpoint["source"] = _load_template("source", source)
    if loading is not None:
        checkpoint["loading"] = _load_template("loading", loading)
    if modifiers:
        checkpoint["modifiers"] = [m if isinstance(m, dict) else _freezing_modifier() for m in modifiers]

    training: dict = {"checkpoint": checkpoint} if checkpoint else {}
    return OmegaConf.create({"training": training})


def test_builder_orders_stages_source_loader_modifiers() -> None:
    pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing", reason="PR #442")
    cfg = compose_test_config(
        source="local",
        loading="weights_only",
        modifiers=["freezing", "freezing"],
    )
    pipeline = build_checkpoint_pipeline(cfg)
    names = [type(s).__name__ for s in pipeline.stages]
    assert names[0].endswith("Source")
    assert names[1].endswith("Loader")
    assert all(n.endswith("Stage") for n in names[2:])


def test_builder_orders_source_loader_without_modifiers() -> None:
    cfg = compose_test_config(source="local", loading="weights_only", modifiers=[])
    pipeline = build_checkpoint_pipeline(cfg)
    names = [type(s).__name__ for s in pipeline.stages]
    assert names[0].endswith("Source")
    assert names[1].endswith("Loader")
    assert len(names) == 2


def test_hydra_defaults_compose() -> None:
    """Every shipped template under training/checkpoint/* composes without error."""
    templates = sorted(_checkpoint_template_dir().rglob("*.yaml"))
    assert templates, f"no checkpoint templates found under {_checkpoint_template_dir()}"
    for template in templates:
        cfg = OmegaConf.load(template)
        # Instantiation (not merely loading) is what catches an unsupported kwarg,
        # e.g. a `strict` key on a loader that does not accept one (-> TypeError).
        instantiate(cfg)


def test_modifiers_list_order_preserved() -> None:
    """D11: list order == execution order."""
    pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing", reason="PR #442")
    cfg = compose_test_config(
        source="local",
        loading="weights_only",
        modifiers=[_freezing_modifier(["encoder"]), _freezing_modifier(["decoder"])],
    )
    pipeline = build_checkpoint_pipeline(cfg)
    frozen = [stage.submodules_to_freeze for stage in pipeline.stages[2:]]
    assert frozen == [["encoder"], ["decoder"]]


def test_no_checkpoint_config_builds_empty_pipeline() -> None:
    """Absent training.checkpoint => no pipeline; trainer behaves as today."""
    cfg = OmegaConf.create({"training": {}})
    pipeline = build_checkpoint_pipeline(cfg)
    assert len(pipeline.stages) == 0
