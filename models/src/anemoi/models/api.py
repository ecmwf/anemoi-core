# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch_geometric.data import HeteroData

from anemoi.models.builders import resolve_target
from anemoi.models.config_types import Settings


def normalize_config(config: Any) -> Settings:
    """Normalize config into typed Settings."""
    if isinstance(config, Settings):
        return config
    if hasattr(config, "model_dump"):
        return Settings.model_validate(config.model_dump(by_alias=True))
    if isinstance(config, Mapping):
        return Settings.model_validate(config)
    raise TypeError(f"Config must be a mapping or Pydantic model, got {type(config).__name__}.")


def _extract_model_target(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    model_section = config.get("model")
    if model_section is None:
        raise ValueError("Config must include a 'model' section.")

    model_block = model_section.get("model") if isinstance(model_section, Mapping) else None
    if model_block is None:
        model_block = model_section

    if not isinstance(model_block, Mapping):
        raise TypeError("Config model section must be a mapping.")

    target = model_block.get("_target_") or model_block.get("target_") or model_block.get("target")
    if not target:
        raise ValueError("Model config must define '_target_' (or 'target_').")

    return model_block, target


def _require_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = config.get(key)
    if section is None:
        raise ValueError(f"Config must include a '{key}' section.")
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return section


def build_model(
    config: Settings | Mapping[str, Any] | Any,
    *,
    graph_data: HeteroData | dict[str, HeteroData],
    data_indices: dict,
    statistics: dict,
) -> Any:
    """Build a model instance from a config mapping.

    The config should follow the training schema structure and include:
    - config.model.model._target_ (or config.model._target_ for model-only configs)
    - config.graph and config.training entries used by model classes
    """
    cfg = normalize_config(config)
    _require_section(cfg, "graph")
    _require_section(cfg, "training")
    _, target = _extract_model_target(cfg)
    model_cls = resolve_target(target)

    return model_cls(
        model_config=cfg,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    )


__all__ = ["normalize_config", "build_model"]
