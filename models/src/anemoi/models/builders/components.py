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
from functools import partial
from importlib import import_module
from typing import Any


def resolve_target(target: Any) -> Any:
    """Resolve a target spec into a Python object."""
    if callable(target):
        return target
    if not isinstance(target, str):
        raise TypeError(f"Target must be a dotted import path or callable, got {type(target).__name__}.")

    module_name, _, attr = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Target must be a dotted import path, got '{target}'.")
    module = import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Target '{target}' could not be resolved.") from exc


def _normalize_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(by_alias=True)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"Config must be a mapping or Pydantic model, got {type(config).__name__}.")


def _has_target(config: Mapping[str, Any]) -> bool:
    return any(key in config for key in ("_target_", "target_", "target"))


def _resolve_nested(value: Any, recursive: bool) -> Any:
    if not recursive:
        return value
    if isinstance(value, Mapping):
        if _has_target(value):
            return build_component(value)
        return {key: _resolve_nested(item, recursive) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_nested(item, recursive) for item in value]
    return value


def build_component(config: Any, *, partial_override: bool | None = None, **kwargs: Any) -> Any:
    """Instantiate a model component from a config mapping with a target path."""
    cfg = _normalize_config(config)

    target = cfg.pop("_target_", None)
    if target is None:
        target = cfg.pop("target_", None)
    if target is None:
        target = cfg.pop("target", None)
    if target is None:
        raise ValueError("Component config must define '_target_' (or 'target_').")

    partial_flag = bool(cfg.pop("_partial_", False))
    recursive_flag = cfg.pop("_recursive_", False)
    cfg.pop("_convert_", None)

    cfg = {key: _resolve_nested(value, recursive_flag) for key, value in cfg.items()}
    cfg.update(kwargs)

    component_cls = resolve_target(target)
    if partial_override is not None:
        partial_flag = partial_override

    if partial_flag:
        return partial(component_cls, **cfg)
    return component_cls(**cfg)
