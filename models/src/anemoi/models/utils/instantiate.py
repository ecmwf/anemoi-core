# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Backend-switchable replacement for ``hydra.utils.instantiate``.

The model modules build their children with ``instantiate(config, ...)``. Historically
that was ``hydra.utils.instantiate``, which forces Hydra (and the whole training stack)
to be importable wherever a model is reconstructed -- notably at inference time.

This module provides a drop-in ``instantiate`` with two backends:

* ``"hydra"`` (default): forwards to :func:`hydra.utils.instantiate`. Behaviour is
  unchanged, so training is unaffected.
* ``"native"``: a small pure-Python implementation that recreates objects from a plain
  resolved config (e.g. a ``dict`` / :class:`anemoi.utils.config.DotDict` loaded from
  JSON/YAML) **without importing Hydra**.

The active backend is the "global variable" that distinguishes training from inference
behaviour. Selection priority (highest first):

1. the :func:`instantiation_backend` context manager,
2. an explicit :func:`set_instantiation_backend` call,
3. the ``ANEMOI_INSTANTIATE_BACKEND`` environment variable,
4. the default (``"hydra"``).

See ``no-pickle-plan.md`` at the repository root for the full design rationale.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
from contextlib import contextmanager
from typing import Any
from typing import Iterator

LOGGER = logging.getLogger(__name__)

__all__ = [
    "instantiate",
    "get_object",
    "get_class",
    "InstantiationError",
    "set_instantiation_backend",
    "instantiation_backend",
    "current_backend",
]

# Keys with a special meaning that must not be forwarded as constructor kwargs.
_SPECIAL_KEYS = frozenset({"_target_", "_args_", "_partial_", "_recursive_", "_convert_"})

_VALID_BACKENDS = ("hydra", "native")


class InstantiationError(Exception):
    """Raised when a config cannot be turned into an object.

    Replaces ``hydra.errors.InstantiationException`` on the native path so that callers
    do not need to import Hydra to handle instantiation failures.
    """


# --------------------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------------------


def _default_backend() -> str:
    backend = os.environ.get("ANEMOI_INSTANTIATE_BACKEND", "hydra").strip().lower()
    if backend not in _VALID_BACKENDS:
        LOGGER.warning(
            "Unknown ANEMOI_INSTANTIATE_BACKEND=%r; falling back to 'hydra'. Valid values: %s",
            backend,
            _VALID_BACKENDS,
        )
        backend = "hydra"
    return backend


# Module-level state (the "global variable"). ``None`` in the context stack means
# "use the explicitly-set / env / default backend".
_explicit_backend: str | None = None
_context_backend: str | None = None


def set_instantiation_backend(backend: str | None) -> None:
    """Set the process-wide instantiation backend.

    Parameters
    ----------
    backend : {"hydra", "native"} or None
        The backend to use. ``None`` resets to the environment / default value.
    """
    global _explicit_backend
    if backend is not None:
        backend = backend.strip().lower()
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unknown instantiation backend {backend!r}; valid values: {_VALID_BACKENDS}")
    _explicit_backend = backend


def current_backend() -> str:
    """Return the backend that :func:`instantiate` would use right now."""
    if _context_backend is not None:
        return _context_backend
    if _explicit_backend is not None:
        return _explicit_backend
    return _default_backend()


@contextmanager
def instantiation_backend(backend: str) -> Iterator[None]:
    """Temporarily select the instantiation backend.

    Example
    -------
    >>> with instantiation_backend("native"):
    ...     model = AnemoiModelInterface(config=cfg, ...)
    """
    global _context_backend
    backend = backend.strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown instantiation backend {backend!r}; valid values: {_VALID_BACKENDS}")
    previous = _context_backend
    _context_backend = backend
    try:
        yield
    finally:
        _context_backend = previous


# --------------------------------------------------------------------------------------
# Dotted-path resolution
# --------------------------------------------------------------------------------------


def get_object(path: str) -> Any:
    """Resolve a dotted import path (``"pkg.mod.attr"``) to the object it names.

    Pure-Python replacement for ``hydra.utils.get_object`` / ``get_method``.
    """
    if not isinstance(path, str) or not path:
        raise InstantiationError(f"_target_ must be a non-empty dotted path, got {path!r}")

    # Try progressively shorter module prefixes, resolving the remainder as attributes.
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError as err:
            raise InstantiationError(f"Could not resolve attribute path {path!r}: {err}") from err
        return obj

    raise InstantiationError(f"Could not import any module prefix of {path!r}")


def get_class(path: str) -> type:
    """Resolve a dotted path to a class (parity with ``hydra.utils.get_class``)."""
    obj = get_object(path)
    if not isinstance(obj, type):
        raise InstantiationError(f"{path!r} resolved to {type(obj)!r}, expected a class")
    return obj


# --------------------------------------------------------------------------------------
# Native instantiate
# --------------------------------------------------------------------------------------


def _is_mapping(obj: Any) -> bool:
    # dict, DotDict (dict subclass) and OmegaConf DictConfig all expose items()/keys().
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__") and not isinstance(obj, (str, bytes))


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple)) or type(obj).__name__ == "ListConfig"


def _has_target(obj: Any) -> bool:
    return _is_mapping(obj) and "_target_" in obj


def _to_plain(obj: Any) -> Any:
    """Coerce OmegaConf / DotDict containers to plain dict / list (for ``_convert_``)."""
    if _is_mapping(obj):
        return {k: _to_plain(obj[k]) for k in obj.keys()}
    if _is_sequence(obj):
        return [_to_plain(v) for v in obj]
    return obj


def _convert_value(value: Any, convert: str, recursive: bool) -> Any:
    """Resolve a config value into a constructor argument."""
    if recursive and _has_target(value):
        return _native_instantiate(value, _recursive_=recursive, _convert_=convert)
    if recursive and _is_sequence(value):
        return [_convert_value(v, convert, recursive) for v in value]
    if recursive and _is_mapping(value):
        # A nested config node without a _target_: recurse for any inner _target_ nodes.
        return {k: _convert_value(value[k], convert, recursive) for k in value.keys() if k not in _SPECIAL_KEYS}
    # No recursion (or a leaf): honour _convert_ for container coercion.
    if convert == "none":
        return value
    return _to_plain(value)


def _native_instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """Pure-Python instantiation from a resolved config (no Hydra import)."""
    # Pull the overridable special keywords out of the call-time kwargs.
    override_partial = kwargs.pop("_partial_", None)
    override_recursive = kwargs.pop("_recursive_", None)
    override_convert = kwargs.pop("_convert_", None)

    if config is None:
        return None

    if _is_sequence(config):
        recursive = True if override_recursive is None else override_recursive
        convert = override_convert or "none"
        return [_convert_value(v, convert, recursive) for v in config]

    if not _is_mapping(config):
        # A primitive passed straight through (Hydra returns it unchanged).
        return config

    if "_target_" not in config:
        # A config node without a target: return the (recursively resolved) container.
        recursive = True if override_recursive is None else override_recursive
        convert = override_convert or "none"
        node = _convert_value(config, convert, recursive)
        # Merge any call-time kwargs (rare for node configs, but keep parity).
        if kwargs:
            node = {**node, **kwargs}
        return node

    target_path = config["_target_"]
    partial = bool(config.get("_partial_", False)) if override_partial is None else bool(override_partial)
    recursive = bool(config.get("_recursive_", True)) if override_recursive is None else bool(override_recursive)
    convert = override_convert or config.get("_convert_", "none")

    target = get_object(target_path)

    # Positional arguments: config _args_ first, then call-time *args.
    config_args = config.get("_args_", []) or []
    pos_args = [_convert_value(a, convert, recursive) for a in config_args]
    pos_args.extend(args)

    # Keyword arguments: config params first, call-time kwargs win on conflict.
    params: dict[str, Any] = {}
    for key in config.keys():
        if key in _SPECIAL_KEYS:
            continue
        params[key] = _convert_value(config[key], convert, recursive)
    params.update(kwargs)

    try:
        if partial:
            return functools.partial(target, *pos_args, **params)
        return target(*pos_args, **params)
    except InstantiationError:
        raise
    except Exception as err:  # noqa: BLE001 - re-wrap with the offending target for clarity
        raise InstantiationError(f"Error instantiating {target_path!r}: {err}") from err


# --------------------------------------------------------------------------------------
# Public dispatch
# --------------------------------------------------------------------------------------


def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """Instantiate ``config`` using the active backend.

    Drop-in replacement for ``hydra.utils.instantiate``. With the default ``"hydra"``
    backend this simply forwards to Hydra; with the ``"native"`` backend it uses the
    Hydra-free implementation in this module.
    """
    if current_backend() == "native":
        return _native_instantiate(config, *args, **kwargs)

    try:
        from hydra.utils import instantiate as _hydra_instantiate
    except ImportError as err:  # pragma: no cover - depends on the install
        raise InstantiationError(
            "The 'hydra' instantiation backend is selected but Hydra is not installed. "
            "Install hydra-core, or select the native backend "
            "(set ANEMOI_INSTANTIATE_BACKEND=native or use instantiation_backend('native'))."
        ) from err
    return _hydra_instantiate(config, *args, **kwargs)
