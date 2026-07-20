# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract *parametrisation* passed to every model and graph constructor.

This replaces the Hydra ``instantiate`` mechanism on the construction path.
Modules no longer receive nested config trees; they receive a
:class:`Parametrisation` and read semantic values from it with
:meth:`Parametrisation.get`. Sub-modules that used to be built with
``instantiate`` are now passed in directly (see ``refactor.md``):

    model = MyModel(params, some_layer=MyLayer(params, ...))

``some_layer`` defaults to ``None`` (the module builds its default class), may be
a string (resolved through :meth:`Parametrisation.create_module`, which is where
Hydra can be reattached later), or an already-built instance.

The abstract base lives in ``anemoi.graphs`` because it is the lowest in-repo
layer shared by both ``anemoi.graphs`` and ``anemoi.models``. A
:class:`Parametrisation` must be JSON-serialisable via
:meth:`Parametrisation.to_dict`: in training it is built from the dataset, in
inference from the serialised JSON of a training parametrisation.
"""

from __future__ import annotations

import functools
import importlib
import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

__all__ = [
    "Parametrisation",
    "JSONParametrisation",
    "ParametrisationError",
    "MISSING",
    "build",
    "get_object",
    "get_class",
]

# Sentinel distinguishing "no default given" from ``default=None``.
MISSING: Any = object()

# Keys with a special meaning that must not be forwarded as constructor kwargs.
_SPECIAL_KEYS = frozenset({"_target_", "_args_", "_partial_", "_recursive_", "_convert_"})


class ParametrisationError(Exception):
    """Raised when a parameter is missing or an object cannot be built."""


# --------------------------------------------------------------------------------------
# Dotted-path resolution
# --------------------------------------------------------------------------------------


def get_object(path: str) -> Any:
    """Resolve a dotted import path (``"pkg.mod.attr"``) to the object it names."""
    if not isinstance(path, str) or not path:
        raise ParametrisationError(f"_target_ must be a non-empty dotted path, got {path!r}")

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
            raise ParametrisationError(f"Could not resolve attribute path {path!r}: {err}") from err
        return obj

    raise ParametrisationError(f"Could not import any module prefix of {path!r}")


def get_class(path: str) -> type:
    """Resolve a dotted path to a class."""
    obj = get_object(path)
    if not isinstance(obj, type):
        raise ParametrisationError(f"{path!r} resolved to {type(obj)!r}, expected a class")
    return obj


# --------------------------------------------------------------------------------------
# Object construction from a spec (the Hydra-free ``instantiate``)
# --------------------------------------------------------------------------------------


def _is_mapping(obj: Any) -> bool:
    # dict, DotDict (dict subclass) and OmegaConf DictConfig all expose keys()/__getitem__.
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__") and not isinstance(obj, (str, bytes))


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple)) or type(obj).__name__ == "ListConfig"


def _has_target(obj: Any) -> bool:
    return _is_mapping(obj) and "_target_" in obj


def _to_plain(obj: Any) -> Any:
    """Coerce OmegaConf / DotDict containers to plain dict / list."""
    if _is_mapping(obj):
        return {k: _to_plain(obj[k]) for k in obj.keys()}
    if _is_sequence(obj):
        return [_to_plain(v) for v in obj]
    return obj


def _resolve_value(value: Any, recursive: bool) -> Any:
    """Resolve a spec value into a constructor argument."""
    if recursive and _has_target(value):
        return build(value, _recursive_=recursive)
    if recursive and _is_sequence(value):
        return [_resolve_value(v, recursive) for v in value]
    if recursive and _is_mapping(value):
        return {k: _resolve_value(value[k], recursive) for k in value.keys() if k not in _SPECIAL_KEYS}
    return _to_plain(value)


def build(spec: Any, *args: Any, **kwargs: Any) -> Any:
    """Build an object from a ``spec`` without importing Hydra.

    ``spec`` may be:

    * ``None`` -> ``None``;
    * a dotted-path string -> the named callable is invoked with ``*args, **kwargs``;
    * a mapping with ``_target_`` -> the target is resolved and called, merging spec
      params with call-time kwargs (call-time wins), honouring ``_partial_`` /
      ``_recursive_`` / ``_args_``;
    * a mapping without ``_target_`` -> returned as a (recursively resolved) plain dict;
    * a list/tuple -> element-wise built.
    """
    override_partial = kwargs.pop("_partial_", None)
    override_recursive = kwargs.pop("_recursive_", None)

    if spec is None:
        return None

    if isinstance(spec, str):
        target = get_object(spec)
        partial = bool(override_partial)
        return functools.partial(target, *args, **kwargs) if partial else target(*args, **kwargs)

    if _is_sequence(spec):
        recursive = True if override_recursive is None else override_recursive
        return [_resolve_value(v, recursive) for v in spec]

    if not _is_mapping(spec):
        return spec

    if "_target_" not in spec:
        recursive = True if override_recursive is None else override_recursive
        node = _resolve_value(spec, recursive)
        return {**node, **kwargs} if kwargs else node

    target_path = spec["_target_"]
    partial = bool(spec.get("_partial_", False)) if override_partial is None else bool(override_partial)
    recursive = bool(spec.get("_recursive_", True)) if override_recursive is None else bool(override_recursive)

    target = get_object(target_path)

    pos_args = [_resolve_value(a, recursive) for a in (spec.get("_args_") or [])]
    pos_args.extend(args)

    params: dict[str, Any] = {
        key: _resolve_value(spec[key], recursive) for key in spec.keys() if key not in _SPECIAL_KEYS
    }
    params.update(kwargs)

    try:
        if partial:
            return functools.partial(target, *pos_args, **params)
        return target(*pos_args, **params)
    except ParametrisationError:
        raise
    except Exception as err:  # noqa: BLE001 - re-wrap with the offending target for clarity
        raise ParametrisationError(f"Error building {target_path!r}: {err}") from err


# --------------------------------------------------------------------------------------
# The abstract Parametrisation
# --------------------------------------------------------------------------------------


class Parametrisation(ABC):
    """Abstract parametrisation passed to model and graph constructors."""

    @abstractmethod
    def get(self, key: str, default: Any = MISSING) -> Any:
        """Return the semantic value for ``key``.

        Raise :class:`ParametrisationError` if ``key`` is absent and no ``default``
        is given.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this parametrisation."""

    def create_module(self, spec: Any, *args: Any, **kwargs: Any) -> Any:
        """Build an object from ``spec`` (a dotted-path string or a ``_target_`` mapping).

        This is the single choke point where module construction happens; a Hydra
        backend can be reattached here later without touching call sites.
        """
        return build(spec, *args, **kwargs)

    def resolve(self, value: Any, default_factory, *args: Any, **kwargs: Any) -> Any:
        """Resolve a constructor-injected sub-module argument.

        * ``value is None`` -> ``default_factory()`` (the module's default class);
        * ``value`` is a string -> :meth:`create_module` with the remaining args;
        * otherwise -> ``value`` unchanged (an already-built instance).
        """
        if value is None:
            return default_factory()
        if isinstance(value, str):
            return self.create_module(value, *args, **kwargs)
        return value


class JSONParametrisation(Parametrisation):
    """Concrete :class:`Parametrisation` backed by a JSON-serialisable mapping.

    Keys may be dotted to reach nested values (``params.get("model.num_channels")``).
    This is the parametrisation used both in training (built from the dataset) and in
    inference (rebuilt from the JSON of a serialised training parametrisation).
    """

    def __init__(self, data: Any = None) -> None:
        self._data: dict = dict(data) if data is not None else {}

    @classmethod
    def from_json(cls, text: str) -> "JSONParametrisation":
        """Build from a JSON document (string)."""
        return cls(json.loads(text))

    @classmethod
    def from_file(cls, path: str | Path) -> "JSONParametrisation":
        """Build from a JSON file on disk."""
        return cls.from_json(Path(path).read_text())

    def get(self, key: str, default: Any = MISSING) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if _is_mapping(node) and part in node:
                node = node[part]
            else:
                if default is MISSING:
                    raise ParametrisationError(f"Missing parameter {key!r}")
                return default
        return node

    def to_dict(self) -> dict:
        return _to_plain(self._data)

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON document (string)."""
        return json.dumps(self.to_dict(), **kwargs)
