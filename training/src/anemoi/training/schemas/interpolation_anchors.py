# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Identify and prune OmegaConf interpolation anchors not declared by the schema.

Background
----------
Hydra configs often use interpolation anchors as convenient aliases, e.g.:

    layer_kernels: &lk
      Linear: torch.nn.Linear
    encoder:
      layer_kernels: ${layer_kernels}   # reference to the anchor above

After interpolation is resolved, the top-level ``layer_kernels`` key is no longer
needed, and if the schema doesn't declare it, keeping it just makes comparison between
strict and lenient mode confusing. This module provides the tooling to identify and remove
such undeclared anchors safely.

Public API:
    undeclared_interpolation_anchor_paths(config, schema)
        -- identify which anchor paths are safe to prune
    prune_undeclared_interpolation_anchors(config, paths)
        -- delete them from the config
    resolve_and_prune_undeclared_interpolation_anchors(config, schema)
        -- convenience wrapper that does both in the correct order
"""

from __future__ import annotations

from functools import lru_cache
import re
from types import UnionType
from typing import Any
from typing import Literal
from typing import Union
from typing import get_args
from typing import get_origin

from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from pydantic import BaseModel as PydanticBaseModel

from .schema_defaults import _unwrap_annotated

_INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]+)\}")
# Used in schema path patterns to represent an arbitrary dict key.
_WILDCARD_PATH_SEGMENT = "*"


# ---------------------------------------------------------------------------
# Step 1 — collect interpolation references from the config
# ---------------------------------------------------------------------------

def _extract_plain_interpolation_paths(value: Any) -> set[tuple[str, ...]]:
    """Parse ``${a.b.c}`` references out of a single string value.

    Resolver expressions such as ``${oc.env:HOME}`` (containing ``:``) are
    ignored — only plain key-path references are relevant for anchor pruning.
    """
    if not isinstance(value, str):
        return set()
    refs: set[tuple[str, ...]] = set()
    for match in _INTERPOLATION_PATTERN.findall(value):
        if ":" not in match:
            if parts := tuple(p for p in match.split(".") if p):
                refs.add(parts)
    return refs


def _collect_interpolation_paths(config: DictConfig | ListConfig) -> set[tuple[str, ...]]:
    """Walk the config tree and collect all plain interpolation source paths.

    For each key/index that is itself an interpolation, extract the referenced
    paths. For nested DictConfig/ListConfig values, recurse.

    We read the raw (unresolved) string from ``OmegaConf.to_container`` to avoid
    triggering resolution errors for unresolved env-var interpolations.
    """
    refs: set[tuple[str, ...]] = set()
    raw = OmegaConf.to_container(config, resolve=False)
    if isinstance(config, DictConfig) and isinstance(raw, dict):
        keys_and_raw = ((k, raw.get(k)) for k in config.keys())
    elif isinstance(config, ListConfig) and isinstance(raw, list):
        keys_and_raw = enumerate(raw)
    else:
        return refs
    for key, raw_value in keys_and_raw:
        if OmegaConf.is_interpolation(config, key):
            refs.update(_extract_plain_interpolation_paths(raw_value))
        else:
            try:
                child = config[key]
            except Exception:
                continue
            if isinstance(child, DictConfig | ListConfig):
                refs.update(_collect_interpolation_paths(child))
    return refs


# ---------------------------------------------------------------------------
# Step 2 — build the set of paths declared by the schema
# ---------------------------------------------------------------------------

def _dict_key_patterns(key_annotation: Any) -> list[str]:
    """Return the concrete key segments (or ``*``) declared for a dict key type.

    ``Literal["train", "val"]`` → ``["train", "val"]``
    Any other type           → ``["*"]``  (wildcard — any key is valid)
    """
    key_annotation, _ = _unwrap_annotated(key_annotation)
    if get_origin(key_annotation) is Literal:
        return [str(v) for v in get_args(key_annotation)] or [_WILDCARD_PATH_SEGMENT]
    return [_WILDCARD_PATH_SEGMENT]


def _collect_annotation_patterns(
    annotation: Any,
    prefix: tuple[str, ...],
    seen_models: tuple[type[PydanticBaseModel], ...],
) -> set[tuple[str, ...]]:
    """Collect all schema-declared path patterns reachable from *annotation*.

    Recursively expands:
    - Model types → their field paths (via ``_collect_schema_path_patterns``).
    - Unions → union of patterns from all branches.
    - list/tuple/set → element patterns with a ``*`` appended to the prefix.
    - dict → value patterns with each key segment (or ``*``) appended to the prefix.
    """
    annotation, _ = _unwrap_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return _collect_schema_path_patterns(annotation, prefix, seen_models)
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return {p for arg in get_args(annotation) for p in _collect_annotation_patterns(arg, prefix, seen_models)}
    if origin in (list, tuple, set):
        args = get_args(annotation)
        return _collect_annotation_patterns(args[0] if args else Any, (*prefix, _WILDCARD_PATH_SEGMENT), seen_models)
    if origin is dict:
        args = get_args(annotation)
        if len(args) < 2:
            return set()
        return {
            p
            for seg in _dict_key_patterns(args[0])
            for p in _collect_annotation_patterns(args[1], (*prefix, seg), seen_models)
        }
    return set()


def _collect_schema_path_patterns(
    model: type[PydanticBaseModel],
    prefix: tuple[str, ...] = (),
    seen_models: tuple[type[PydanticBaseModel], ...] = (),
) -> set[tuple[str, ...]]:
    """Return the set of all config paths declared by *model* (rooted at *prefix*).

    Each field contributes its own path and recursively all paths reachable through
    its annotation. ``seen_models`` guards against infinite recursion in self-referential
    schemas (e.g. a model that contains a list of itself).
    """
    if model in seen_models:
        return set()
    next_seen = (*seen_models, model)
    patterns: set[tuple[str, ...]] = set()
    for field_name, field_info in model.model_fields.items():
        field_path = (*prefix, field_name)
        patterns.add(field_path)
        patterns.update(_collect_annotation_patterns(field_info.annotation, field_path, next_seen))
    return patterns


@lru_cache(maxsize=None)
def _schema_path_patterns(schema: type[PydanticBaseModel]) -> tuple[tuple[str, ...], ...]:
    """Cached, sorted tuple of all paths declared by *schema*.

    Sorted so that ``_has_schema_coverage`` can rely on a stable order, and cached
    because the schema structure never changes at runtime.
    """
    return tuple(sorted(_collect_schema_path_patterns(schema)))


# ---------------------------------------------------------------------------
# Step 3 — decide which paths are safe to prune
# ---------------------------------------------------------------------------

def _has_schema_coverage(path: tuple[str, ...], schema_patterns: tuple[tuple[str, ...], ...]) -> bool:
    """Return True if the schema declares *path* itself or any descendant of it.

    A pattern *covers* a path when:
    - It is at least as long as the path, and
    - Each segment of the pattern matches the corresponding segment of the path
      (either exactly or via the ``*`` wildcard).

    If any pattern covers the path, the key at that path (or a sub-key) may be
    needed by the schema and should not be pruned.
    """
    return any(
        len(pattern) >= len(path)
        and all(p in (_WILDCARD_PATH_SEGMENT, s) for p, s in zip(pattern, path))
        for pattern in schema_patterns
    )


def _normalize_prunable_paths(paths: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Deduplicate and keep only the topmost paths, dropping any covered descendants.

    E.g. ``[("a",), ("a", "b"), ("x", "y")]`` → ``[("a",), ("x", "y")]``.
    Deleting ``("a",)`` already removes everything under it, so ``("a", "b")`` is redundant.
    """
    minimized: list[tuple[str, ...]] = []
    for path in sorted(set(paths), key=lambda p: (len(p), p)):
        if not any(path[: len(parent)] == parent for parent in minimized):
            minimized.append(path)
    return minimized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def undeclared_interpolation_anchor_paths(
    config: DictConfig,
    schema: type[PydanticBaseModel],
) -> list[tuple[str, ...]]:
    """Return the config paths that are interpolation anchors but undeclared by the schema.

    A path is considered prunable when all of the following hold:
    - It is referenced by at least one plain ``${...}`` interpolation in the config.
    - Neither the path itself nor any of its descendants is declared by *schema*.
    - The key actually exists in *config* right now.

    The result is normalized so that only the topmost paths are returned — deleting
    a parent automatically covers its children.
    """
    if not isinstance(config, DictConfig):
        return []

    schema_patterns = _schema_path_patterns(schema)
    undeclared_paths: list[tuple[str, ...]] = []
    for path in sorted(_collect_interpolation_paths(config)):
        if not path or _has_schema_coverage(path, schema_patterns):
            continue
        # Verify the key actually exists before scheduling it for deletion.
        if len(path) == 1:
            path_exists = path[0] in config
        else:
            parent = OmegaConf.select(config, ".".join(path[:-1]), default=None)
            path_exists = isinstance(parent, DictConfig | dict) and path[-1] in parent
        if path_exists:
            undeclared_paths.append(path)
    return _normalize_prunable_paths(undeclared_paths)


def prune_undeclared_interpolation_anchors(config: DictConfig, paths: list[tuple[str, ...]]) -> None:
    """Delete the given anchor paths from *config* in-place.

    Paths are normalized before deletion so that removing a parent key does not
    cause a KeyError when a child of the same parent also appears in *paths*.
    """
    for path in _normalize_prunable_paths(paths):
        if not path:
            continue
        parent = config if len(path) == 1 else OmegaConf.select(config, ".".join(path[:-1]), default=None)
        if isinstance(parent, DictConfig):
            with open_dict(parent):
                parent.pop(path[-1], None)
        elif isinstance(parent, dict):
            parent.pop(path[-1], None)


def resolve_and_prune_undeclared_interpolation_anchors(
    config: DictConfig,
    schema: type[PydanticBaseModel],
) -> None:
    """Resolve all interpolations in *config*, then prune undeclared anchor keys.

    The ordering is critical:
    1. Collect undeclared anchor paths **before** resolving — the interpolations
       must still be present to identify which keys are anchors.
    2. Resolve interpolations — anchor values are copied to their reference sites.
    3. Prune the anchor keys — they are no longer needed and would fail schema validation.
    """
    paths = undeclared_interpolation_anchor_paths(config, schema)
    OmegaConf.resolve(config)
    prune_undeclared_interpolation_anchors(config, paths)
