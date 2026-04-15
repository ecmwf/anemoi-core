# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Schema utilities: default injection and interpolation anchor pruning."""

from __future__ import annotations

import re
from functools import cache
from types import UnionType
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from omegaconf.errors import OmegaConfBaseException
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_serializer
from pydantic_core import PydanticUndefined

T = TypeVar("T")
DatasetDict: TypeAlias = dict[Literal["datasets"], dict[str, T]]


class NullDropSchema(PydanticBaseModel):
    """Mixin for schemas whose output is unpacked with ``**`` into a function call.

    Some functions (e.g. ``open_dataset``) cannot handle explicit ``None`` keyword
    arguments — ``open_dataset(select=None)`` passes ``None`` as a selection rather
    than meaning "no selection".  Schemas that are forwarded this way must omit
    ``None``-valued fields entirely from their output.

    Inheriting from this mixin enforces that guarantee on both execution paths:

    * **Strict path** (pydantic validation): the ``_exclude_none`` serializer
      removes ``None``-valued entries whenever pydantic serialises this model,
      including when it appears as a nested field inside a parent ``model_dump``.
    * **Lenient path** (``apply_schema_defaults``): ``schema_defaults`` detects
      ``NullDropSchema`` subclasses via ``issubclass`` and skips any field whose
      resolved value is ``None`` — no extra class variable needed.

    Only inherit from this mixin when the schema output is passed as ``**kwargs``
    to a function that cannot accept ``None``-valued arguments.  Most schemas
    should *not* use this — ``None`` is a valid and meaningful value in the
    majority of schema fields (e.g. ``end=None`` means "no end date").
    """

    @model_serializer(mode="wrap")
    def _exclude_none(self, handler: Any) -> dict:
        return {k: v for k, v in handler(self).items() if v is not None}


class _NoDefault:
    """Sentinel for 'field has no default', distinct from a default value of ``None``."""

    def __repr__(self) -> str:
        return "<no default>"


_NO_DEFAULT = _NoDefault()
_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------
# Annotation helpers (shared by both halves)
# ---------------------------------------------------------------------------


def _unwrap_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Strip ``Annotated`` wrappers and return ``(bare_type, metadata)``."""
    metadata: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation, metadata = args[0], [*metadata, *args[1:]]
    return annotation, tuple(metadata)


def _model_candidates(annotation: Any) -> list[type[PydanticBaseModel]]:
    """Return all Pydantic model classes reachable as direct Union branches of *annotation*."""
    annotation, _ = _unwrap_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return [annotation]
    if get_origin(annotation) in (Union, UnionType):
        return [c for arg in get_args(annotation) for c in _model_candidates(arg)]
    return []


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    """Normalise a default value to a plain Python container for OmegaConf merging."""
    if isinstance(value, PydanticBaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _as_mapping(value: Any) -> dict:
    plain = _to_plain(value)
    return plain if isinstance(plain, dict) else {}


def _field_default(field_info: Any, *, skip_factory: bool = False) -> Any:
    """Return the field's declared default, or ``_NO_DEFAULT`` if none exists."""
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if skip_factory or field_info.default_factory is None:
        return _NO_DEFAULT
    try:
        return field_info.default_factory()
    except Exception:  # noqa: BLE001
        return _NO_DEFAULT


def _field_value(data: Any, field_name: str, field_info: Any) -> Any:
    """Return the field's value from *data*, checking alias then name.

    Returns ``_NO_DEFAULT`` when the key is absent (distinct from an explicit
    ``None``), and ``None`` for unresolved interpolations or non-mapping data.
    """
    if not isinstance(data, DictConfig | dict):
        return None
    alias = field_info.alias
    key = alias if (alias and alias in data) else field_name if field_name in data else None
    if key is None:
        return _NO_DEFAULT
    try:
        return data.get(key)
    except OmegaConfBaseException:
        return None


def _resolve_union_model(
    annotation: Any,
    metadata: tuple[Any, ...],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Pick the right Union branch using a discriminator annotation.

    Handles both string discriminators (``Field(discriminator="field_name")``) and
    callable discriminators (``Discriminator(fn)`` with ``Tag`` annotations).
    Returns ``None`` when the union cannot be resolved.
    """
    discriminator = next((m.discriminator for m in metadata if hasattr(m, "discriminator")), None)
    if discriminator is None:
        return None
    if callable(discriminator):
        try:
            tag = discriminator(value)
        except Exception:  # noqa: BLE001
            return None
        return next(
            (
                ann
                for arg in get_args(annotation)
                for ann, meta in [_unwrap_annotated(arg)]
                if isinstance(ann, type)
                and issubclass(ann, PydanticBaseModel)
                and any(getattr(m, "tag", None) == tag for m in meta)
            ),
            None,
        )
    if not isinstance(value, DictConfig | dict):
        return None

    matches = []
    for candidate in get_args(annotation):
        if not (isinstance(candidate, type) and issubclass(candidate, PydanticBaseModel)):
            continue
        if discriminator not in candidate.model_fields:
            continue
        field = candidate.model_fields[discriminator]
        disc_ann, _ = _unwrap_annotated(field.annotation)
        if get_origin(disc_ann) is not Literal:
            continue
        # Check alias (e.g. _target_) first, then field name
        disc_val = value.get(field.alias) if field.alias else None
        if disc_val is None:
            disc_val = value.get(discriminator)
        if disc_val in get_args(disc_ann):
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        msg = f"Ambiguous discriminator {discriminator!r}: matched multiple branches."
        raise ValueError(msg)
    return None


def _resolve_model(
    annotation: Any,
    metadata: tuple[Any, ...],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Return the single Pydantic model *annotation* resolves to, or ``None``."""
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return annotation
    if get_origin(annotation) not in (Union, UnionType):
        return None
    candidates = _model_candidates(annotation)
    if len(candidates) == 1:
        return candidates[0]
    return _resolve_union_model(annotation, metadata, value)


def _deep_merge(default: Any, user: Any) -> Any:
    """Recursively merge *default* into *user*; user always wins."""
    if isinstance(default, dict) and isinstance(user, dict):
        merged = dict(default)
        for k, v in user.items():
            merged[k] = _deep_merge(merged[k], v) if k in merged else v
        return merged
    if isinstance(default, list) and isinstance(user, list):
        return [_deep_merge(default[i], v) if i < len(default) else v for i, v in enumerate(user)]
    return user


def _inject_container_defaults(annotation: Any, metadata: tuple[Any, ...], value: Any) -> Any | None:
    """Recursively inject schema defaults into list/dict container elements.

    Handles Union (tries each non-None branch), list/tuple/set (per-element),
    and dict (per-value) recursively. Returns ``None`` when the annotation is not
    a container or the value type doesn't match.
    """
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        for branch in get_args(annotation):
            if branch is type(None):
                continue
            branch_ann, branch_meta = _unwrap_annotated(branch)
            result = _inject_element_defaults(branch_ann, (*branch_meta, *metadata), value)
            if result is not None:
                return result
        return None

    if origin in (list, tuple, set):
        if not isinstance(value, list | ListConfig | tuple):
            return None
        args = get_args(annotation)
        el_ann, el_meta = _unwrap_annotated(args[0] if args else Any)
        return [_inject_element_defaults(el_ann, el_meta, el) or el for el in value]

    if origin is dict:
        if not isinstance(value, DictConfig | dict):
            return None
        args = get_args(annotation)
        val_ann, val_meta = _unwrap_annotated(args[1] if len(args) > 1 else Any)
        return {k: _inject_element_defaults(val_ann, val_meta, v) or v for k, v in value.items()}

    return None


def _inject_element_defaults(annotation: Any, metadata: tuple[Any, ...], value: Any) -> Any | None:
    """Inject defaults for a single element: try model-level first, then container recursion."""
    model = _resolve_model(annotation, metadata, value)
    if model is not None and isinstance(value, DictConfig | dict):
        return _deep_merge(schema_defaults(model, value), _as_mapping(value))
    return _inject_container_defaults(annotation, metadata, value)


def _field_defaults(field_name: str, field_info: Any, data: Any) -> tuple[bool, Any]:
    """Return ``(has_default, value)`` for one schema field, merging three sources.

    Priority, lowest to highest:

    1. *nested* — defaults from the field's nested model type.
    2. *explicit* — the field's ``default=`` / ``default_factory=``.
       Suppressed when nested defaults exist and the explicit default is ``None``
       but the user provided a value, so ``encoder: Encoder | None = None`` injects
       encoder defaults when the user set an encoder but stays ``None`` when omitted.
    3. *container* — per-element defaults for ``list[Model]`` / ``dict[str, Model]``
       fields (one level deep).
    """
    annotation, ann_meta = _unwrap_annotated(field_info.annotation)
    metadata = (*ann_meta, *tuple(field_info.metadata))
    value = _field_value(data, field_name, field_info)
    user_value = None if isinstance(value, _NoDefault) else value

    nested_model = _resolve_model(annotation, metadata, user_value)
    nested = (
        schema_defaults(nested_model, user_value if isinstance(user_value, DictConfig | dict) else None)
        if nested_model is not None
        else {}
    )
    explicit = _field_default(field_info, skip_factory=nested_model is not None)
    if explicit is None and nested and isinstance(user_value, DictConfig | dict):
        explicit = _NO_DEFAULT  # don't let None clobber nested defaults when user provided a value

    # Inject defaults into container fields (list/dict elements) and unresolved
    # Union fields. Delegates to _inject_container_defaults which recurses fully.
    # Skip if nested_model was already resolved — those defaults are in `nested`.
    container = None if nested_model is not None else _inject_container_defaults(annotation, metadata, user_value)

    # Fold layers left-to-right (later wins): nested < explicit < container.
    result = _NO_DEFAULT
    has = False
    for layer in (nested or _NO_DEFAULT, explicit, container if container is not None else _NO_DEFAULT):
        if layer is _NO_DEFAULT:
            continue
        has = True
        plain = _to_plain(layer)
        result = plain if result is _NO_DEFAULT else _deep_merge(result, plain)
    return has, (None if result is _NO_DEFAULT else result)


def schema_defaults(model: type[PydanticBaseModel], data: Any = None) -> dict[str, Any]:
    """Collect all defaults declared by *model*, skipping required fields with no default.

    User-provided values in *data* always win when merged back into the config.
    """
    defaults: dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        has, value = _field_defaults(field_name, field_info, data)
        if not has:
            continue
        # NullDropSchema subclasses are forwarded as **kwargs to functions that
        # cannot handle explicit None arguments (e.g. open_dataset(select=None)
        # treats None as a list element rather than "no selection").
        if value is None and issubclass(model, NullDropSchema):
            continue
        defaults[field_info.alias or field_name] = value
    return defaults


def apply_schema_defaults(config: DictConfig, schema: type[PydanticBaseModel]) -> DictConfig:
    """Return a new DictConfig with schema defaults merged into *config*."""
    merged = _deep_merge(schema_defaults(schema, config), _as_mapping(config))
    return OmegaConf.create(merged)


# ---------------------------------------------------------------------------
# Interpolation anchor pruning
# ---------------------------------------------------------------------------


def _walk_config(
    config: DictConfig | ListConfig,
    prefix: tuple[str, ...] = (),
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Walk *config* and return ``(interpolation_refs, all_paths)``.

    ``interpolation_refs`` — plain ``${a.b.c}`` reference targets (resolver
    expressions like ``${oc.env:HOME}`` are excluded).
    ``all_paths``          — every key path present in the config tree.
    """
    refs: set[tuple[str, ...]] = set()
    paths: set[tuple[str, ...]] = set()
    raw = OmegaConf.to_container(config, resolve=False)

    if isinstance(config, DictConfig) and isinstance(raw, dict):
        pairs = ((k, raw.get(k)) for k in config)
    elif isinstance(config, ListConfig) and isinstance(raw, list):
        pairs = enumerate(raw)
    else:
        return refs, paths

    for key, raw_val in pairs:
        path = (*prefix, str(key))
        paths.add(path)
        if OmegaConf.is_interpolation(config, key):
            for match in _INTERP_RE.findall(raw_val if isinstance(raw_val, str) else ""):
                if ":" not in match and (parts := tuple(p for p in match.split(".") if p)):
                    refs.add(parts)
        else:
            try:
                child = config[key]
            except Exception:  # noqa: BLE001, S112
                continue
            if isinstance(child, DictConfig | ListConfig):
                child_refs, child_paths = _walk_config(child, path)
                refs.update(child_refs)
                paths.update(child_paths)
    return refs, paths


def _interpolation_refs(config: DictConfig | ListConfig) -> set[tuple[str, ...]]:
    refs, _ = _walk_config(config)
    return refs


def _path_in_union_annotation(path: tuple[str, ...], annotation: Any) -> bool:
    """Return True if *path[1:]* is reachable through any non-None branch of a Union annotation."""
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        inner, _ = _unwrap_annotated(arg)
        if get_origin(inner) in (list, tuple, set, dict):
            return True
        if isinstance(inner, type) and issubclass(inner, PydanticBaseModel) and _path_in_schema(path[1:], inner):
            return True
    return False


@cache
def _path_in_schema(path: tuple[str, ...], schema: type[PydanticBaseModel]) -> bool:
    """Return ``True`` if *path* (or any descendant) is declared by *schema*.

    Container fields (list, dict) cover any sub-path.  Cached because the
    schema never changes at runtime.

    Examples
    --------
    - ``("model", "layer_kernels")`` → ``False``: not a field of ``ModelSchema``,
      so the top-level anchor is prunable.
    - ``("model", "encoder", "layer_kernels")`` → ``True``: declared by the
      encoder schema, so the resolved reference site is kept.
    - ``("diagnostics", "plot", "colormaps")`` → ``False``: convenience anchor
      not declared in the plot schema, prunable.
    - ``("diagnostics", "plot", "callbacks", "0")`` → ``True``: ``callbacks``
      is a ``list`` field, so any sub-path is covered.
    """
    if not path:
        return True
    # If the schema accepts extra fields, any path within it is valid.
    if getattr(schema, "model_config", {}).get("extra") == "allow":
        return True
    segment = path[0]
    for field_name, field_info in schema.model_fields.items():
        if segment not in (field_name, field_info.alias):
            continue
        if len(path) == 1:
            return True  # exact field match
        annotation, _ = _unwrap_annotated(field_info.annotation)
        origin = get_origin(annotation)
        # Any sub-path within a container field is valid
        if origin in (list, tuple, set, dict):
            return True
        # For union types, check each branch
        if origin in (Union, UnionType):
            return _path_in_union_annotation(path, annotation)
        if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
            return _path_in_schema(path[1:], annotation)
        return False  # scalar — no sub-paths possible
    return False


def _minimize_paths(paths: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Return the minimal set of paths whose deletion covers all candidates.

    Examples
    --------
    - ``[("model", "layer_kernels"), ("model", "layer_kernels", "Linear")]``
      → ``[("model", "layer_kernels")]``: parent deletion covers the child.
    - ``[("model", "layer_kernels"), ("diagnostics", "plot", "colormaps")]``
      → both kept: neither is a prefix of the other.
    """
    minimized: list[tuple[str, ...]] = []
    for path in sorted(set(paths), key=lambda p: (len(p), p)):
        if not any(path[: len(parent)] == parent for parent in minimized):
            minimized.append(path)
    return minimized


def _schema_root_of_path(path: tuple[str, ...], schema: type[PydanticBaseModel]) -> tuple[str, ...]:
    """Return the shortest prefix of *path* that is not declared by *schema*.

    When an interpolation reference points deep into an undeclared subtree
    (e.g. ``${diagnostics.plot.frequency.batch}``), the entire parent key
    (``diagnostics.plot.frequency``) should be pruned, not just the leaf.
    Walking up from the full path to the shortest undeclared prefix ensures
    siblings like ``frequency.epoch`` are pruned as collateral.
    """
    for i in range(1, len(path) + 1):
        if not _path_in_schema(path[:i], schema):
            return path[:i]
    return path


def _undeclared_anchor_paths(
    refs: set[tuple[str, ...]],
    config: DictConfig,
    schema: type[PydanticBaseModel],
) -> list[tuple[str, ...]]:
    """From pre-collected interpolation *refs*, return the prunable anchor paths."""
    prunable = []
    for path in sorted(refs):
        if not path or _path_in_schema(path, schema):
            continue
        root = _schema_root_of_path(path, schema)
        if len(root) == 1:
            exists = root[0] in config
        else:
            parent = OmegaConf.select(config, ".".join(root[:-1]), default=None)
            exists = isinstance(parent, DictConfig | dict) and root[-1] in parent
        if exists:
            prunable.append(root)
    return _minimize_paths(prunable)


def undeclared_interpolation_anchor_paths(
    config: DictConfig,
    schema: type[PydanticBaseModel],
) -> list[tuple[str, ...]]:
    """Return paths that are ``${...}`` interpolation sources but undeclared by *schema*.

    A path is prunable when it is referenced by an interpolation, not declared
    by the schema, and actually exists in the config.  The shortest undeclared
    prefix is used so that sibling keys in the same undeclared subtree are also
    removed (e.g. ``frequency.epoch`` when only ``frequency.batch`` is referenced).
    """
    if not isinstance(config, DictConfig):
        return []
    refs, _ = _walk_config(config)
    return _undeclared_anchor_paths(refs, config, schema)


def prune_undeclared_interpolation_anchors(config: DictConfig, paths: list[tuple[str, ...]]) -> None:
    """Delete the given anchor paths from *config* in-place."""
    for path in _minimize_paths(paths):
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
    """Resolve all interpolations then prune keys undeclared by *schema*.

    Two passes:
    1. Collect interpolation-anchor paths before resolving (interpolations must
       still be present to be detectable), then delete them after resolving so
       their values are already copied to the reference sites.
    2. Walk the resolved config recursively and remove any remaining keys not
       declared by the schema (covers non-referenced extras like ``focus_areas``
       or ``attributes.nodes``).
    """
    refs, all_paths = _walk_config(config)
    anchor_paths = _undeclared_anchor_paths(refs, config, schema)
    OmegaConf.resolve(config)
    prune_undeclared_interpolation_anchors(config, anchor_paths)
    extra = _minimize_paths(
        [_schema_root_of_path(path, schema) for path in all_paths if not _path_in_schema(path, schema)],
    )
    prune_undeclared_interpolation_anchors(config, extra)


__all__ = [
    "DatasetDict",
    "apply_schema_defaults",
    "prune_undeclared_interpolation_anchors",
    "resolve_and_prune_undeclared_interpolation_anchors",
    "schema_defaults",
    "undeclared_interpolation_anchor_paths",
]
