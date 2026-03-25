# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Extract and merge Pydantic schema defaults into OmegaConf configs.

The central problem: configs may arrive without optional keys (e.g. user omitted
``training.deterministic``). We want to fill those gaps using the defaults declared
on the schema — but without running full Pydantic validation, so that partially-valid
configs (lenient mode) still get sensible defaults injected.

Public API:
    schema_defaults(model, data)       -- collect defaults from a model class
    apply_schema_defaults(config, schema) -- merge those defaults into a DictConfig
"""

from __future__ import annotations

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
from omegaconf.errors import OmegaConfBaseException
from pydantic import BaseModel as PydanticBaseModel
from pydantic_core import PydanticUndefined

T = TypeVar("T")
# Convenience alias used by training/dataloader schemas to type dataset-keyed dicts.
DatasetDict: TypeAlias = dict[Literal["datasets"], dict[str, T]]

# Sentinel that distinguishes "no default exists" from "default is None".
_MISSING = object()


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _unwrap_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Peel off all Annotated wrappers, collecting metadata along the way.

    ``Annotated[Annotated[T, m1], m2]`` → ``(T, (m1, m2))``.
    """
    metadata: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation, metadata = args[0], [*metadata, *args[1:]]
    return annotation, tuple(metadata)


def _strip_annotated(annotation: Any) -> Any:
    """Return the bare type, discarding any Annotated metadata."""
    return _unwrap_annotated(annotation)[0]


def _annotation_model_candidates(annotation: Any) -> list[type[PydanticBaseModel]]:
    """Return every Pydantic model class reachable from an annotation.

    Walks Union, list/tuple/set (element type), and dict (value type) recursively.
    Used to decide whether a field's annotation can yield schema defaults at all.
    """
    annotation, _ = _unwrap_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return [annotation]
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return [c for arg in get_args(annotation) for c in _annotation_model_candidates(arg)]
    if origin in (list, tuple, set):
        args = get_args(annotation)
        return _annotation_model_candidates(args[0]) if args else []
    if origin is dict:
        args = get_args(annotation)
        return _annotation_model_candidates(args[1]) if len(args) >= 2 else []
    return []


def _direct_model_candidates(annotation: Any) -> list[type[PydanticBaseModel]]:
    """Return models that are direct Union branches of *annotation* — not inside containers.

    Unlike ``_annotation_model_candidates``, this does NOT recurse into list/tuple/set/dict.
    Used by ``_resolve_model_from_annotation`` so that e.g. ``dict[str, Model] | None`` is
    not mistakenly treated as if it were the model itself: the dict is a container whose
    elements are models, and that case is handled by ``_apply_container_defaults``.
    """
    annotation, _ = _unwrap_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return [annotation]
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return [c for arg in get_args(annotation) for c in _direct_model_candidates(arg)]
    return []


# ---------------------------------------------------------------------------
# Plain-data conversion helpers
# ---------------------------------------------------------------------------


def _to_plain_default(value: Any) -> Any:
    """Normalise a default value to a plain Python container.

    OmegaConf merges plain dicts/lists more predictably than Pydantic model
    instances, so we convert before merging.
    """
    if isinstance(value, PydanticBaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    """Return *value* as a plain ``dict``, or ``{}`` if it is not a mapping."""
    plain = _to_plain_default(value)
    return plain if isinstance(plain, dict) else {}


# ---------------------------------------------------------------------------
# Field introspection helpers
# ---------------------------------------------------------------------------


def _safe_field_default(field_info: Any, *, allow_factory: bool = True) -> Any:
    """Return the field's declared default, or ``_MISSING`` if none exists.

    ``allow_factory=False`` skips calling ``default_factory``. We set this when a
    nested model has already been resolved — calling the factory would produce a
    plain model instance that could overwrite the richer nested defaults we collected.
    """
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if not allow_factory or field_info.default_factory is None:
        return _MISSING
    try:
        return field_info.default_factory()
    except Exception:
        # Factories that depend on runtime state (e.g. env vars) may fail here.
        return _MISSING


def _field_key_present(data: DictConfig | dict[str, Any] | None, field_name: str, field_info: Any) -> bool:
    """Return True if the field's key (name or alias) exists in *data* at all.

    Distinguishes an absent key from an explicitly-set ``None`` value — both
    would return ``None`` from ``_field_input_value``, but only the latter has
    the key present.  Used to decide whether injecting a ``None`` default is safe.
    """
    if not isinstance(data, DictConfig | dict):
        return False
    alias = field_info.alias
    return bool((alias and alias in data) or field_name in data)


def _field_input_value(data: DictConfig | dict[str, Any] | None, field_name: str, field_info: Any) -> Any:
    """Read the current value of a field from the user config.

    Checks both the canonical field name and its alias (e.g. ``target_`` /
    ``_target_``). Returns ``None`` when the key is absent or the value is an
    unresolved interpolation that would raise at access time.
    """
    if not isinstance(data, DictConfig | dict):
        return None
    alias = field_info.alias
    try:
        return data.get(alias) if alias and alias in data else data.get(field_name)
    except OmegaConfBaseException:
        # Unresolved interpolations (e.g. missing env vars) — treat as absent.
        return None


# ---------------------------------------------------------------------------
# Union / discriminator resolution
# ---------------------------------------------------------------------------


def _resolve_discriminated_union_model(
    annotation: Any,
    metadata: tuple[Any, ...],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Identify which branch of a discriminated union matches *value*.

    Looks for a ``Discriminator("field_name")`` object in *metadata* (placed there
    by ``Annotated[Union[...], Discriminator(...)]``). Then reads that field from
    *value* and matches it against each branch's ``Literal`` annotation.

    Returns ``None`` (rather than raising) for any unresolvable case — unknown
    discriminator value, missing key, non-dict value — so callers can fall back
    gracefully. Only raises when the schema itself is ambiguous (two branches
    declare the same discriminator value).
    """
    discriminator = next(
        (m.discriminator for m in metadata if hasattr(m, "discriminator") and isinstance(m.discriminator, str)),
        None,
    )
    if discriminator is None or not isinstance(value, DictConfig | dict):
        return None

    candidates = [c for c in get_args(annotation) if isinstance(c, type) and issubclass(c, PydanticBaseModel)]
    # Collect both the field name and any alias (e.g. target_ and _target_)
    # so we can find the discriminator value regardless of which key the user used.
    aliases = [
        c.model_fields[discriminator].alias
        for c in candidates
        if discriminator in c.model_fields and c.model_fields[discriminator].alias
    ]
    keys = list(dict.fromkeys([discriminator, *aliases]))  # ordered, deduplicated

    discriminator_value = next((value.get(k) for k in keys if k in value and value.get(k) is not None), None)
    if discriminator_value is None:
        return None  # discriminator key absent from config — can't determine branch

    matches = [
        c
        for c in candidates
        if discriminator in c.model_fields
        and get_origin(_strip_annotated(c.model_fields[discriminator].annotation)) is Literal
        and discriminator_value in get_args(_strip_annotated(c.model_fields[discriminator].annotation))
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Cannot determine schema branch for discriminator {discriminator!r}: "
            f"value {discriminator_value!r} matched multiple branches.",
        )
    return None  # value not listed in any schema branch (e.g. external/unknown class)


def _resolve_model_from_annotation(
    annotation: Any,
    metadata: tuple[Any, ...],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Return the single Pydantic model that *annotation* resolves to, or ``None``.

    For a plain model type, returns it directly. For a union, uses the discriminator
    in *metadata* to pick the right branch. Returns ``None`` when the annotation
    contains no model (e.g. scalar types) or when the branch can't be identified.
    """
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return annotation
    if get_origin(annotation) not in (Union, UnionType):
        return None
    model_candidates = _direct_model_candidates(annotation)
    if len(model_candidates) == 1:
        # Only one possible model — no discriminator needed.
        return model_candidates[0]
    return _resolve_discriminated_union_model(annotation, metadata, value)


# ---------------------------------------------------------------------------
# Default merging
# ---------------------------------------------------------------------------


def _deep_merge_defaults(default_value: Any, user_value: Any) -> Any:
    """Recursively merge *default_value* into *user_value*; user always wins.

    - Dicts: merged key-by-key; user keys override defaults, extra user keys kept.
    - Lists: merged element-by-element up to the user list's length; extra user
      elements are kept as-is (list length is never extended by defaults).
    - Scalars / type mismatch: return *user_value* unchanged.
    """
    if isinstance(default_value, dict) and isinstance(user_value, dict):
        merged = dict(default_value)
        for k, v in user_value.items():
            merged[k] = _deep_merge_defaults(merged[k], v) if k in merged else v
        return merged
    if isinstance(default_value, list) and isinstance(user_value, list):
        return [
            _deep_merge_defaults(default_value[i], v) if i < len(default_value) else v for i, v in enumerate(user_value)
        ]
    return user_value


def _merge_default_layers(*layers: Any) -> tuple[bool, Any]:
    """Fold multiple default layers into one, left-to-right (later layers win).

    ``_MISSING`` layers are skipped entirely. Returns ``(has_default, value)``
    where ``has_default`` is False only if every layer was ``_MISSING``.
    """
    merged: Any = _MISSING
    has_default = False
    for layer in layers:
        if layer is _MISSING:
            continue
        has_default = True
        plain = _to_plain_default(layer)
        merged = plain if merged is _MISSING else _deep_merge_defaults(merged, plain)
    return has_default, None if merged is _MISSING else merged


# ---------------------------------------------------------------------------
# Container / element default injection
# ---------------------------------------------------------------------------


def _resolved_model_defaults_for_value(annotation: Any, metadata: tuple[Any, ...], value: Any) -> dict[str, Any] | None:
    """If *annotation* resolves to a model and *value* is a mapping, inject defaults.

    Returns the user mapping with schema defaults filled in, or ``None`` if the
    annotation doesn't resolve to a model or the value isn't a mapping.
    """
    model = _resolve_model_from_annotation(annotation, metadata, value)
    if model is None or not isinstance(value, DictConfig | dict):
        return None
    return _deep_merge_defaults(schema_defaults(model, value), _as_plain_mapping(value))


def _try_resolve_element(annotation: Any, metadata: tuple[Any, ...], value: Any) -> Any | None:
    """Try both model-level and container-level default injection for one element.

    First attempts to treat the element as a model instance (mapping → model
    defaults). Falls back to container traversal (list/dict of models). Returns
    ``None`` if neither applies.
    """
    return _resolved_model_defaults_for_value(annotation, metadata, value) or _apply_container_defaults(
        annotation,
        metadata,
        value,
    )


def _apply_container_defaults(annotation: Any, metadata: tuple[Any, ...], value: Any) -> Any | None:
    """Recursively inject schema defaults into list/dict container elements.

    Handles three structural cases:
    - Union: tries each non-None branch in order, returns first match.
    - list/tuple/set: applies element-level defaults to each item.
    - dict: applies value-level defaults to each value.

    Returns ``None`` when the annotation isn't a container or the value doesn't
    match the expected container type.
    """
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        for candidate in get_args(annotation):
            if candidate is type(None):
                continue
            c_annotation, c_metadata = _unwrap_annotated(candidate)
            if result := _try_resolve_element(c_annotation, (*c_metadata, *metadata), value):
                return result
        return None

    if origin in (list, tuple, set):
        if not isinstance(value, (list, ListConfig, tuple)):
            return None
        args = get_args(annotation)
        el_annotation, el_metadata = _unwrap_annotated(args[0] if args else Any)
        return [_try_resolve_element(el_annotation, (*el_metadata, *metadata), el) or el for el in value]

    if origin is dict:
        if not isinstance(value, DictConfig | dict):
            return None
        args = get_args(annotation)
        val_annotation, val_metadata = _unwrap_annotated(args[1] if len(args) > 1 else Any)
        return {k: _try_resolve_element(val_annotation, (*val_metadata, *metadata), v) or v for k, v in value.items()}

    return None


# ---------------------------------------------------------------------------
# Per-field default resolution
# ---------------------------------------------------------------------------


def _default_for_field(
    field_name: str,
    field_info: Any,
    data: DictConfig | dict[str, Any] | None,
) -> tuple[bool, Any]:
    """Compute the composed default for a single schema field.

    Three default sources are collected and priority-merged (lowest → highest):

    1. **nested_defaults**: if the field's type is a Pydantic model, recurse into
       it with ``schema_defaults`` to collect its own defaults, guided by the user's
       actual value so the correct union branch is selected.

    2. **explicit_default**: the ``default=`` / ``default_factory=`` declared on the
       field itself. ``default_factory`` is skipped when a nested model was resolved
       (the factory would produce an empty instance that could overwrite richer
       nested defaults). A ``default=None`` is treated as absent (``_MISSING``) when
       nested defaults exist, so an optional model field doesn't erase its children.

    3. **container_defaults**: when the field holds a list/dict of models, inject
       defaults into each element of the user's current value.

    Returns ``(has_default, value)``. ``has_default`` is False only for required
    fields with no default at any layer — those are left untouched by the caller.
    """
    annotation, annotation_metadata = _unwrap_annotated(field_info.annotation)
    metadata = (*annotation_metadata, *tuple(field_info.metadata))
    field_value = _field_input_value(data, field_name, field_info)

    # --- Layer 1: nested model defaults ---
    nested_model = _resolve_model_from_annotation(annotation, metadata, field_value)
    nested_defaults = (
        schema_defaults(nested_model, field_value if isinstance(field_value, DictConfig | dict) else None)
        if nested_model is not None
        else {}
    )

    # --- Layer 2: explicit field default ---
    explicit_default = _safe_field_default(field_info, allow_factory=nested_model is None)
    # An explicit None on an optional model field (e.g. `encoder: Encoder | None = None`)
    # would clobber the nested defaults we just collected for a user-provided encoder.
    if explicit_default is None and nested_defaults:
        explicit_default = _MISSING

    # --- Layer 3: container element defaults ---
    container_defaults = _apply_container_defaults(annotation, metadata, field_value)

    return _merge_default_layers(
        nested_defaults or _MISSING,
        explicit_default,
        container_defaults if container_defaults is not None else _MISSING,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def schema_defaults(model: type[PydanticBaseModel], data: DictConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect the defaults declared by *model*, shaped to match *data*.

    Only fields that have a default (at any layer) are included. Required fields
    without defaults are omitted so they are never accidentally filled in.
    User-provided values in *data* always win when these defaults are later merged
    back into the config.
    """
    defaults: dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        has_default, default_value = _default_for_field(field_name, field_info, data)
        if has_default:
            # Some schemas signal that their absent optional fields should NOT
            # have ``None`` injected (see ``_SKIP_NULL_DEFAULTS`` marker below).
            # This protects downstream consumers that are called with ``**config``
            # and cannot handle explicit ``None`` keyword arguments (e.g. open_dataset
            # treats ``select=None`` as a list containing None).
            if (
                default_value is None
                and getattr(model, "_skip_null_defaults", False)
                and not _field_key_present(data, field_name, field_info)
            ):
                continue
            # Use the alias as the config key when present (e.g. _target_ instead of target_).
            defaults[field_info.alias or field_name] = default_value
    return defaults


def apply_schema_defaults(config: DictConfig, schema: type[PydanticBaseModel]) -> DictConfig:
    """Return a new DictConfig with schema defaults merged into *config*.

    Schema defaults fill in missing keys; any value already present in *config*
    is preserved unchanged.
    """
    merged = _deep_merge_defaults(schema_defaults(schema, config), _as_plain_mapping(config))
    return OmegaConf.create(merged)
