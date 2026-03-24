# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re
from types import UnionType
from typing import Any
from typing import Annotated
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
from pydantic import BaseModel as PydanticBaseModel
from pydantic_core import PydanticUndefined

from .system import SystemSchema

# The DatasetDict type alias is intended to standardize the structure of dataset-related dictionaries
# across the codebase, improving type safety and code readability.
# The dataset-specific configurations are represented as a dictionary where keys are the dataset names
T = TypeVar("T")
DatasetDict: TypeAlias = dict[Literal["datasets"], dict[str, T]]
_MISSING = object()


def expand_paths(config_system: Union[SystemSchema, DictConfig]) -> Union[SystemSchema, DictConfig]:
    """Expand output paths relative to the configured root directory.

    This runs after the config is loaded so downstream code can always work
    with fully expanded paths instead of rebuilding them in multiple places.
    """
    output_config = config_system.output
    root_output_path = Path(output_config.root) if output_config.root else Path()
    # OutputSchema
    if output_config.plots:
        config_system.output.plots = root_output_path / output_config.plots
    if output_config.profiler:
        config_system.output.profiler = root_output_path / output_config.profiler

    # LogsSchema
    config_system.output.logs.root = (
        root_output_path / output_config.logs.root if output_config.logs.root else root_output_path
    )
    base = config_system.output.logs.root

    # LogsSchema
    output_config.logs.wandb = base / "wandb" if output_config.logs.wandb is None else base / output_config.logs.wandb
    output_config.logs.mlflow = (
        base / "mlflow" if output_config.logs.mlflow is None else base / output_config.logs.mlflow
    )
    # CheckPointSchema
    output_config.checkpoints.root = (
        root_output_path / output_config.checkpoints.root if output_config.checkpoints.root else root_output_path
    )

    return config_system


def _strip_annotated(annotation: Any) -> Any:
    """Return the underlying type from an Annotated annotation."""
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _unwrap_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split an Annotated type into its base type and attached metadata."""
    metadata: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        metadata.extend(args[1:])
    return annotation, tuple(metadata)


def _to_plain_default(value: Any) -> Any:
    """Convert model defaults into plain Python data before merging them.

    OmegaConf merges plain containers more predictably than Pydantic model
    instances, so we normalise nested defaults here first.
    """
    if isinstance(value, PydanticBaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _safe_field_default(field_info: Any, *, allow_factory: bool = True) -> Any:
    """Return a field's declared default value, or `_MISSING`.

    Default factories are best-effort: if they fail at import/runtime config
    boundaries, we skip them and let nested/container defaults fill what they can.
    """
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if not allow_factory or field_info.default_factory is None:
        return _MISSING
    try:
        return field_info.default_factory()
    except Exception:
        return _MISSING


def _resolve_discriminated_union_model(
    annotation: Any,
    metadata: tuple[Any, ...],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Pick the matching branch of a discriminated union.

    In lenient mode we still need to know which schema branch applies before
    we can copy over its defaults. We use the discriminator field declared on
    the union, for example `training.model_task`, and raise a plain error when
    the branch cannot be identified.
    """
    discriminator = next(
        (
            meta.discriminator
            for meta in metadata
            if hasattr(meta, "discriminator") and isinstance(meta.discriminator, str)
        ),
        None,
    )
    if discriminator is None:
        return None

    if not isinstance(value, DictConfig | dict):
        msg = f"Cannot determine schema branch without a mapping for discriminator '{discriminator}'."
        raise ValueError(msg)

    discriminator_keys = [discriminator]
    for candidate in get_args(annotation):
        if not (isinstance(candidate, type) and issubclass(candidate, PydanticBaseModel)):
            continue
        field_info = candidate.model_fields.get(discriminator)
        if field_info is None:
            continue
        if field_info.alias:
            discriminator_keys.append(field_info.alias)

    discriminator_value = None
    for key in discriminator_keys:
        if key in value and value.get(key) is not None:
            discriminator_value = value.get(key)
            break

    if discriminator_value is None:
        keys_display = ", ".join(repr(key) for key in dict.fromkeys(discriminator_keys))
        msg = f"Cannot determine schema branch: missing discriminator ({keys_display})."
        raise ValueError(msg)

    matches = []
    for candidate in get_args(annotation):
        if not (isinstance(candidate, type) and issubclass(candidate, PydanticBaseModel)):
            continue
        field_info = candidate.model_fields.get(discriminator)
        if field_info is None:
            continue
        field_annotation = _strip_annotated(field_info.annotation)
        if get_origin(field_annotation) is Literal and discriminator_value in get_args(field_annotation):
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = (
            f"Cannot determine schema branch for discriminator '{discriminator}': "
            f"got {discriminator_value!r}."
        )
        raise ValueError(msg)

    msg = (
        f"Cannot determine schema branch for discriminator '{discriminator}': "
        f"value {discriminator_value!r} matched multiple branches."
    )
    raise ValueError(msg)


def _resolve_model_from_annotation(
    annotation: Any,
    metadata: tuple[Any, ...],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Resolve a model type from an annotation and value."""
    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin not in (Union, UnionType):
        return None

    model_candidates = [
        candidate
        for candidate in get_args(annotation)
        if isinstance(candidate, type) and issubclass(candidate, PydanticBaseModel)
    ]
    if len(model_candidates) == 1:
        return model_candidates[0]
    resolved = _resolve_discriminated_union_model(annotation, metadata, value)
    if resolved is not None:
        return resolved
    return _resolve_union_by_target_field(model_candidates, value)


def _has_discriminator(metadata: tuple[Any, ...]) -> bool:
    """Return whether annotation metadata declares a discriminator."""
    return any(hasattr(meta, "discriminator") and isinstance(meta.discriminator, str) for meta in metadata)


def _resolve_union_by_target_field(
    model_candidates: list[type[PydanticBaseModel]],
    value: Any,
) -> type[PydanticBaseModel] | None:
    """Fallback union resolver using `_target_`/`target_` literal values.

    Some schema unions are not explicitly declared as discriminated unions but
    still carry an implicit discriminator via a `target_` field (aliased as
    `_target_`). This resolver keeps lenient default injection generic for
    those cases.
    """
    if not isinstance(value, DictConfig | dict):
        return None

    target_value = value.get("_target_", value.get("target_"))
    if target_value is None:
        return None

    matches: list[type[PydanticBaseModel]] = []
    for candidate in model_candidates:
        target_field = candidate.model_fields.get("target_")
        if target_field is None:
            continue
        target_annotation = _strip_annotated(target_field.annotation)
        if get_origin(target_annotation) is Literal and target_value in get_args(target_annotation):
            matches.append(candidate)

    return matches[0] if len(matches) == 1 else None


def _deep_merge_defaults(default_value: Any, user_value: Any) -> Any:
    """Recursively merge defaults into user values without clobbering list items.

    Rules:
    - Dictionaries merge per key; user values always win.
    - Lists merge by index for overlapping items; user list length is preserved.
    - Scalars and mismatched types return user value.
    """
    if isinstance(default_value, dict) and isinstance(user_value, dict):
        merged = dict(default_value)
        for key, value in user_value.items():
            if key in merged:
                merged[key] = _deep_merge_defaults(merged[key], value)
            else:
                merged[key] = value
        return merged

    if isinstance(default_value, list) and isinstance(user_value, list):
        merged_list = []
        for index, value in enumerate(user_value):
            if index < len(default_value):
                merged_list.append(_deep_merge_defaults(default_value[index], value))
            else:
                merged_list.append(value)
        return merged_list

    return user_value


def _merge_plain_defaults(defaults: dict[str, Any], value: DictConfig | dict[str, Any]) -> dict[str, Any]:
    plain_value = _to_plain_default(value)
    if not isinstance(plain_value, dict):
        return defaults
    return _deep_merge_defaults(defaults, plain_value)


def _merge_default_layers(*layers: Any) -> tuple[bool, Any]:
    """Merge default layers in order, where later layers override earlier.

    Use `_MISSING` to skip a layer. This keeps field-default composition
    (nested defaults, explicit defaults, container-derived defaults) compact
    and consistent.
    """
    merged: Any = _MISSING
    has_default = False

    for layer in layers:
        if layer is _MISSING:
            continue
        has_default = True
        plain_layer = _to_plain_default(layer)
        if merged is _MISSING:
            merged = plain_layer
            continue
        merged = _deep_merge_defaults(merged, plain_layer)

    return has_default, None if merged is _MISSING else merged


def _apply_container_defaults(annotation: Any, metadata: tuple[Any, ...], value: Any) -> Any | None:
    """Apply schema defaults recursively to list/dict containers."""
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        # Optional[T] and union-typed fields are handled by trying each concrete
        # branch until one applies to the runtime value.
        non_none_candidates = [candidate for candidate in get_args(annotation) if candidate is not type(None)]
        for candidate in non_none_candidates:
            candidate_annotation, candidate_metadata = _unwrap_annotated(candidate)
            candidate_metadata = (*candidate_metadata, *metadata)
            try:
                candidate_model = _resolve_model_from_annotation(candidate_annotation, candidate_metadata, value)
            except ValueError:
                candidate_model = None

            if candidate_model is not None and isinstance(value, DictConfig | dict):
                candidate_defaults = schema_defaults(candidate_model, value)
                return _merge_plain_defaults(candidate_defaults, value)

            candidate_container = _apply_container_defaults(candidate_annotation, candidate_metadata, value)
            if candidate_container is not None:
                return candidate_container

        return None

    if origin in (list, tuple, set):
        if not isinstance(value, (list, ListConfig, tuple)):
            return None

        element_annotation = get_args(annotation)[0] if get_args(annotation) else Any
        resolved_elements = []
        for element in value:
            item_annotation, item_metadata = _unwrap_annotated(element_annotation)
            item_metadata = (*item_metadata, *metadata)
            try:
                item_model = _resolve_model_from_annotation(item_annotation, item_metadata, element)
            except ValueError:
                item_model = None
            if item_model is not None and isinstance(element, DictConfig | dict):
                item_defaults = schema_defaults(item_model, element)
                resolved_elements.append(_merge_plain_defaults(item_defaults, element))
                continue

            nested_element = _apply_container_defaults(item_annotation, item_metadata, element)
            resolved_elements.append(nested_element if nested_element is not None else element)

        return resolved_elements

    if origin is dict:
        if not isinstance(value, DictConfig | dict):
            return None

        args = get_args(annotation)
        value_annotation = args[1] if len(args) > 1 else Any
        resolved_mapping = {}
        for key, item in value.items():
            item_annotation, item_metadata = _unwrap_annotated(value_annotation)
            item_metadata = (*item_metadata, *metadata)
            try:
                item_model = _resolve_model_from_annotation(item_annotation, item_metadata, item)
            except ValueError:
                item_model = None
            if item_model is not None and isinstance(item, DictConfig | dict):
                item_defaults = schema_defaults(item_model, item)
                resolved_mapping[key] = _merge_plain_defaults(item_defaults, item)
                continue

            nested_item = _apply_container_defaults(item_annotation, item_metadata, item)
            resolved_mapping[key] = nested_item if nested_item is not None else item

        return resolved_mapping

    return None


def resolve_field_model(field_info: Any, value: Any) -> type[PydanticBaseModel] | None:
    """Resolve the concrete nested model for a field, if there is one.

    This handles both direct nested models and unions of models. For unions
    with multiple branches, the branch must be identified through the field's
    discriminator metadata.
    """
    annotation, annotation_metadata = _unwrap_annotated(field_info.annotation)
    metadata = (*annotation_metadata, *tuple(field_info.metadata))

    return _resolve_model_from_annotation(annotation, metadata, value)


def schema_defaults(model: type[PydanticBaseModel], data: DictConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect only the defaults declared by a schema.

    The goal is to reuse Pydantic defaults even when strict validation is off.
    Required fields without defaults are left untouched, and user-provided
    values always win when the defaults are merged back into the config.
    """
    defaults: dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        annotation, annotation_metadata = _unwrap_annotated(field_info.annotation)
        metadata = (*annotation_metadata, *tuple(field_info.metadata))
        field_value = None
        if isinstance(data, DictConfig | dict):
            field_value = data.get(field_name)

        # Pick the concrete nested model first so defaults come from the same
        # branch the user selected, e.g. the right training schema for model_task.
        try:
            nested_model = resolve_field_model(field_info, field_value)
        except ValueError:
            if _has_discriminator(metadata) and field_value is not None:
                raise
            nested_model = None
        nested_defaults = (
            schema_defaults(nested_model, field_value if isinstance(field_value, DictConfig | dict) else None)
            if nested_model is not None
            else {}
        )
        explicit_default = _safe_field_default(field_info, allow_factory=nested_model is None)
        # Preserve previous behavior: when nested defaults are available, an
        # explicit `None` on the parent field should not erase them.
        if explicit_default is None and nested_defaults:
            explicit_default = _MISSING
        container_defaults = _apply_container_defaults(annotation, metadata, field_value)
        has_default, default_value = _merge_default_layers(
            nested_defaults if nested_defaults else _MISSING,
            explicit_default,
            container_defaults if container_defaults is not None else _MISSING,
        )

        if has_default:
            defaults[field_name] = default_value

    return defaults


def apply_schema_defaults(config: DictConfig, schema: type[PydanticBaseModel]) -> DictConfig:
    """Merge schema defaults into the raw config without overriding the user."""
    defaults = schema_defaults(schema, config)
    config_plain = _to_plain_default(config)
    if not isinstance(config_plain, dict):
        config_plain = {}
    # Use the custom deep merge instead of OmegaConf.merge so defaults also
    # propagate into nested list items (e.g. graph edge_builders entries).
    merged = _deep_merge_defaults(defaults, config_plain)
    return OmegaConf.create(merged)


_INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]+)\}")
_WILDCARD_PATH_SEGMENT = "*"


def _extract_plain_interpolation_paths(value: Any) -> set[tuple[str, ...]]:
    """Extract plain `${a.b}` references from a single interpolation value."""
    refs: set[tuple[str, ...]] = set()
    if not isinstance(value, str):
        return refs
    for match in _INTERPOLATION_PATTERN.findall(value):
        # Skip resolver expressions like ${oc.env:HOME}
        if ":" in match:
            continue
        parts = tuple(part for part in match.split(".") if part)
        if parts:
            refs.add(parts)
    return refs


def _collect_interpolation_paths(config: DictConfig | ListConfig) -> set[tuple[str, ...]]:
    """Collect interpolation source paths used by this config tree."""
    refs: set[tuple[str, ...]] = set()

    if isinstance(config, DictConfig):
        raw = OmegaConf.to_container(config, resolve=False)
        if not isinstance(raw, dict):
            return refs
        for key in config.keys():
            if OmegaConf.is_interpolation(config, key):
                refs.update(_extract_plain_interpolation_paths(raw.get(key)))
                continue
            try:
                child = config.get(key)
            except Exception:
                continue
            if isinstance(child, DictConfig | ListConfig):
                refs.update(_collect_interpolation_paths(child))
        return refs

    raw = OmegaConf.to_container(config, resolve=False)
    if not isinstance(raw, list):
        return refs
    for index in range(len(config)):
        if OmegaConf.is_interpolation(config, index):
            refs.update(_extract_plain_interpolation_paths(raw[index]))
            continue
        try:
            child = config[index]
        except Exception:
            continue
        if isinstance(child, DictConfig | ListConfig):
            refs.update(_collect_interpolation_paths(child))
    return refs


def _annotation_model_candidates(annotation: Any) -> list[type[PydanticBaseModel]]:
    """Return model candidates nested in an annotation."""
    annotation, _ = _unwrap_annotated(annotation)

    if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
        return [annotation]

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        candidates: list[type[PydanticBaseModel]] = []
        for candidate in get_args(annotation):
            candidates.extend(_annotation_model_candidates(candidate))
        return candidates

    if origin in (list, tuple, set):
        args = get_args(annotation)
        if not args:
            return []
        return _annotation_model_candidates(args[0])

    if origin is dict:
        args = get_args(annotation)
        if len(args) < 2:
            return []
        return _annotation_model_candidates(args[1])

    return []


def _collect_schema_path_patterns(
    model: type[PydanticBaseModel],
    prefix: tuple[str, ...] = (),
    seen_models: tuple[type[PydanticBaseModel], ...] = (),
) -> set[tuple[str, ...]]:
    """Collect schema-declared paths, using '*' for mapping key segments."""
    if model in seen_models:
        return set()

    patterns: set[tuple[str, ...]] = set()
    next_seen = (*seen_models, model)

    for field_name, field_info in model.model_fields.items():
        field_path = (*prefix, field_name)
        patterns.add(field_path)

        annotation, _ = _unwrap_annotated(field_info.annotation)
        origin = get_origin(annotation)

        if origin is dict:
            args = get_args(annotation)
            if len(args) > 1:
                for candidate in _annotation_model_candidates(args[1]):
                    patterns.update(
                        _collect_schema_path_patterns(
                            candidate,
                            (*field_path, _WILDCARD_PATH_SEGMENT),
                            next_seen,
                        ),
                    )
            continue

        for candidate in _annotation_model_candidates(annotation):
            patterns.update(_collect_schema_path_patterns(candidate, field_path, next_seen))

    return patterns


@lru_cache(maxsize=None)
def _schema_path_patterns(schema: type[PydanticBaseModel]) -> tuple[tuple[str, ...], ...]:
    """Cache schema path patterns for fast interpolation-anchor checks."""
    return tuple(sorted(_collect_schema_path_patterns(schema)))


def _is_schema_declared_path(
    schema: type[PydanticBaseModel],
    path: tuple[str, ...],
) -> bool:
    """Return True if a path is part of the schema-declared shape."""
    for pattern in _schema_path_patterns(schema):
        if len(pattern) != len(path):
            continue
        if all(
            pattern_part == _WILDCARD_PATH_SEGMENT or pattern_part == path_part
            for pattern_part, path_part in zip(pattern, path, strict=False)
        ):
            return True
    return False


def _delete_mapping_key_path(config: DictConfig, path: tuple[str, ...]) -> None:
    """Delete a mapping key path from DictConfig if present."""
    if not path:
        return
    if len(path) == 1:
        with open_dict(config):
            config.pop(path[0], None)
        return

    parent = OmegaConf.select(config, ".".join(path[:-1]), default=None)
    if isinstance(parent, DictConfig | dict):
        with open_dict(parent):
            parent.pop(path[-1], None)


def undeclared_interpolation_anchor_paths(
    config: DictConfig,
    schema: type[PydanticBaseModel],
) -> list[tuple[str, ...]]:
    """Return interpolation-anchor paths that are not declared by schema.

    This keeps lenient-mode output shape aligned with strict-mode shape while
    still allowing config-level forwarding aliases to be used during resolution.
    """
    if not isinstance(config, DictConfig):
        return []
    undeclared_paths: list[tuple[str, ...]] = []
    for path in sorted(_collect_interpolation_paths(config)):
        if not _is_schema_declared_path(schema, path):
            undeclared_paths.append(path)
    return undeclared_paths


def prune_undeclared_interpolation_anchors(config: DictConfig, paths: list[tuple[str, ...]]) -> None:
    """Drop previously collected interpolation-anchor key paths."""
    for path in paths:
        _delete_mapping_key_path(config, path)


def resolve_lineage_run(
    run_id: str | None,
    fork_run_id: str | None,
    parent_run_server2server: str | None,
) -> str | None:
    """Resolve the effective lineage run used for output directories."""
    if run_id:
        return parent_run_server2server or run_id
    if fork_run_id:
        return parent_run_server2server or fork_run_id
    return None


def build_runtime_system(
    system_config: SystemSchema | DictConfig,
    run_id: str | None,
    fork_run_id: str | None,
    parent_run_server2server: str | None,
) -> SystemSchema | DictConfig:
    """Return the finalized system config used by the trainer at runtime.

    This is the runtime step that happens after the base schema has already
    expanded output paths. It appends the effective run lineage to the output
    directories used by checkpoints and plots.
    """
    system_with_lineage = deepcopy(system_config)
    lineage_run = resolve_lineage_run(run_id, fork_run_id, parent_run_server2server)
    if lineage_run is not None:
        system_with_lineage.output.checkpoints.root = Path(system_with_lineage.output.checkpoints.root, lineage_run)
        if system_with_lineage.output.plots is not None:
            system_with_lineage.output.plots = Path(system_with_lineage.output.plots, lineage_run)

    return system_with_lineage
