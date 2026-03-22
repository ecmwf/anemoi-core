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
from pathlib import Path
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
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic_core import PydanticUndefined

from .system import SystemSchema

# The DatasetDict type alias is intended to standardize the structure of dataset-related dictionaries
# across the codebase, improving type safety and code readability.
# The dataset-specific configurations are represented as a dictionary where keys are the dataset names
T = TypeVar("T")
DatasetDict: TypeAlias = dict[Literal["datasets"], dict[str, T]]


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

    discriminator_value = value.get(discriminator)
    if discriminator_value is None:
        msg = f"Cannot determine schema branch: missing discriminator '{discriminator}'."
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


def resolve_field_model(field_info: Any, value: Any) -> type[PydanticBaseModel] | None:
    """Resolve the concrete nested model for a field, if there is one.

    This handles both direct nested models and unions of models. For unions
    with multiple branches, the branch must be identified through the field's
    discriminator metadata.
    """
    annotation, annotation_metadata = _unwrap_annotated(field_info.annotation)
    metadata = (*annotation_metadata, *tuple(field_info.metadata))

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

    return _resolve_discriminated_union_model(annotation, metadata, value)


def schema_defaults(model: type[PydanticBaseModel], data: DictConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect only the defaults declared by a schema.

    The goal is to reuse Pydantic defaults even when strict validation is off.
    Required fields without defaults are left untouched, and user-provided
    values always win when the defaults are merged back into the config.
    """
    defaults: dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        field_value = None
        if isinstance(data, DictConfig | dict):
            field_value = data.get(field_name)

        # Pick the concrete nested model first so defaults come from the same
        # branch the user selected, e.g. the right training schema for model_task.
        nested_model = resolve_field_model(field_info, field_value)
        has_default = field_info.default is not PydanticUndefined or field_info.default_factory is not None
        default_value = None
        if field_info.default is not PydanticUndefined:
            default_value = _to_plain_default(field_info.default)
        elif field_info.default_factory is not None and nested_model is None:
            default_value = _to_plain_default(field_info.default_factory())
        nested_defaults = (
            schema_defaults(nested_model, field_value if isinstance(field_value, DictConfig | dict) else None)
            if nested_model is not None
            else {}
        )

        if nested_defaults:
            if isinstance(default_value, dict):
                default_value = {**nested_defaults, **default_value}
            elif default_value is None:
                default_value = nested_defaults
                has_default = True
            elif has_default:
                default_value = default_value
            else:
                default_value = nested_defaults
                has_default = True

        if has_default:
            defaults[field_name] = default_value

    return defaults


def apply_schema_defaults(config: DictConfig, schema: type[PydanticBaseModel]) -> DictConfig:
    """Merge schema defaults into the raw config without overriding the user."""
    defaults = OmegaConf.create(schema_defaults(schema, config))
    return OmegaConf.merge(defaults, config)


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
