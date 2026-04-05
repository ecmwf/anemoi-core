# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Self
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic import ValidationError
from pydantic import model_validator
from pydantic_core import PydanticCustomError

from anemoi.models.schemas.decoder import GraphTransformerDecoderSchema
from anemoi.utils.schemas import BaseModel

from .schema_utils import apply_schema_defaults
from .schema_utils import resolve_and_prune_undeclared_interpolation_anchors
from .validation_errors import ConfigValidationError
from .validation_errors import format_validation_error

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)


if TYPE_CHECKING:
    from anemoi.graphs.schemas.base_graph import BaseGraphSchema
    from anemoi.models.schemas.models import ModelSchema

    from .data import DataSchema
    from .dataloader import DataLoaderSchema
    from .diagnostics import DiagnosticsSchema
    from .system import SystemSchema
    from .training import TrainingSchema

LOGGER = logging.getLogger(__name__)


def expand_paths(config_system: Union[SystemSchema, DictConfig]) -> Union[SystemSchema, DictConfig]:
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


_DEPRECATED_TARGETS: dict[str, str] = {
    "anemoi.training.diagnostics.callbacks.plot.LongRolloutPlots": (
        "This callback has been deprecated and removed, update your config to remove any references to it. "
    ),
}


def _find_deprecated_in_mapping(data: Any, deprecated: dict[str, str]) -> tuple[str, str] | None:
    """Search a dict-like node for deprecated _target_ values."""
    try:
        target = data.get("_target_")
    except OmegaConfBaseException:
        target = None
    if target in deprecated:
        return target, deprecated[target]
    for key in data:
        try:
            v = data[key]
        except OmegaConfBaseException:
            continue
        result = _find_deprecated_target(v, deprecated)
        if result:
            return result
    return None


def _find_deprecated_in_sequence(data: Any, deprecated: dict[str, str]) -> tuple[str, str] | None:
    """Search a list-like node for deprecated _target_ values."""
    for item in data:
        try:
            result = _find_deprecated_target(item, deprecated)
        except OmegaConfBaseException:
            continue
        if result:
            return result
    return None


def _find_deprecated_target(data: Any, deprecated: dict[str, str]) -> tuple[str, str] | None:
    """Recursively search for deprecated _target_ values anywhere in a config.

    Skips values that raise OmegaConf errors (unresolved interpolations, mandatory
    placeholders) so the check is safe to run on partially-resolved configs.
    """
    if isinstance(data, str):
        return None
    if hasattr(data, "keys"):  # dict / DictConfig (not ListConfig)
        return _find_deprecated_in_mapping(data, deprecated)
    if hasattr(data, "__iter__"):  # list / ListConfig
        return _find_deprecated_in_sequence(data, deprecated)
    return None


def _check_deprecated_targets(config: Any) -> None:
    """Raise ConfigValidationError if any _target_ in the config is deprecated."""
    result = _find_deprecated_target(config, _DEPRECATED_TARGETS)
    if result:
        target, hint = result
        msg = f"'{target}' is deprecated and has been removed. {hint}"
        raise ConfigValidationError(msg)


def apply_runtime_postprocessing(config: BaseSchema | UnvalidatedBaseSchema) -> None:
    """Apply shared runtime adjustments after parsing either schema type.

    These updates are not really validation rules. They are normalisation
    steps that both strict and lenient configs should receive.
    """
    expand_paths(config.system)

    if not config.dataloader.read_group_size:
        config.dataloader.read_group_size = config.system.hardware.num_gpus_per_model
    if config.diagnostics.log.mlflow.enabled and (
        config.system.output.logs.mlflow != config.diagnostics.log.mlflow.save_dir
    ):
        LOGGER.info("adjusting save_dir path to match output mlflow logs")
        config.diagnostics.log.mlflow.save_dir = str(config.system.output.logs.mlflow)


class BaseSchema(BaseModel):
    """Top-level schema for the training configuration.

    Args:
        data: Data configuration
        dataloader: Dataloader configuration
        diagnostics: Diagnostics configuration such as logging, plots and metrics
        system: System configuration, including filesystem and hardware specification
        graph: Graph configuration
        model: Model configuration
        training: Training configuration
        config_validation: Flag to disable validation of the configuration. Defaults to True.

    """

    data: DataSchema
    """Data configuration."""
    dataloader: DataLoaderSchema
    """Dataloader configuration."""
    diagnostics: DiagnosticsSchema
    """Diagnostics configuration such as logging, plots and metrics."""
    system: SystemSchema
    """System configuration, including filesystem and hardware specification."""
    graph: BaseGraphSchema
    """Graph configuration."""
    model: ModelSchema
    """Model configuration."""
    training: TrainingSchema
    """Training configuration."""
    config_validation: bool = True
    """Flag to disable validation of the configuration"""

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)

    @model_validator(mode="after")
    def check_bounding_not_used_with_data_extractor_zero(self) -> Self:
        """Check that bounding is not used with zero data extractor."""
        if not self.config_validation:
            return self

        if (
            isinstance(self.model.decoder, GraphTransformerDecoderSchema)
            and self.model.decoder.initialise_data_extractor_zero
            and self.model.bounding
        ):
            error = "bounding_conflict_with_data_extractor_zero"
            msg = (
                "Boundings cannot be used with zero initialized weights in decoder. "
                "Set initalise_data_extractor_zero to False."
            )
            raise PydanticCustomError(
                error,
                msg,
            )
        return self


class UnvalidatedBaseSchema(PydanticBaseModel):
    """Permissive top-level schema used when strict validation is disabled.

    This keeps the original user values, including extras and invalid values,
    while still allowing us to merge in defaults and run shared postprocessing.
    """

    model_config = ConfigDict(extra="allow")

    data: Any
    dataloader: Any
    diagnostics: Any
    system: Any
    graph: Any
    model: Any
    training: Any
    config_validation: bool = False

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)


def build_schema(config: DictConfig) -> BaseSchema | UnvalidatedBaseSchema:
    """Build the top-level config in strict or lenient mode.

    Strict mode parses the full typed schema. Lenient mode keeps the permissive
    schema, but first fills in any defaults that can be identified safely from
    the strict schema definition.
    """
    if config.get("config_validation", True):
        LOGGER.info("Performing strict config validation.")
        OmegaConf.resolve(config)
        _check_deprecated_targets(config)
        try:
            parsed_config = BaseSchema(**config)
        except ValidationError as error:
            raise ConfigValidationError(format_validation_error(error)) from error
        apply_runtime_postprocessing(parsed_config)
    else:
        LOGGER.info("Skipping strict config validation.")
        # Apply defaults generically from the full schema tree, then resolve any
        # remaining interpolations before creating the permissive runtime model.
        # After resolution, prune interpolation-anchor keys not declared in
        # schema so lenient output shape matches strict output shape.
        config_with_defaults = apply_schema_defaults(config, BaseSchema)
        resolve_and_prune_undeclared_interpolation_anchors(config_with_defaults, BaseSchema)
        _check_deprecated_targets(config_with_defaults)
        parsed_config = UnvalidatedBaseSchema(**config_with_defaults)
        apply_runtime_postprocessing(parsed_config)
    return parsed_config


def convert_to_omegaconf(config: BaseSchema | UnvalidatedBaseSchema) -> dict:
    """Convert either schema representation back into an OmegaConf object."""
    config = config.model_dump(by_alias=True)
    return OmegaConf.create(config)
