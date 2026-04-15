# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
import sys
from pathlib import Path
from typing import Any
from typing import Self
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from pydantic_core import PydanticCustomError
from pydantic_core import ValidationError

from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.models.schemas.decoder import GraphTransformerDecoderSchema
from anemoi.models.schemas.models import ModelSchema
from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import CUSTOM_MESSAGES
from anemoi.utils.schemas.errors import convert_errors

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)
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


def _get(obj: Any, key: str) -> Any:
    """Attribute-or-mapping access that works for both Pydantic models and plain dicts/DictConfigs."""
    return getattr(obj, key, None) or (obj.get(key) if hasattr(obj, "get") else None)


def _validate_multiscale_loss(training: Any) -> None:
    """Validate multiscale loss matrix source config.

    Runs for both validated (BaseSchema) and unvalidated (UnvalidatedBaseSchema)
    configs so the check is enforced regardless of the ``config_validation`` flag.
    """
    training_loss = _get(training, "training_loss")
    if training_loss is None:
        return

    loss_matrices = _get(training_loss, "loss_matrices")
    loss_matrices_graph = _get(training_loss, "loss_matrices_graph") or False

    file_based = loss_matrices is not None
    graph_based = loss_matrices_graph is True or isinstance(loss_matrices_graph, list)

    if file_based and graph_based:
        msg = "Specify either loss_matrices or loss_matrices_graph, not both."
        raise ValueError(msg)
    if not file_based and not graph_based:
        msg = "Specify loss_matrices, loss_matrices_graph=True, or an explicit loss_matrices_graph list."
        raise ValueError(msg)


def _validate_noise_projection(model: Any) -> None:
    """Validate that noise_matrix and noise_edges_name are not both specified.

    Runs for both validated (BaseSchema) and unvalidated (UnvalidatedBaseSchema)
    configs so the check is enforced regardless of the ``config_validation`` flag.
    Only applies to model configs that carry a ``noise_injector`` field
    (e.g. ``EnsModelSchema``).
    """
    noise_injector = _get(model, "noise_injector")
    if noise_injector is None:
        return
    noise_matrix = _get(noise_injector, "noise_matrix")
    noise_edges_name = _get(noise_injector, "noise_edges_name")
    if noise_matrix is not None and noise_edges_name is not None:
        msg = "Specify either noise_matrix or noise_edges_name, not both."
        raise ValueError(msg)


_DEPRECATED_TARGETS: dict[str, str] = {
    "anemoi.training.diagnostics.callbacks.plot.LongRolloutPlots": (
        "This callback has been deprecated and removed, update your config to remove any references to it. "
    ),
}


def _find_deprecated_target(data: Any, deprecated: dict[str, str]) -> tuple[str, str] | None:
    """Recursively search for deprecated _target_ values anywhere in a config."""
    if isinstance(data, str):
        return None
    if hasattr(data, "keys"):  # dict / DictConfig (not ListConfig)
        target = data.get("_target_")
        if target in deprecated:
            return target, deprecated[target]
        for v in data.values():
            result = _find_deprecated_target(v, deprecated)
            if result:
                return result
    elif hasattr(data, "__iter__"):  # list / ListConfig
        for item in data:
            result = _find_deprecated_target(item, deprecated)
            if result:
                return result
    return None


class SchemaCommonMixin:
    """Shared logic for schema objects."""

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)

    @model_validator(mode="before")
    @classmethod
    def _check_deprecated_targets(cls, values: Any) -> Any:
        """Raise before validation if any _target_ in the config is deprecated."""
        result = _find_deprecated_target(values, _DEPRECATED_TARGETS)
        if result:
            target, hint = result
            msg = f"'{target}' is deprecated and has been removed. {hint}"
            raise ValueError(msg)
        return values

    def model_post_init(self, _: Any) -> None:
        expand_paths(self.system)
        _validate_multiscale_loss(self.training)
        _validate_noise_projection(self.model)
        if self.diagnostics.log.mlflow.enabled and (
            self.system.output.logs.mlflow != self.diagnostics.log.mlflow.save_dir
        ):
            LOGGER.info("adjusting save_dir path to match output mlflow logs")
            self.diagnostics.log.mlflow.save_dir = str(self.system.output.logs.mlflow)


class BaseSchema(SchemaCommonMixin, BaseModel):
    """Top-level schema for the training configuration."""

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

    @model_validator(mode="after")
    def set_read_group_size_if_not_provided(self) -> Self:
        if not self.dataloader.read_group_size:
            self.dataloader.read_group_size = self.system.hardware.num_gpus_per_model
        return self

    @model_validator(mode="after")
    def check_bounding_not_used_with_data_extractor_zero(self) -> Self:
        """Check that bounding is not used with zero data extractor."""
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


class UnvalidatedBaseSchema(SchemaCommonMixin, PydanticBaseModel):
    data: Any
    """Data configuration."""
    dataloader: Any
    """Dataloader configuration."""
    diagnostics: Any
    """Diagnostics configuration such as logging, plots and metrics."""
    system: Any
    """Hardware configuration."""
    graph: Any
    """Graph configuration."""
    model: Any
    """Model configuration."""
    training: Any
    """Training configuration."""
    config_validation: bool = False
    """Flag to disable validation of the configuration"""


def convert_to_omegaconf(config: BaseSchema) -> DictConfig:
    config = config.model_dump(by_alias=True)
    return OmegaConf.create(config)


def validate_schema(config: DictConfig) -> BaseSchema:
    try:
        config = BaseSchema(**config)
    except ValidationError as e:
        errors = convert_errors(e, CUSTOM_MESSAGES)
        LOGGER.error(errors)  # noqa: TRY400
        sys.exit(0)
    else:
        return config
