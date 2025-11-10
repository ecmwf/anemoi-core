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
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from pydantic._internal import _model_construction
from pydantic_core import PydanticCustomError
from pydantic_core import ValidationError
from typing_extensions import Self

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
from .datamodule import DataModuleSchema
from .diagnostics import DiagnosticsSchema
from .system import SystemSchema
from .training import TrainingSchema

_object_setattr = _model_construction.object_setattr

LOGGER = logging.getLogger(__name__)


def expand_paths(config_system: Union[SystemSchema, DictConfig]) -> None:
    output_config = config_system.output
    root_output_path = Path(output_config.root) if output_config.root else Path()
    # OutputSchema
    if output_config.plots:
        config_system.output.plots = root_output_path / output_config.plots
    if output_config.profiler:
        config_system.output.profiler = root_output_path / output_config.profiler
    if output_config.logs:
        config_system.output.logs.root = root_output_path / output_config.logs.root

        base = root_output_path / output_config.logs.root

        # LogsSchema
        if output_config.logs.wandb is None:
            output_config.logs.wandb = base / "wandb"
        if output_config.logs.mlflow is None:
            output_config.logs.mlflow = base / "mlflow"
        if output_config.logs.tensorboard is None:
            output_config.logs.tensorboard = base / "tensorboard"

    # CheckPointSchema
    output_config.checkpoints.root = root_output_path / output_config.checkpoints.root


class BaseSchema(BaseModel):
    """Top-level schema for the training configuration."""

    data: DataSchema
    """Data configuration."""
    dataloader: DataLoaderSchema
    """Dataloader configuration."""
    datamodule: DataModuleSchema
    """Datamodule configuration."""
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
    def check_log_paths_available_for_loggers(self) -> Self:
        logger = []
        if self.diagnostics.log.wandb.enabled and (not self.system.output.logs or not self.system.output.logs.wandb):
            logger.append("wandb")
        if self.diagnostics.log.mlflow.enabled and (not self.system.output.logs or not self.system.output.logs.mlflow):
            logger.append("mlflow")
        if self.diagnostics.log.mlflow.enabled and (not self.diagnostics.log.mlflow.save_dir):
            self.diagnostics.log.mlflow.save_dir = str(self.hardware.paths.logs.mlflow)
        if self.diagnostics.log.tensorboard.enabled and (
            not self.system.output.logs or not self.system.output.logs.tensorboard
        ):
            logger.append("tensorboard")

        if logger:
            msg = ", ".join(logger) + " logging path(s) not provided."
            raise PydanticCustomError("logger_path_missing", msg)  # noqa: EM101
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

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)

    def model_post_init(self, _: Any) -> None:
        expand_paths(self.system)


class UnvalidatedBaseSchema(PydanticBaseModel):
    data: Any
    """Data configuration."""
    dataloader: Any
    """Dataloader configuration."""
    datamodule: Any
    """Datamodule configuration."""
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

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)

    def model_post_init(self, _: Any) -> None:
        expand_paths(self.system)


def convert_to_omegaconf(config: BaseSchema) -> dict:
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
