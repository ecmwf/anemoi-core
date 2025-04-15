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
import sys
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from pydantic._internal import _model_construction
from pydantic_core import PydanticCustomError
from pydantic_core import ValidationError

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)
from .data import DataSchema  # noqa: TC001
from .dataloader import DataLoaderSchema  # noqa: TC001
from .datamodule import DataModuleSchema  # noqa: TC001
from .diagnostics import DiagnosticsSchema  # noqa: TC001
from .graphs.base_graph import BaseGraphSchema  # noqa: TC001
from .hardware import HardwareSchema  # noqa: TC001
from .models.models import ModelSchema  # noqa: TC001
from .training import TrainingSchema  # noqa: TC001
from .utils import CUSTOM_MESSAGES
from .utils import BaseModel
from .utils import convert_errors

_object_setattr = _model_construction.object_setattr

LOGGER = logging.getLogger(__name__)


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
    hardware: HardwareSchema
    """Hardware configuration."""
    graph: BaseGraphSchema
    """Graph configuration."""
    model: ModelSchema
    """Model configuration."""
    training: TrainingSchema
    """Training configuration."""
    config_validation: bool = True
    """Flag to disable validation of the configuration"""

    @model_validator(mode="after")
    def set_read_group_size_if_not_provided(self) -> BaseSchema:
        if not self.dataloader.read_group_size:
            self.dataloader.read_group_size = self.hardware.num_gpus_per_model
        return self

    @model_validator(mode="after")
    def check_log_paths_available_for_loggers(self) -> BaseSchema:
        logger = []
        if self.diagnostics.log.wandb.enabled and (not self.hardware.paths.logs or not self.hardware.paths.logs.wandb):
            logger.append("wandb")
        if self.diagnostics.log.mlflow.enabled and (
            not self.hardware.paths.logs or not self.hardware.paths.logs.mlflow
        ):
            logger.append("mlflow")
        if self.diagnostics.log.tensorboard.enabled and (
            not self.hardware.paths.logs or not self.hardware.paths.logs.tensorboard
        ):
            logger.append("tensorboard")

        if logger:
            msg = ", ".join(logger) + " logging path(s) not provided."
            raise PydanticCustomError("logger_path_missing", msg)  # noqa: EM101
        return self

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)


class UnvalidatedBaseSchema(PydanticBaseModel):
    data: Any
    """Data configuration."""
    dataloader: Any
    """Dataloader configuration."""
    datamodule: Any
    """Datamodule configuration."""
    diagnostics: Any
    """Diagnostics configuration such as logging, plots and metrics."""
    hardware: Any
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
