# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

import logging
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import model_validator
from pydantic_core import PydanticCustomError

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)
from .data import DataSchema  # noqa: TC001
from .dataloader import DataLoaderSchema  # noqa: TC001
from .diagnostics import DiagnosticsSchema  # noqa: TC001
from .graphs.base_graph import BaseGraphSchema  # noqa: TC001
from .hardware import HardwareSchema  # noqa: TC001
from .models.models import GNNSchema  # noqa: TC001
from .models.models import GraphTransformerSchema  # noqa: TC001
from .models.models import TransformerSchema  # noqa: TC001
from .training import TrainingSchema  # noqa: TC001

LOGGER = logging.getLogger(__name__)


class BaseSchema(BaseModel):
    """Top-level schema for the training configuration."""

    data: DataSchema
    """Data configuration."""
    dataloader: DataLoaderSchema
    """Dataloader configuration."""
    diagnostics: DiagnosticsSchema
    """Diagnostics configuration such as logging, plots and metrics."""
    hardware: HardwareSchema
    """Hardware configuration."""
    graph: BaseGraphSchema
    """Graph configuration."""
    model: GNNSchema | TransformerSchema | GraphTransformerSchema
    """Model configuration."""
    training: TrainingSchema
    """Training configuration."""

    class Config:
        """Pydantic configuration."""

        use_attribute_docstrings = True
        use_enum_values = True
        validate_assignment = True
        validate_default = True
        extra = "allow"

    @model_validator(mode="after")
    def set_read_group_size_if_not_provided(self) -> BaseSchema:
        if not self.dataloader.read_group_size:
            self.dataloader.read_group_size = self.hardware.num_gpus_per_model
        return self

    @model_validator(mode="after")
    def adjust_lr_to_hardware_settings(self) -> BaseSchema:
        self.training.lr.rate = (
            self.hardware.num_nodes
            * self.hardware.num_gpus_per_node
            * self.training.lr.rate
            / self.hardware.num_gpus_per_model
        )
        return self

    @model_validator(mode="after")
    def check_log_paths_available_for_loggers(self) -> BaseSchema:
        logger = []
        if self.diagnostics.log.wandb.enabled and not self.hardware.paths.logs.wandb:
            logger.append("wandb")
        if self.diagnostics.log.mlflow.enabled and not self.hardware.paths.logs.mlflow:
            logger.append("mlflow")
        if self.diagnostics.log.tensorboard.enabled and not self.hardware.paths.logs.tensorboard:
            logger.append("tensorboard")

        if logger:
            msg = ", ".join(logger) + " logging path(s) not provided."
            raise PydanticCustomError("logger_path_missing", msg)  # noqa: EM101


class UnvalidatedBaseSchema(BaseModel):
    data: Any
    dataloader: Any
    diagnostics: Any
    hardware: Any
    graph: Any
    model: Any
    training: Any

    class Config:
        """Pydantic configuration."""

        extra = "allow"
        arbitrary_types_allowed = True


def convert_to_omegaconf(config: BaseSchema) -> dict:
    config = config.model_dump(by_alias=True)
    return OmegaConf.create(config)
