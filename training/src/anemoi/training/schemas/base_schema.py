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

from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import model_validator

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)
from .data import DataSchema  # noqa: TC001
from .dataloader import DataLoaderSchema  # noqa: TC001
from .diagnostics import DiagnosticsSchema  # noqa: TC001
from .graphs.base_graph import BaseGraphSchema  # noqa: TC001
from .hardware import HardwareSchema  # noqa: TC001
from .models.models import GNNConfig  # noqa: TC001
from .models.models import GraphTransformerConfig  # noqa: TC001
from .models.models import TransformerConfig  # noqa: TC001
from .training import TrainingSchema  # noqa: TC001

LOGGER = logging.getLogger(__name__)


class BaseSchema(BaseModel):
    data: DataSchema
    dataloader: DataLoaderSchema
    diagnostics: DiagnosticsSchema
    hardware: HardwareSchema
    graph: BaseGraphSchema
    model: GNNConfig | TransformerConfig | GraphTransformerConfig
    training: TrainingSchema

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

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


def convert_to_omegaconf(config: BaseSchema) -> dict:

    config = {
        "data": config.data.model_dump(by_alias=True),
        "dataloader": config.dataloader.model_dump(),
        "diagnostics": config.diagnostics.model_dump(),
        "hardware": config.hardware.model_dump(),
        "graph": config.graph.model_dump(by_alias=True),
        "model": config.model.model_dump(by_alias=True),
        "training": config.training.model_dump(by_alias=True),
    }

    return OmegaConf.create(config)
