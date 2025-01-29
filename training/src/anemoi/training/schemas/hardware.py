# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from functools import partial
from pathlib import Path  # noqa: TC003
from typing import Annotated

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import field_validator
from pydantic import model_validator

from anemoi.training.schemas.utils import allowed_values


class Checkpoint(BaseModel):
    # TODO(HELEN): Discuss merging with diagnostics checkpoint.
    every_n_epochs: str = "anemoi-by_epoch-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by epoch frequency."
    every_n_train_steps: str = "anemoi-by_step-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by step frequency."
    every_n_minutes: str = "anemoi-by_time-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by time frequency (minutes)."


class FilesSchema(BaseModel):
    dataset: Path  # TODO(Helen): Change to FilePath, only posisble after refactor
    "Path to the dataset file."
    graph: Path | None = Field(default=None)
    "Path to the graph file."
    checkpoint: dict[str, str]
    "Each dictionary key is a checkpoint name, and the value is the path to the checkpoint file."
    warm_start: str | None = None


class Logs(BaseModel):
    # TODO(Helen): Discuss merging with logging in diagnsotics
    wandb: Path | None = None
    "Path to output wandb logs."
    mlflow: Path | None = None
    "Path to output mlflow logs."
    tensorboard: Path | None = None
    "Path to output tensorboard logs."


class PathsSchema(BaseModel):
    data: Path
    "Path to the data directory."
    grids: Path
    "Path to the grids directory."
    graph: Path
    "Path to the graph directory."
    output: Path
    "Path to the output directory."
    logs: Logs
    "Logging directories."
    checkpoints: Path
    "Path to the checkpoints directory."
    plots: Path
    "Path to the plots directory."
    profiler: Path
    "Path to the profiler directory."


class HardwareSchema(BaseModel):
    accelerator: Annotated[
        str,
        AfterValidator(partial(allowed_values, values=["cpu", "gpu", "auto", "cuda", "tpu"])),
    ] = "auto"
    "Accelerator to use for training."
    num_gpus_per_node: NonNegativeInt = 1
    "Number of GPUs per node."
    num_nodes: NonNegativeInt = 1
    "Number of nodes."
    num_gpus_per_model: NonNegativeInt = 1
    "Number of GPUs per model."
    files: FilesSchema
    "Files schema."
    paths: PathsSchema
    "Paths schema."

    @field_validator("num_gpus_per_node")
    @classmethod
    def check_valid_num_gpus_per_node(cls, num_gpus_per_node: int) -> int:
        assert num_gpus_per_node <= 4, "num_gpus_per_node must be less than 4"
        return num_gpus_per_node

    @model_validator(mode="before")
    @classmethod
    def check_valid_num_gpus_per_model(cls, data: dict) -> dict:
        assert (
            data["num_gpus_per_model"] <= data["num_gpus_per_node"]
        ), "num_gpus_per_model must be less than num_gpus_per_node"
        return data
