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
from typing import Union

from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeInt

from anemoi.training.schemas.utils import allowed_values

from .utils import BaseModel


class Checkpoint(BaseModel):
    every_n_epochs: str = "anemoi-by_epoch-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by epoch frequency."
    every_n_train_steps: str = "anemoi-by_step-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by step frequency."
    every_n_minutes: str = "anemoi-by_time-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by time frequency (minutes)."


class FilesSchema(PydanticBaseModel):
    dataset: Union[Path, dict[str, Path]]  # dict option for multiple datasets
    "Path to the dataset file."
    graph: Union[Path, None] = Field(default=None)
    "Path to the graph file."
    checkpoint: dict[str, str]
    "Each dictionary key is a checkpoint name, and the value is the path to the checkpoint file."
    warm_start: Union[str, None] = None


class Logs(PydanticBaseModel):
    wandb: Union[Path, None] = None
    "Path to output wandb logs."
    mlflow: Union[Path, None] = None
    "Path to output mlflow logs."
    tensorboard: Union[Path, None] = None
    "Path to output tensorboard logs."


class PathsSchema(BaseModel):
    data: Path
    "Path to the data directory."
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
