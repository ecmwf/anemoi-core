# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from functools import partial
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeInt

from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import allowed_values


class Checkpoint(BaseModel):
    every_n_epochs: str = "anemoi-by_epoch-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by epoch frequency."
    every_n_train_steps: str = "anemoi-by_step-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by step frequency."
    every_n_minutes: str = "anemoi-by_time-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by time frequency (minutes)."


class FilesSchema(PydanticBaseModel):
    dataset: Path | dict[str, Path] | None = Field(default=None)  # dict option for multiple datasets
    "Path to the dataset file."
    graph: Path | None = None
    "Path to the graph file."
    truncation: Path | None = None
    "Path to the truncation matrix file."
    truncation_inv: Path | None = None
    "Path to the inverse truncation matrix file."
    checkpoint: dict[str, str]
    "Each dictionary key is a checkpoint name, and the value is the path to the checkpoint file."
    warm_start: str | None = None
    "Name of the checkpoint file to use for warm starting the training"


class Logs(PydanticBaseModel):
    wandb: Path | None = None
    "Path to output wandb logs."
    mlflow: Path | None = None
    "Path to output mlflow logs."
    tensorboard: Path | None = None
    "Path to output tensorboard logs."


class PathsSchema(BaseModel):
    data: Path | dict[str, Path] | None = None
    "Path to the data directory."
    graph: Path | None = None
    "Path to the graph directory."
    truncation: Path | None = None
    "Path to the truncation matrix directory."
    output: Path | None = None
    "Path to the output directory."
    logs: Logs | None = None
    "Logging directories."
    checkpoints: Path
    "Path to the checkpoints directory."
    plots: Path | None = None
    "Path to the plots directory."
    profiler: Path | None
    "Path to the profiler directory."
    warm_start: str | None = None
    "Path to the checkpoint to use for warm starting the training"


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
    num_gpus_per_ensemble: NonNegativeInt = 1
    "Number of GPUs per ensemble."
    files: FilesSchema
    "Files schema."
    paths: PathsSchema
    "Paths schema."
