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


class InputSchema(PydanticBaseModel):
    dataset: Path | dict[str, Path] | None = Field(default=None)  # dict option for multiple datasets
    "Path to the dataset file."
    graph: Path | None = None
    "Path to the graph file."
    truncation: Path | None = None
    "Path to the truncation matrix file."
    truncation_inv: Path | None = None
    "Path to the inverse truncation matrix file."
    loss_matrices_path: Path | None = None
    "Path to the directory containing smoothing matrices for the multiscale loss"
    warm_start: Path | None = None
    "Path of the checkpoint file to use for warm starting the training"


class CheckpointsSchema(BaseModel):
    root: Path
    "Root directory for saving checkpoint files."
    every_n_epochs: str = "anemoi-by_epoch-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by epoch frequency."
    every_n_train_steps: str = "anemoi-by_step-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by step frequency."
    every_n_minutes: str = "anemoi-by_time-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by time frequency (minutes)."


class Logs(PydanticBaseModel):
    root: Path
    wandb: Path | None = None
    "Path to output wandb logs."
    mlflow: Path | None = None
    "Path to output mlflow logs."
    tensorboard: Path | None = None
    "Path to output tensorboard logs."


class OutputSchema(BaseModel):
    root: Path | None = None
    "Path to the output directory."
    logs: Logs | None = None
    "Logging directories."
    checkpoints: CheckpointsSchema = Field(default_factory=CheckpointsSchema)
    "Paths to the checkpoints."
    plots: Path | None = None
    "Path to the plots directory."
    profiler: Path | None
    "Path to the profiler directory."


class SystemSchema(BaseModel):
    hardware: HardwareSchema
    "Specification of hardware and compute resources available including the number of nodes, GPUs, and accelerator."
    input: InputSchema
    "Definitions of specific input and output artifacts used relative to the directories defined in `output`."
    output: OutputSchema
    "High-level directory structure describing where data is read from."
