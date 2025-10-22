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
    checkpoint: Path | None = None
    "Path to checkpoint for resuming training."
    warm_start: str | None = None
    "Path of the checkpoint file to use for warm starting the training"


class CheckpointsSchema(BaseModel):
    root: Path = Path("checkpoints")
    "Root directory for saving checkpoint files."
    every_n_epochs: str = "anemoi-by_epoch-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by epoch frequency."
    every_n_train_steps: str = "anemoi-by_step-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by step frequency."
    every_n_minutes: str = "anemoi-by_time-epoch_{epoch:03d}-step_{step:06d}"
    "File name pattern for checkpoint files saved by time frequency (minutes)."

    def model_post_init(self, _):
        self.expand_paths()

    def expand_paths(self):
        self.every_n_epochs = str(self.root / self.every_n_epochs)
        self.every_n_train_steps = str(self.root / self.every_n_train_steps)
        self.every_n_minutes = str(self.root / self.every_n_minutes)
        return self


class Logs(PydanticBaseModel):
    root: Path = Path("logs")
    wandb: Path | None = None
    "Path to output wandb logs."
    mlflow: Path | None = None
    "Path to output mlflow logs."
    tensorboard: Path | None = None
    "Path to output tensorboard logs."

    def model_post_init(self, _):
        self.expand_paths()

    def expand_paths(self):
        base = self.root
        if self.wandb is None:
            self.wandb = base / "wandb"
        if self.mlflow is None:
            self.mlflow = base / "mlflow"
        if self.tensorboard is None:
            self.tensorboard = base / "tensorboard"


class OutputSchema(BaseModel):
    root: Path = Path("output")
    "Path to the output directory."
    logs: Logs | None = None
    "Logging directories."
    checkpoints: CheckpointsSchema = Field(default_factory=CheckpointsSchema)
    "Paths to the checkpoints."
    plots: Path | None = None
    "Path to the plots directory."
    profiler: Path | None
    "Path to the profiler directory."

    def model_post_init(self, _):
        self.expand_paths()

    def expand_paths(self):

        if self.plots:
            self.plots = self.root / self.plots
        if self.profiler:
            self.profiler = self.root / self.profiler
        if self.logs:
            self.logs.root = self.root / self.logs.root
            self.logs.expand_paths()

        self.checkpoints.root = self.root / self.checkpoints.root
        self.checkpoints.expand_paths()

        return self


class SystemSchema(BaseModel):
    hardware: HardwareSchema
    "Specification of hardware and compute resources available including the number of nodes, GPUs, and accelerator."
    input: InputSchema
    "Definitions of specific input and output artifacts used relative to the directories defined in `paths`."
    output: OutputSchema
    "High-level directory structure describing where data is read from."
