# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Any
from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import PositiveInt
from pydantic import RootModel
from pydantic import computed_field

from anemoi.utils.dates import frequency_to_timedelta

from .utils import BaseModel


class Frequency(RootModel):
    root: Any

    @computed_field
    def as_timedelta(self) -> datetime.timedelta:
        return frequency_to_timedelta(self.root)

    @computed_field
    def as_string(self) -> str:
        delta = self.as_timedelta

        if delta.days > 0 and delta.seconds == 0:
            return f"{delta.days}d"

        if delta.days == 0 and delta.seconds >= 3600 and delta.seconds % 3600 == 0:
            return f"{delta.seconds // 3600}h"

        if delta.days == 0 and delta.seconds >= 60 and delta.seconds % 60 == 0:
            return f"{delta.seconds // 60}m"

        if delta.days == 0 and delta.seconds < 60:
            return f"{delta.seconds}s"

        return str(delta)

    @computed_field
    def as_seconds(self) -> int:
        return int(self.as_timedelta.total_seconds())


class DatasetSchema(BaseModel):
    """Dataset configuration schema."""

    dataset: str | dict | Path | list[dict]
    "Dataset, see anemoi-datasets"
    start: datetime.datetime | None = Field(default=None)  # TODO(Helen): should be a before valiadtion
    "Starting datetime for sample of the dataset."
    end: datetime.datetime | None = Field(default=None)
    "Ending datetime [inclusive] for sample of the dataset."
    frequency: Frequency
    "Temporal resolution, frequency must be >= to dataset frequency."
    drop: list | None = Field(default=None)
    "List of variables to drop from dataset"


class LoaderSet(BaseModel):
    training: PositiveInt | None = Field(example=None)
    "Value for training dataset"
    validation: PositiveInt | None = Field(example=None)
    "Value for validation dataset"
    test: PositiveInt | None = Field(example=None)
    "Value for test dataset"


class FullGridIndicesSchema(BaseModel):
    target_: Literal["anemoi.training.data.grid_indices.FullGrid"] = Field(
        "anemoi.training.data.grid_indices.FullGrid",
        alias="_target_",
    )
    "Grid indices for full grid class implementation from anemoi.training.data.grid_indices."
    nodes_name: str = Field(examples=["data"])
    "Name of the grid nodes."


class MaskedGridIndicesSchema(BaseModel):
    target_: Literal["anemoi.training.data.grid_indices.MaskedGrid"] = Field(
        "anemoi.training.data.grid_indices.MaskedGrid",
        alias="_target_",
    )
    "Grid indices for masked grid class implementation from anemoi.training.data.grid_indices."
    nodes_name: str = Field(examples=["data"])
    "Name of the grid nodes."
    node_attribute_name: str = Field(examples=["indices_connected_nodes"])
    "Name of the nodes graph attribute used for masking."


class DataLoaderSchema(PydanticBaseModel):
    prefetch_factor: int = Field(example=2, ge=0)
    "Number of batches loaded in advance by each worker."
    pin_memory: bool = Field(example=True)
    "If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."
    num_workers: LoaderSet
    "Number of process per-GPU for batch distribution."
    batch_size: LoaderSet
    "Per-GPU batch size."
    limit_batches: LoaderSet = Field(example=None)
    "Limit number of batches to run. Default value null, will run on all the batches."
    training: DatasetSchema
    "Training DatasetSchema."
    validation: DatasetSchema
    "Validation DatasetSchema."
    test: DatasetSchema
    "Test DatasetSchema."
    validation_rollout: PositiveInt = Field(example=1)
    "Number of rollouts to use for validation, must be equal or greater than rollout expected by callbacks."
    # TODO(Helen): Ccheck that this equal or greater than the number of rollouts expected by callbacks ???
    read_group_size: PositiveInt = Field(example=None)
    "Number of GPUs per reader group. Defaults to number of GPUs (see BaseSchema validators)."
    grid_indices: FullGridIndicesSchema | MaskedGridIndicesSchema
    "Grid indice schema."
