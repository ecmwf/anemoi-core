# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import RootModel
from pydantic import computed_field
from pydantic import model_validator
from pydantic_core import PydanticCustomError

from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.schemas import BaseModel


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


class DatasetConfigSchema(PydanticBaseModel):
    """Dictionary-style dataset config passed directly to open_dataset."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    dataset: str | Path | dict | list[dict]
    "Dataset source identifier."
    frequency: Frequency | None = Field(default=None)
    "Optional frequency requested from open_dataset."
    drop: list[str] | None = Field(default=None)
    "Optional list of variables to drop from the dataset."
    select: list[str] | None = Field(default=None)
    "Optional list of variables to select from the dataset."
    statistics: str | Path | None = Field(default=None)
    "Optional path to custom statistics file."
    step_start: NonNegativeInt | None = Field(default=None)
    "First forecast step to include, in hours (inclusive). Selects ForecastStepDataset when set."
    step_end: PositiveInt | None = Field(default=None)
    "Last forecast step to include, in hours (inclusive). Selects ForecastStepDataset when set."
    step_frequency: PositiveInt | None = Field(default=None)
    "Stride between selected forecast steps, in hours. Selects ForecastStepDataset when set."

    # Note this should be extended in the future to have a full schema for the keys
    # supported by open_dataset and be moved to anemoi-datasets.


class NativeDatasetSchema(BaseModel):
    """Dataset configuration schema."""

    dataset_config: str | DatasetConfigSchema | Path | list[dict] | None = None
    "Dataset definition passed to open_dataset."
    start: str | int | None = Field(default=None)
    "Starting datetime for sample of the dataset."
    end: str | int | None = Field(default=None)
    "Ending datetime [inclusive] for sample of the dataset."


class TrajectorySamplingSchema(PydanticBaseModel):
    """Trajectory anchor sampling configuration."""

    stride: int | None = Field(default=None)
    "Stride between anchor positions. None = window size (non-overlapping); 1 = every position."


class TrajectorySchema(PydanticBaseModel):
    """Trajectory (forecast) reader configuration for 5-D trajectory zarr datasets."""

    sampling: TrajectorySamplingSchema | None = Field(default=None)
    "Anchor sampling config. stride=None → non-overlapping; stride=1 → all; stride=N → step N."


class TrajectoryDatasetSchema(NativeDatasetSchema):
    """Dataset configuration schema."""

    trajectory: TrajectorySchema | None = Field(default=None)
    "Trajectory configuration."


class LoaderSet(BaseModel):
    training: NonNegativeInt | None = Field(example=None)
    "Value for training dataset"
    validation: NonNegativeInt | None = Field(example=None)
    "Value for validation dataset"
    test: NonNegativeInt | None = Field(example=None)
    "Value for test dataset"


ReaderSchema = NativeDatasetSchema | TrajectoryDatasetSchema
"""A single reader definition (one anemoi-datasets dataset with its time range)."""


class ParticipantsSchema(BaseModel):
    """A dataset realised by several participants (multi-domain training).

    All participants of a dataset provide the SAME variables and frequency and are consumed under
    the same dataset name by the model; one participant is drawn per batch. Grids may differ.
    """

    participants: dict[str, ReaderSchema]
    "Participant name -> reader definition. Names are the keys and hence unique."
    statistics_from: str | None = Field(default=None)
    "Participant whose statistics are used for the whole dataset. Defaults to the first participant."

    @model_validator(mode="after")
    def check_participants(self) -> Self:
        if not self.participants:
            msg = "'participants' must define at least one participant."
            error = "empty_participants"
            raise PydanticCustomError(error, msg)
        if self.statistics_from is not None and self.statistics_from not in self.participants:
            msg = (
                f"'statistics_from' must name one of the participants {sorted(self.participants)}, "
                f"got {self.statistics_from!r}."
            )
            error = "unknown_statistics_participant"
            raise PydanticCustomError(error, msg)
        return self


class _DatasetsSchema(BaseModel):
    """Common part of the per-stage dataset configuration.

    Each entry of ``datasets`` is either a single reader (the dataset has exactly one participant, named
    after the dataset) or a ``participants:`` block.
    """

    datasets: dict[str, ReaderSchema | ParticipantsSchema]
    "Dataset name -> reader definition or participants block."

    def participants(self, dataset_name: str) -> dict[str, ReaderSchema]:
        """Return ``{participant_name: reader}`` for a dataset; the single form maps the dataset onto itself."""
        entry = self.datasets[dataset_name]
        if isinstance(entry, ParticipantsSchema):
            return dict(entry.participants)
        return {dataset_name: entry}

    def iter_readers(self) -> Iterator[ReaderSchema]:
        """Iterate over every reader definition of every dataset, participants flattened."""
        for dataset_name in self.datasets:
            yield from self.participants(dataset_name).values()


class MultiDatasetSchema(_DatasetsSchema):
    """Configuration for a MultiDataset."""

    target_: Literal["anemoi.training.data.datasets.MultiDataset"] = Field(..., alias="_target_")

    @model_validator(mode="after")
    def check_single_participant_per_dataset(self) -> Self:
        counts = {name: len(self.participants(name)) for name in self.datasets}
        offending = {name: count for name, count in counts.items() if count > 1}
        if offending:
            msg = (
                "MultiDataset supports at most one participant per dataset; use "
                "'_target_: anemoi.training.data.datasets.MultiDomainDataset' for datasets with several "
                f"participants. Offending datasets (name: count): {offending}."
            )
            error = "multidataset_multiple_participants"
            raise PydanticCustomError(error, msg)
        return self


class MultiDomainDatasetSchema(_DatasetsSchema):
    """Configuration for a MultiDomainDataset."""

    target_: Literal["anemoi.training.data.datasets.MultiDomainDataset"] = Field(..., alias="_target_")


class DataLoaderSchema(PydanticBaseModel):

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    prefetch_factor: int = Field(example=2, ge=0)
    "Number of batches loaded in advance by each worker."
    pin_memory: bool = Field(example=True)
    "If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."
    persistent_workers: bool = Field(default=True)
    "Keep dataloader workers alive between epochs. Automatically disabled when the rollout changes between epochs."
    num_workers: LoaderSet
    "Number of process per-GPU for batch distribution."
    batch_size: LoaderSet
    "Per-GPU batch size."
    limit_batches: LoaderSet = Field(example=None)
    "Limit number of batches to run. Default value null, will run on all the batches."
    training: MultiDatasetSchema | MultiDomainDatasetSchema
    "Training DatasetSchema."
    validation: MultiDatasetSchema | MultiDomainDatasetSchema
    "Validation DatasetSchema."
    test: MultiDatasetSchema | MultiDomainDatasetSchema
    "Test DatasetSchema."
    read_group_size: PositiveInt = Field(example=None)
    "Number of GPUs per reader group. Defaults to number of GPUs (see BaseSchema validators)."
    trajectory_sampling: TrajectorySamplingSchema | None = Field(default=None)
    "Default trajectory anchor sampling used across all splits. stride=None → non-overlapping; stride=N → step N."
    multiprocessing_context: str | None = Field(default=None, examples=[None, "spawn", "fork", "forkserver"])
    "Multiprocessing context to use for workers. If None, the default context will be used"
