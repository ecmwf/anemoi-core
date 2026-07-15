# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence 2.0.

import datetime

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import as_timedelta


class Downscaler(BaseTask):
    """Single-valid-time task with explicit named source and target datasets."""

    name: str = "downscaler"

    def __init__(
        self,
        input_datasets: list[str],
        output_datasets: list[str],
        input_offset: str = "0H",
        output_offset: str = "0H",
        **_kwargs,
    ) -> None:
        self.input_datasets = tuple(input_datasets)
        self.output_datasets = tuple(output_datasets)
        if not self.input_datasets or not self.output_datasets:
            raise ValueError("Downscaler requires non-empty input_datasets and output_datasets.")
        if len(set(self.input_datasets)) != len(self.input_datasets):
            raise ValueError("Downscaler input_datasets contains duplicates.")
        if len(set(self.output_datasets)) != len(self.output_datasets):
            raise ValueError("Downscaler output_datasets contains duplicates.")
        super().__init__(
            input_offsets=[as_timedelta(input_offset)],
            output_offsets=[as_timedelta(output_offset)],
        )

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        inputs = super().get_inputs(batch, data_indices, **kwargs)
        self._validate_roles(inputs, self.input_datasets, "input")
        return {name: inputs[name] for name in self.input_datasets}

    def get_targets(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        targets = super().get_targets(batch, **kwargs)
        self._validate_roles(targets, self.output_datasets, "output")
        return {name: targets[name] for name in self.output_datasets}

    @staticmethod
    def _validate_roles(batch: dict[str, torch.Tensor], names: tuple[str, ...], role: str) -> None:
        missing = [name for name in names if name not in batch]
        if missing:
            raise KeyError(f"Downscaler {role}_datasets are missing from the loaded batch: {missing}")

    def _get_timestep_for_metadata(self) -> str:
        return self._format_physical_offset(self._input_offsets[0])
