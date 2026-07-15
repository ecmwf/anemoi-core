# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence 2.0.

from __future__ import annotations

import datetime

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string


class SpatialDownscaler(BaseTask):
    """Single-valid-time task with explicit named source and target datasets.

    The public contract uses integer dataset-relative offsets (i.e. positions
    in the dataset's own time axis, not physical durations). The dataset
    frequency is not known at construction time -- it is only known once the
    datamodule has opened the data readers -- so this task starts out
    operating purely on integers and only converts to ``timedelta`` offsets
    once :meth:`bind_data_frequency` has been called (see
    :func:`bind_task_frequency`). Everything positional (batch indices,
    ordering, etc.) is identical before and after binding, since multiplying
    a sorted list of integers by a single positive frequency preserves order.
    """

    name: str = "spatial-downscaler"

    def __init__(
        self,
        input_datasets: list[str],
        output_datasets: list[str],
        input_offsets: list[int],
        output_offsets: list[int],
        **_kwargs,
    ) -> None:
        self.input_datasets = tuple(input_datasets)
        self.output_datasets = tuple(output_datasets)
        if not self.input_datasets or not self.output_datasets:
            raise ValueError("SpatialDownscaler requires non-empty input_datasets and output_datasets.")
        if len(set(self.input_datasets)) != len(self.input_datasets):
            raise ValueError(f"SpatialDownscaler input_datasets contains duplicates: {input_datasets}")
        if len(set(self.output_datasets)) != len(self.output_datasets):
            raise ValueError(f"SpatialDownscaler output_datasets contains duplicates: {output_datasets}")

        self._validate_offsets(input_offsets, "input_offsets")
        self._validate_offsets(output_offsets, "output_offsets")

        if len(set(input_offsets)) != len(input_offsets):
            raise ValueError(f"SpatialDownscaler input_offsets contains duplicates: {input_offsets}")
        if len(set(output_offsets)) != len(output_offsets):
            raise ValueError(f"SpatialDownscaler output_offsets contains duplicates: {output_offsets}")

        # THE invariant: outputs must be a subset of inputs. There is no fallback of any kind.
        missing = sorted(set(output_offsets) - set(input_offsets))
        if missing:
            raise ValueError(
                f"SpatialDownscaler output_offsets {missing} have no matching input_offsets; "
                f"output_offsets must be a subset of input_offsets ({sorted(set(input_offsets))})."
            )

        self._integer_input_offsets: list[int] = sorted(input_offsets)
        self._integer_output_offsets: list[int] = sorted(output_offsets)
        self._frequency: datetime.timedelta | None = None

        super().__init__(
            input_offsets=list(self._integer_input_offsets),
            output_offsets=list(self._integer_output_offsets),
        )

    @staticmethod
    def _validate_offsets(offsets: list[int], label: str) -> None:
        if not offsets:
            raise ValueError(f"SpatialDownscaler {label} must be a non-empty list of integers.")
        for offset in offsets:
            # bool is a subclass of int in Python; reject it explicitly since a
            # dataset-relative offset of True/False is never a meaningful value.
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ValueError(f"SpatialDownscaler {label} must contain only integers, got {offset!r} in {offsets}.")

    def bind_data_frequency(self, frequency: datetime.timedelta) -> None:
        """Late-bind the dataset frequency, converting integer offsets to timedeltas.

        No-op if already bound to the same frequency; raises if already bound
        to a different one.
        """
        if not isinstance(frequency, datetime.timedelta) or frequency <= datetime.timedelta(0):
            raise ValueError(f"SpatialDownscaler frequency must be a positive timedelta, got {frequency!r}.")

        if self._frequency is not None:
            if self._frequency == frequency:
                return
            raise ValueError(
                f"SpatialDownscaler is already bound to frequency {self._frequency}; "
                f"cannot rebind to a different frequency {frequency}.",
            )

        self._frequency = frequency
        # Multiplying a sorted list of ints by one positive scalar preserves order.
        self._input_offsets = [offset * frequency for offset in self._integer_input_offsets]
        self._output_offsets = [offset * frequency for offset in self._integer_output_offsets]
        self._offsets = sorted(set(self._input_offsets + self._output_offsets))

    def output_to_input_positions(self) -> list[int]:
        """For each output offset (in output order), return its position among the input offsets.

        Strict same-offset helper: an output offset with no matching input
        offset raises ``ValueError``. The ``__init__`` invariant already makes
        this unreachable in practice; this is defense-in-depth, not a fallback.
        """
        positions = []
        for offset in self._integer_output_offsets:
            try:
                positions.append(self._integer_input_offsets.index(offset))
            except ValueError as exc:
                raise ValueError(
                    f"SpatialDownscaler output offset {offset} has no matching input offset "
                    f"in {self._integer_input_offsets}.",
                ) from exc
        return positions

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
            raise KeyError(f"SpatialDownscaler {role}_datasets are missing from the loaded batch: {missing}")

    def _get_timestep_for_metadata(self) -> str:
        if self._frequency is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no bound data frequency; call bind_data_frequency() first "
                "(this normally happens in the datamodule via bind_task_frequency()).",
            )
        return frequency_to_string(self._frequency)

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information.

        Extends ``BaseTask.fill_metadata`` by recording the integer
        dataset-relative offsets and the dataset roles, which are not
        expressible in the base ``timesteps`` dict (which only carries
        physical timedeltas).
        """
        super().fill_metadata(md_dict)

        dataset_names = md_dict["metadata_inference"]["dataset_names"]
        for dataset_name in dataset_names:
            timesteps = md_dict["metadata_inference"][dataset_name]["timesteps"]
            timesteps["input_offsets"] = self._integer_input_offsets
            timesteps["output_offsets"] = self._integer_output_offsets

        md_dict["input_datasets"] = list(self.input_datasets)
        md_dict["output_datasets"] = list(self.output_datasets)


def bind_task_frequency(task, data_readers: dict) -> None:
    """Late-bind ``task`` to the (single) frequency of ``data_readers``, if supported.

    No-op unless ``task`` implements ``bind_data_frequency`` (e.g. tasks whose
    offsets are physical timedeltas from construction, such as ``Forecaster``,
    do not need this seam). Spatial downscaling requires every dataset to
    share one frequency; if the readers disagree, raise naming each dataset's
    frequency.
    """
    if not hasattr(task, "bind_data_frequency"):
        return

    frequencies = {name: dr.frequency for name, dr in data_readers.items()}
    distinct = {frequency_to_string(freq) for freq in frequencies.values()}
    if len(distinct) > 1:
        listing = ", ".join(f"{name}={frequency_to_string(freq)}" for name, freq in frequencies.items())
        raise ValueError(
            f"Spatial downscaling requires all datasets to share one frequency, but got: {listing}.",
        )

    task.bind_data_frequency(next(iter(frequencies.values())))
