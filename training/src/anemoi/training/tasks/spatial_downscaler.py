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
    """Spatial-downscaling task: predict high-resolution target datasets from
    lower-resolution (and same-grid conditioning) input datasets at matched valid times.

    There is no rollout and no input advancement. A sample is the set of
    ``input_datasets`` and ``output_datasets`` read at the configured integer
    offsets -- multi-input / multi-output when several offsets are given -- and
    every output offset must also be an input offset, so each predicted state
    has a source state at the same valid time.

    Offsets are integers in units of the dataset frequency; the frequency itself
    is attached once at data setup (see :func:`bind_task_frequency`).
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
        # Types, non-emptiness and uniqueness are enforced by the pydantic task schema at config
        # time. The one invariant with physics content is re-checked here because tasks are also
        # constructed programmatically: outputs must be a subset of inputs, no fallback of any kind.
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

    def bind_data_frequency(self, frequency: datetime.timedelta) -> None:
        """Late-bind the dataset frequency, converting integer offsets to timedeltas.

        No-op if already bound to the same frequency; otherwise (re)binds and overwrites the derived
        timedelta offsets. The multi-frequency guard lives in :func:`bind_task_frequency`.
        """
        if self._frequency == frequency:
            return

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
        # A dataset missing from the batch raises KeyError(name) here -- loud enough.
        return {name: inputs[name] for name in self.input_datasets}

    def get_targets(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        targets = super().get_targets(batch, **kwargs)
        return {name: targets[name] for name in self.output_datasets}

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
