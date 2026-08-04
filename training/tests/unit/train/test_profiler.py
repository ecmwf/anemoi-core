# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.profiler import AnemoiProfiler


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), name_to_index)


def test_profiler_example_input_uses_task_num_input_timesteps() -> None:
    """Profiler example inputs slice with the instantiated task, not forecaster-only config keys."""
    profiler = AnemoiProfiler.__new__(AnemoiProfiler)
    profiler.task = TemporalDownscaler(input_timestep="18h", output_timestep="6h")
    profiler.config = DictConfig({"task": {}, "dataloader": {"read_group_size": 1}})
    profiler.data_indices = {"data": _make_minimal_index_collection({"A": 0, "B": 1})}

    batch = {"data": torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2)}

    class _DataModule:
        def train_dataloader(self) -> list[dict[str, torch.Tensor]]:
            return [batch]

    profiler.datamodule = _DataModule()

    example_input_array = profiler.get_example_input_array()

    torch.testing.assert_close(
        example_input_array["data"],
        batch["data"][:, : profiler.task.num_input_timesteps, ..., profiler.data_indices["data"].data.input.full],
    )
