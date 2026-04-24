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
from anemoi.training.tasks import SparseForecaster
from anemoi.training.utils.masks import NoOutputMask


def _make_index_collection() -> dict[str, IndexCollection]:
    cfg = DictConfig({"forcing": ["force"], "diagnostic": [], "target": []})
    return {"data": IndexCollection(cfg, {"prog": 0, "force": 1})}


def test_sparse_forecaster_preserves_datamodule_sparse_timing_metadata() -> None:
    task = SparseForecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "maximum": 2},
    )
    metadata = {
        "metadata_inference": {
            "dataset_names": ["data"],
            "data": {
                "timesteps": {
                    "relative_date_indices_training_by_dataset": {"data": [0, 2]},
                    "relative_date_indices_validation_by_dataset": {"data": [0, 2]},
                },
            },
        },
    }

    task.fill_metadata(metadata)

    timesteps = metadata["metadata_inference"]["data"]["timesteps"]
    assert timesteps["relative_date_indices_training_by_dataset"]["data"] == [0, 2]
    assert task.dataset_time_maps["data"] == {0: 0, 2: 1}


def test_sparse_forecaster_advance_input_reuses_latest_available_sparse_timestep() -> None:
    task = SparseForecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "maximum": 2},
    )
    metadata = {
        "metadata_inference": {
            "dataset_names": ["data"],
            "data": {
                "timesteps": {
                    "relative_date_indices_training_by_dataset": {"data": [0, 2]},
                    "relative_date_indices_validation_by_dataset": {"data": [0, 2]},
                },
            },
        },
    }
    task.fill_metadata(metadata)
    data_indices = _make_index_collection()

    # tensor dims: (batch, time, ens, grid, variable)
    batch = {
        "data": torch.tensor(
            [
                [
                    [[[1.0, 10.0]]],
                    [[[3.0, 30.0]]],
                ],
            ],
            dtype=torch.float32,
        ),
    }
    x = task.get_inputs(batch, data_indices)
    y = task.get_targets(batch, rollout_step=1)
    y_pred = {"data": torch.tensor([[[[[100.0]]]]], dtype=torch.float32)}

    updated = task.advance_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        data_indices=data_indices,
        output_mask={"data": NoOutputMask()},
        grid_shard_slice={"data": None},
    )

    # rollout_step=1 target is the exact sparse index at relative time 2
    torch.testing.assert_close(y["data"][0, 0, 0, 0, 0], torch.tensor(3.0))
    # next input time is relative time 1, so sparse rollout falls back to time 0
    # for forcings, while prognostics are overwritten by the model prediction.
    torch.testing.assert_close(updated["data"][0, 0, 0, 0, 0], torch.tensor(100.0))
    torch.testing.assert_close(updated["data"][0, 0, 0, 0, 1], torch.tensor(10.0))
