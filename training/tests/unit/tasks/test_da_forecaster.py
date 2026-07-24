# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import DAForecaster
from anemoi.training.utils.masks import NoOutputMask


def _make_index_collection(
    name_to_index: dict[str, int],
    *,
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
    corrector: list[str] | None = None,
) -> IndexCollection:
    cfg = DictConfig(
        {
            "forcing": forcing or [],
            "diagnostic": diagnostic or [],
            "corrector": corrector or [],
        },
    )
    return IndexCollection(cfg, name_to_index)


# ── steps / offsets ───────────────────────────────────────────────────────


def test_da_forecaster_steps_tag_da_cycles() -> None:
    task = DAForecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
        da_cycles=3,
    )
    steps = task.steps("training")
    assert [s["rollout_step"] for s in steps] == [0, 1, 2, 3, 4]
    assert [s["is_da"] for s in steps] == [True, True, True, False, False]


def test_da_forecaster_offsets_include_da_cycles() -> None:
    task = DAForecaster(multistep_input=2, multistep_output=1, timestep="6h", rollout={"start": 1}, da_cycles=3)
    # 2 inputs [-6h, 0] + (da_cycles + rollout)=4 outputs [6h,12h,18h,24h]
    offsets = task.get_offsets("training")
    assert offsets == [
        datetime.timedelta(hours=-6),
        datetime.timedelta(0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
        datetime.timedelta(hours=18),
        datetime.timedelta(hours=24),
    ]


def test_da_forecaster_zero_cycles_matches_forecaster_offsets() -> None:
    task = DAForecaster(multistep_input=2, multistep_output=1, timestep="6h", rollout={"start": 2, "maximum": 2})
    assert [s["is_da"] for s in task.steps("training")] == [False, False]
    assert len(task.get_offsets("training")) == 2 + 2  # inputs + 2 forecast outputs


def test_da_forecaster_metric_names() -> None:
    task = DAForecaster(multistep_input=1, multistep_output=1, timestep="6h", da_cycles=2)
    assert task.get_metric_name(rollout_step=0, is_da=True) == "_dacycle0"
    assert task.get_metric_name(rollout_step=1, is_da=True) == "_dacycle1"
    assert task.get_metric_name(rollout_step=2, is_da=False) == "_rstep0"


# ── DA blend ──────────────────────────────────────────────────────────────


def test_da_blend_uses_obs_where_present_pred_where_nan() -> None:
    # two variables: prog (prognostic), force (forcing). model input = [prog, force].
    name_to_index = {"prog": 0, "force": 1}
    data_indices = _make_index_collection(name_to_index, forcing=["force"])
    task = DAForecaster(multistep_input=2, multistep_output=1, timestep="6h", da_cycles=1)

    b, e, g = 1, 1, 3
    v_in = len(data_indices.data.input.full)  # prog + force = 2
    v_out = len(data_indices.model.output.prognostic)  # prog = 1

    x = torch.zeros((b, 2, e, g, v_in))
    # y_pred is the background prediction (prognostic space)
    y_pred = torch.full((b, 1, e, g, v_out), 99.0)

    # batch DATA_FULL: time dim = offsets = [-6h, 0, +6h] -> 3 steps; output at index 2
    batch = torch.zeros((b, 3, e, g, len(name_to_index)))
    # observation prognostic: present at grid 0, NaN at grid 1/2
    batch[:, 2, :, 0, 0] = 5.0
    batch[:, 2, :, 1, 0] = torch.nan
    batch[:, 2, :, 2, 0] = torch.nan
    batch[:, 2, :, :, 1] = 7.0  # forcing present everywhere

    x_out = task._advance_dataset_input_da(x, y_pred, batch, rollout_step=0, data_indices=data_indices)

    prog = x_out[0, -1, 0, :, 0]
    force = x_out[0, -1, 0, :, 1]
    # obs where present (grid 0 -> 5.0), background prediction where obs is NaN (grid 1,2 -> 99.0)
    assert torch.allclose(prog, torch.tensor([5.0, 99.0, 99.0]))
    # forcing carried straight from obs
    assert torch.allclose(force, torch.tensor([7.0, 7.0, 7.0]))


# ── corrector zeroing during forecast rollout ──────────────────────────────


def test_da_forecaster_zeros_corrector_during_forecast() -> None:
    name_to_index = {"prog": 0, "corr": 1}
    data_indices = _make_index_collection(name_to_index, corrector=["corr"])
    task = DAForecaster(multistep_input=2, multistep_output=1, timestep="6h", da_cycles=1)

    b, e, g = 1, 1, 2
    v_in = len(data_indices.model.input.name_to_index)  # prog + corr
    x = torch.ones((b, 2, e, g, v_in))
    corrector_pos = int(data_indices.model.input.corrector[0])
    x[..., corrector_pos] = 3.0  # non-zero corrector slots
    prog_out = len(data_indices.model.output.prognostic)
    y_pred = torch.zeros((b, 1, e, g, prog_out))
    # batch time dim = offsets [-6h,0,+6h,+12h] = 4 (da_cycles=1 + rollout=1)
    batch = torch.zeros((b, 4, e, g, len(name_to_index)))

    # forecast step advance (is_da=False path)
    x_out = task._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=1,
        data_indices=data_indices,
        output_mask=NoOutputMask(),
    )
    # the freshly advanced step must have zeroed corrector slots
    assert torch.all(x_out[:, -1, ..., corrector_pos] == 0.0)


def test_da_forecaster_advance_dispatch_by_is_da() -> None:
    name_to_index = {"prog": 0}
    data_indices = {"data": _make_index_collection(name_to_index)}
    task = DAForecaster(multistep_input=2, multistep_output=1, timestep="6h", da_cycles=1)

    b, e, g = 1, 1, 2
    x = {"data": torch.zeros((b, 2, e, g, 1))}
    y_pred = {"data": torch.full((b, 1, e, g, 1), 42.0)}
    batch = {"data": torch.zeros((b, 4, e, g, 1))}

    # is_da=True -> blend path fills last step from obs/pred (obs all zero here, not NaN)
    out = task.advance_input(x, y_pred, batch, rollout_step=0, is_da=True, data_indices=data_indices)
    assert out["data"].shape == (b, 2, e, g, 1)
