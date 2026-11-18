# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

from anemoi.training.tasks import TemporalDownscaler


def test_temporal_downscaler_interior_offsets_only() -> None:
    """No boundaries: only interior interpolation steps are produced."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task._output_offsets == expected


def test_temporal_downscaler_left_boundary_included() -> None:
    """output_left_boundary=True adds t=0 to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=0), datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task._output_offsets == expected


def test_temporal_downscaler_right_boundary_included() -> None:
    """output_right_boundary=True adds t=input_timestep to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=True,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4), datetime.timedelta(hours=6)]
    assert task._output_offsets == expected


def test_temporal_downscaler_both_boundaries_included() -> None:
    """Both boundaries: offsets span the full [0h, input_timestep] range."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    expected = [
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=2),
        datetime.timedelta(hours=4),
        datetime.timedelta(hours=6),
    ]
    assert task._output_offsets == expected


def test_temporal_downscaler_num_output_timesteps_matches_offsets() -> None:
    """num_output_timesteps equals the length of output_offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    assert task.num_output_timesteps == len(task._output_offsets) == 4


def test_temporal_downscaler_input_offsets_are_boundary_pair() -> None:
    """Input offsets are always [0h, input_timestep] regardless of output settings."""
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    assert task._input_offsets == [datetime.timedelta(0), datetime.timedelta(hours=6)]


# --- night mask -----------------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_VARS = {"insolation": 0, "ssr": 1, "t2m": 2}  # full data space
_MODEL_OUT = {"ssr": 0, "t2m": 1}  # model output space (insolation is a forcing)


def _night_mask_task(**kwargs) -> TemporalDownscaler:
    return TemporalDownscaler(
        input_timestep="3h",
        output_timestep="1h",
        night_mask={"mask_variable": "insolation", "variables": ["ssr"]},
        **kwargs,
    )


def _indices() -> dict:
    model = SimpleNamespace(output=SimpleNamespace(name_to_index=_MODEL_OUT))
    return {"data": SimpleNamespace(name_to_index=_VARS, model=model)}


def _batch(insolation_per_time: list[list[float]]) -> dict:
    """Batch (bs=1, time, ens=1, grid, nvar) with the given insolation per time and grid point."""
    ins = torch.tensor(insolation_per_time)  # (time, grid)
    batch = torch.zeros(1, ins.shape[0], 1, ins.shape[1], len(_VARS))
    batch[0, :, 0, :, _VARS["insolation"]] = ins
    return {"data": batch}


def test_night_mask_two_ended() -> None:
    """Zero only where insolation is 0 at t AND t-1h; t2m untouched."""
    task = _night_mask_task()  # batch times 0,1,2,3h; outputs 1h,2h
    # grid points: 0 always dark, 1 sunset between 0h and 1h, 2 sunrise between 1h and 2h, 3 always lit
    batch = _batch([[0, 0.2, 0, 0.5], [0, 0, 0, 0.5], [0, 0, 0.1, 0.5], [0, 0, 0.3, 0.5]])
    y_pred = {"data": torch.full((1, 2, 1, 4, 2), 7.0, requires_grad=True)}

    out = task.mask_predictions(y_pred, batch, data_indices=_indices())["data"]

    ssr = out[0, :, 0, :, _MODEL_OUT["ssr"]]
    # output 1h (prev 0h): point 0 dark both -> 0; point 1 lit at 0h -> kept (sunset hour)
    assert ssr[0].tolist() == [0.0, 7.0, 0.0, 7.0]
    # output 2h (prev 1h): points 0 and 1 dark both -> 0; point 2 lit at 2h -> kept
    assert ssr[1].tolist() == [0.0, 0.0, 7.0, 7.0]
    assert torch.all(out[..., _MODEL_OUT["t2m"]] == 7.0)
    out.sum().backward()  # gradient flows through unmasked points
    assert y_pred["data"].grad is not None


def test_night_mask_uses_normalised_zero() -> None:
    """With a normaliser, the mask is tested and the fill applied in normalised space."""
    task = _night_mask_task()
    norm_add = torch.tensor([-1.0, -0.5, 0.0])
    norm_mul = torch.tensor([2.0, 1.0, 1.0])
    pre = {"data": SimpleNamespace(processors={"normalizer": SimpleNamespace(_norm_add=norm_add, _norm_mul=norm_mul)})}
    phys = torch.tensor([[0.0], [0.0], [0.4], [0.0]])  # point dark at 0h,1h; lit at 2h
    batch = _batch((phys * norm_mul[0] + norm_add[0]).tolist())
    y_pred = {"data": torch.full((1, 2, 1, 1, 2), 3.0)}

    out = task.mask_predictions(y_pred, batch, data_indices=_indices(), pre_processors=pre)["data"]

    assert out[0, 0, 0, 0, _MODEL_OUT["ssr"]] == pytest.approx(-0.5)  # normalised physical zero of ssr
    assert out[0, 1, 0, 0, _MODEL_OUT["ssr"]] == pytest.approx(3.0)


def test_night_mask_left_boundary_single_ended_and_ensemble_broadcast() -> None:
    """Output at t=0 has no previous time in the batch: only t is used. Ensemble dim broadcasts."""
    task = _night_mask_task(output_left_boundary=True)  # outputs 0h,1h,2h
    assert task._night_mask_time_pairs[0] == (0, None)
    batch = _batch([[0.0], [0.0], [0.0], [0.0]])
    y_pred = {"data": torch.ones(1, 3, 4, 1, 2)}  # 4 ensemble members

    out = task.mask_predictions(y_pred, batch, data_indices=_indices())["data"]

    assert torch.all(out[..., _MODEL_OUT["ssr"]] == 0.0)


def test_no_night_mask_is_identity() -> None:
    task = TemporalDownscaler(input_timestep="3h", output_timestep="1h")
    y_pred = {"data": torch.ones(1, 2, 1, 1, 2)}
    assert task.mask_predictions(y_pred, _batch([[0.0]] * 4), data_indices=_indices()) is y_pred


def test_night_mask_raises_if_mask_variable_missing() -> None:
    task = _night_mask_task()
    model = SimpleNamespace(output=SimpleNamespace(name_to_index=_MODEL_OUT))
    indices = {"data": SimpleNamespace(name_to_index={"ssr": 0, "t2m": 1}, model=model)}
    with pytest.raises(ValueError, match="no dataset contains the mask variable"):
        task.mask_predictions({"data": torch.ones(1, 2, 1, 1, 2)}, _batch([[0.0]] * 4), data_indices=indices)


def test_night_mask_raises_if_variable_not_model_output() -> None:
    task = TemporalDownscaler(
        input_timestep="3h",
        output_timestep="1h",
        night_mask={"mask_variable": "insolation", "variables": ["ssr", "ssrd"]},
    )
    with pytest.raises(ValueError, match="not model outputs"):
        task.mask_predictions({"data": torch.ones(1, 2, 1, 1, 2)}, _batch([[0.0]] * 4), data_indices=_indices())
