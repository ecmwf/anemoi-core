# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``training.forcing_mask``: the mask itself and its wiring into the training module."""

import datetime
import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.forcing_mask import ForcingMask

# data space: two forcings (insolation, lsm), then ssr and t2m ; model output space: ssr, t2m
NAME_TO_INDEX = {"insolation": 0, "lsm": 1, "ssr": 2, "t2m": 3}
SSR, T2M = 0, 1  # model output positions
GRID = 4


def _indices() -> IndexCollection:
    config = DictConfig({"forcing": ["insolation", "lsm"], "diagnostic": [], "target": []})
    return IndexCollection(config, NAME_TO_INDEX)


def _batch(**forcings: list[list[float]]) -> torch.Tensor:
    """Batch ``(1, time, 1, grid, n_data)``; each forcing is given as one list of grid values per time."""
    n_time = len(next(iter(forcings.values())))
    batch = torch.zeros(1, n_time, 1, GRID, len(NAME_TO_INDEX))
    for name, values in forcings.items():
        batch[0, :, 0, :, NAME_TO_INDEX[name]] = torch.tensor(values, dtype=torch.float32)
    return batch


def _mask(**kwargs) -> ForcingMask:
    """Mask on insolation; ``variables`` and ``fill_value`` are shorthand for a single ``fill_values`` entry."""
    variables = kwargs.pop("variables", ["ssr"])
    fill_value = kwargs.pop("fill_value", 0.0)
    kwargs.setdefault("fill_values", {fill_value: variables})
    kwargs.setdefault("mask_variable", "insolation")
    return ForcingMask(**kwargs)


def _apply(mask: ForcingMask, task, y_pred: torch.Tensor, batch: torch.Tensor, pre=None, **step) -> torch.Tensor:
    return mask.apply(
        y_pred,
        batch,
        data_indices=_indices(),
        pre_processors=pre,
        time_indices=mask.time_indices(task, **step),
    )


@pytest.fixture
def downscaler() -> TemporalDownscaler:
    """Batch times 0h, 1h, 2h, 3h; outputs at 1h and 2h."""
    return TemporalDownscaler(input_timestep="3h", output_timestep="1h")


# insolation per time (rows 0h..3h) at 4 points: 0 always dark, 1 sunset (lit at 0h only),
# 2 sunrise (lit from 2h), 3 always lit
INSOLATION = [[0, 0.3, 0, 0.5], [0, 0, 0, 0.5], [0, 0, 0.1, 0.5], [0, 0, 0.3, 0.5]]


def test_condition_at_the_output_and_an_earlier_time(downscaler) -> None:
    """The accumulation use case: zero where insolation <= 0 at t and at t - 1h."""
    y_pred = torch.full((1, 2, 1, GRID, 2), 7.0, requires_grad=True)

    out = _apply(_mask(offsets=["0h", "-1h"]), downscaler, y_pred, _batch(insolation=INSOLATION))

    ssr = out[0, :, 0, :, SSR]
    assert ssr[0].tolist() == [0.0, 7.0, 0.0, 7.0]  # output 1h, window (0h, 1h]: the sunset point is kept
    assert ssr[1].tolist() == [0.0, 0.0, 7.0, 7.0]  # output 2h, window (1h, 2h]: the sunrise point is kept
    assert torch.all(out[..., T2M] == 7.0)  # other variables untouched
    out.sum().backward()  # the gradient flows through the unmasked points
    assert y_pred.grad[0, 0, 0, :, SSR].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_combine_any_and_defaults(downscaler) -> None:
    y_pred = torch.full((1, 2, 1, GRID, 2), 7.0)
    batch = _batch(insolation=INSOLATION)

    # output 2h: insolation is [0, 0, 0.1, 0.5] at 2h and [0, 0, 0, 0.5] at 1h
    any_ = _apply(_mask(offsets=["0h", "-1h"], combine="any"), downscaler, y_pred, batch)
    assert any_[0, 1, 0, :, SSR].tolist() == [0.0, 0.0, 0.0, 7.0]  # dark at 2h or at 1h

    default = _apply(_mask(), downscaler, y_pred, batch)  # operator le, threshold 0, offsets [0h]
    assert default[0, 1, 0, :, SSR].tolist() == [0.0, 0.0, 7.0, 7.0]  # dark at 2h only


def test_another_forcing_operator_threshold_and_fill_value(downscaler) -> None:
    """Not tied to insolation: set t2m to 273 where the land-sea mask is below 0.5 (sea)."""
    lsm = [[0.0, 0.2, 0.7, 1.0]] * 4
    mask = _mask(variables=["t2m"], mask_variable="lsm", operator="lt", threshold=0.5, fill_value=273.0)

    out = _apply(mask, downscaler, torch.full((1, 2, 1, GRID, 2), 7.0), _batch(lsm=lsm))

    assert out[0, 0, 0, :, T2M].tolist() == [273.0, 273.0, 7.0, 7.0]
    assert torch.all(out[..., SSR] == 7.0)


def test_input_is_not_modified(downscaler) -> None:
    y_pred = torch.full((1, 2, 1, GRID, 2), 7.0)
    _apply(_mask(), downscaler, y_pred, _batch(insolation=INSOLATION))
    assert torch.all(y_pred == 7.0)


def test_threshold_and_fill_value_are_converted_to_normalised_space(downscaler) -> None:
    """Both are physical values: the forcing is compared to, and the outputs filled with, their normalised form."""
    norm_add = torch.tensor([-1.0, 0.0, -0.5, 0.0])  # insolation, lsm, ssr, t2m
    norm_mul = torch.tensor([2.0, 1.0, 1.0, 1.0])
    pre = SimpleNamespace(processors={"normalizer": SimpleNamespace(_norm_add=norm_add, _norm_mul=norm_mul)})
    batch = _batch(insolation=INSOLATION)
    ins = NAME_TO_INDEX["insolation"]
    batch[..., ins] = batch[..., ins] * norm_mul[0] + norm_add[0]  # normalise the insolation
    mask = _mask(threshold=0.2, fill_value=10.0)  # insolation <= 0.2 (physical), ssr = 10 (physical)

    out = _apply(mask, downscaler, torch.full((1, 2, 1, GRID, 2), 3.0), batch, pre=pre)

    # points 0, 1 and 2 are at or below 0.2 at 1h (0, 0 and 0): filled with the normalised 10 = 10 * 1 - 0.5
    assert out[0, 0, 0, :, SSR].tolist() == [9.5, 9.5, 9.5, 3.0]


def test_ensemble_dimension_broadcasts(downscaler) -> None:
    out = _apply(_mask(), downscaler, torch.ones(1, 2, 4, GRID, 2), _batch(insolation=[[0.0] * GRID] * 4))
    assert torch.all(out[..., SSR] == 0.0)


def test_offsets_missing_from_the_batch_are_ignored_and_warned_once(caplog) -> None:
    task = TemporalDownscaler(input_timestep="3h", output_timestep="1h", output_left_boundary=True)  # outputs 0h..2h
    mask = _mask(offsets=["0h", "-1h"])
    offsets = task.get_offsets()
    with caplog.at_level(logging.WARNING):
        indices = mask.time_indices(task)
        mask.time_indices(task)

    assert indices.shape == (3, 2)
    assert [offsets[i] for i in indices[0].tolist()] == [datetime.timedelta(0)] * 2  # -1h missing: only 0h is used
    assert [offsets[i] for i in indices[1].tolist()] == [datetime.timedelta(hours=1), datetime.timedelta(0)]
    assert sum("are not in the batch" in r.message for r in caplog.records) == 1


def test_no_offset_in_the_batch_raises(downscaler) -> None:
    with pytest.raises(ValueError, match="none of the times"):
        _mask(offsets=["-10h"]).time_indices(downscaler)


def test_forecaster_with_6h_offsets() -> None:
    """Not specific to a task: a 6-hourly forecaster, with the forcing at the output and the input time."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    indices = _mask(offsets=["0h", "-6h"]).time_indices(task, rollout_step=0)
    offsets = task.get_offsets()
    assert [offsets[i] for i in indices[0].tolist()] == [datetime.timedelta(hours=6), datetime.timedelta(0)]


def test_grid_mismatch_raises(downscaler) -> None:
    with pytest.raises(ValueError, match="same grid"):
        _apply(_mask(), downscaler, torch.ones(1, 2, 1, GRID + 1, 2), _batch(insolation=INSOLATION))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(variables=[]), "at least one fill value"),
        (dict(fill_values={0.0: ["ssr"], 1.0: ["ssr"]}), "more than one fill value"),
        (dict(operator="eq"), "unknown operator"),
        (dict(combine="most"), "unknown combine"),
        (dict(offsets=[]), "non-empty list"),
        (dict(offsets=[-3600]), "as strings"),  # a bare number is not a duration
        (dict(offsets=["0h", "0h"]), "distinct"),
    ],
)
def test_invalid_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _mask(**kwargs)


def test_validate_against_data_indices() -> None:
    _mask().validate(_indices())
    with pytest.raises(ValueError, match="'cos_zenith' is not in the data"):
        _mask(mask_variable="cos_zenith").validate(_indices(), "data")
    with pytest.raises(ValueError, match="'ssr' is not a forcing"):  # a prognostic variable must not decide
        _mask(mask_variable="ssr", variables=["t2m"]).validate(_indices())
    with pytest.raises(ValueError, match=r"\['lsm'\] are not model outputs"):  # a forcing is not an output
        _mask(variables=["ssr", "lsm"]).validate(_indices())


def test_several_fill_values_share_one_condition(downscaler) -> None:
    mask = _mask(fill_values={0.0: ["ssr"], 273.0: ["t2m"]})

    out = _apply(mask, downscaler, torch.full((1, 2, 1, GRID, 2), 7.0), _batch(insolation=INSOLATION))

    assert out[0, 1, 0, :, SSR].tolist() == [0.0, 0.0, 7.0, 7.0]  # output 2h: dark at the first two points
    assert out[0, 1, 0, :, T2M].tolist() == [273.0, 273.0, 7.0, 7.0]


def test_from_config_with_the_imputer_layout() -> None:
    config = {
        "mask_variable": "insolation",
        "operator": "le",
        "threshold": 0,
        "offsets": ["0h", "-1h"],
        "combine": "all",
        "default": "none",
        0: ["ssr", "ssrd"],  # an int key, as in YAML ``0:``
        "273.15": ["t2m"],  # a numeric string key
    }

    mask = ForcingMask.from_config(config)

    assert mask.fill_values == {0.0: ["ssr", "ssrd"], 273.15: ["t2m"]}
    assert mask.variables == ["ssr", "ssrd", "t2m"]
    assert mask.offsets == [datetime.timedelta(0), datetime.timedelta(hours=-1)]
    assert "insolation le 0.0" in repr(mask)
    assert config["default"] == "none"  # the input is not modified


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({0: ["ssr"]}, "`mask_variable` is required"),
        ({"mask_variable": "insolation"}, "at least one fill value"),
        ({"mask_variable": "insolation", 0: []}, "at least one fill value"),
        ({"mask_variable": "insolation", "oops": ["ssr"]}, "unknown key 'oops'"),
        ({"mask_variable": "insolation", True: ["ssr"]}, "unknown key True"),
        ({"mask_variable": "insolation", "default": "mean", 0: ["ssr"]}, "`default` must be 'none'"),
        ({"mask_variable": "insolation", 0: ["ssr"], "0.0": ["ssrd"]}, "given more than once"),
        ({"mask_variable": "insolation", 0: ["ssr"], 1: ["ssr"]}, "more than one fill value"),
    ],
)
def test_from_config_errors(config, match) -> None:
    with pytest.raises(ValueError, match=match):
        ForcingMask.from_config(config)


# --- wiring into the training module ---------------------------------------------------------------


def _module(**attrs) -> SimpleNamespace:
    return SimpleNamespace(supports_forcing_mask=True, **attrs)


def _config(forcing_mask) -> DictConfig:
    return DictConfig({"training": {"forcing_mask": forcing_mask}})


CONFIG = {"mask_variable": "insolation", "offsets": ["0h", "-1h"], 1.5: ["ssr"]}  # fill value: variables


def test_build_forcing_masks_single_and_multi_dataset_forms() -> None:
    indices = {"data": _indices()}

    assert BaseTrainingModule._build_forcing_masks(_module(), _config(None), indices) == {}

    masks = BaseTrainingModule._build_forcing_masks(_module(), _config(CONFIG), indices)
    assert list(masks) == ["data"]
    assert masks["data"].fill_values == {1.5: ["ssr"]}
    assert masks["data"].offsets == [datetime.timedelta(0), datetime.timedelta(hours=-1)]
    assert masks["data"].operator == "le"  # default

    masks = BaseTrainingModule._build_forcing_masks(_module(), _config({"datasets": {"data": CONFIG}}), indices)
    assert list(masks) == ["data"]


def test_build_forcing_masks_errors() -> None:
    indices = {"data": _indices()}

    with pytest.raises(ValueError, match="not supported"):
        BaseTrainingModule._build_forcing_masks(SimpleNamespace(supports_forcing_mask=False), _config(CONFIG), indices)
    with pytest.raises(ValueError, match="unknown dataset 'other'"):
        BaseTrainingModule._build_forcing_masks(_module(), _config({"datasets": {"other": CONFIG}}), indices)
    with pytest.raises(ValueError, match="not model outputs"):
        BaseTrainingModule._build_forcing_masks(
            _module(), _config({"mask_variable": "insolation", 0: ["lsm"]}), indices
        )


def test_apply_forcing_mask_uses_the_task_step(downscaler) -> None:
    module = _module(
        forcing_masks={"data": _mask(offsets=["0h", "-1h"])},
        task=downscaler,
        data_indices={"data": _indices()},
        model=SimpleNamespace(pre_processors={"data": None}),
    )
    y_pred = {"data": torch.full((1, 2, 1, GRID, 2), 7.0)}

    out = BaseTrainingModule._apply_forcing_mask(module, y_pred, {"data": _batch(insolation=INSOLATION)})

    assert out["data"][0, 0, 0, :, SSR].tolist() == [0.0, 7.0, 0.0, 7.0]
    assert torch.all(y_pred["data"] == 7.0)  # the caller's tensors are not modified

    module.forcing_masks = {}
    assert BaseTrainingModule._apply_forcing_mask(module, y_pred, {"data": _batch(insolation=INSOLATION)}) is y_pred
