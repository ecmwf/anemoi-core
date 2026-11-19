# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.forcing_mask import ForcingMask

# data space: two forcings, then ssr and t2m; model output space: ssr, t2m
NAME_TO_INDEX = {"insolation": 0, "lsm": 1, "ssr": 2, "t2m": 3}
SSR, T2M = 0, 1
GRID = 4

# insolation per time (0h..3h) at 4 points: always dark, sunset (lit at 0h only), sunrise (lit from 2h), always lit
INSOLATION = [[0, 0.3, 0, 0.5], [0, 0, 0, 0.5], [0, 0, 0.1, 0.5], [0, 0, 0.3, 0.5]]


def _indices() -> IndexCollection:
    config = DictConfig({"forcing": ["insolation", "lsm"], "diagnostic": [], "target": []})
    return IndexCollection(config, NAME_TO_INDEX)


def _batch(**forcings: list[list[float]]) -> torch.Tensor:
    """Batch ``(1, time, 1, grid, n_data)``; each forcing is one list of grid values per time."""
    batch = torch.zeros(1, len(next(iter(forcings.values()))), 1, GRID, len(NAME_TO_INDEX))
    for name, values in forcings.items():
        batch[0, :, 0, :, NAME_TO_INDEX[name]] = torch.tensor(values, dtype=torch.float32)
    return batch


def _mask(**kwargs) -> ForcingMask:
    kwargs.setdefault("mask_variable", "insolation")
    kwargs.setdefault("fill_values", {0.0: ["ssr"]})
    return ForcingMask(**kwargs)


def _apply(mask: ForcingMask, task, y_pred: torch.Tensor, batch: torch.Tensor, pre=None) -> torch.Tensor:
    time_indices = mask.time_indices(task)
    return mask.apply(y_pred, batch, data_indices=_indices(), pre_processors=pre, time_indices=time_indices)


@pytest.fixture
def downscaler() -> TemporalDownscaler:
    """Batch times 0h..3h, outputs at 1h and 2h."""
    return TemporalDownscaler(input_timestep="3h", output_timestep="1h")


def test_condition_at_the_output_and_an_earlier_time(downscaler) -> None:
    y_pred = torch.full((1, 2, 2, GRID, 2), 7.0, requires_grad=True)  # two ensemble members

    out = _apply(_mask(offsets=["0h", "-1h"]), downscaler, y_pred, _batch(insolation=INSOLATION))

    ssr = out[0, :, 0, :, SSR]
    assert ssr[0].tolist() == [0.0, 7.0, 0.0, 7.0]  # output 1h: the sunset point is kept
    assert ssr[1].tolist() == [0.0, 0.0, 7.0, 7.0]  # output 2h: the sunrise point is kept
    assert torch.all(out[..., T2M] == 7.0)
    assert torch.all(y_pred == 7.0)  # the input is not modified
    out.sum().backward()
    assert y_pred.grad[0, 0, 0, :, SSR].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_combine_any_and_default_offsets(downscaler) -> None:
    y_pred = torch.full((1, 2, 1, GRID, 2), 7.0)
    batch = _batch(insolation=INSOLATION)  # output 2h: [0, 0, 0.1, 0.5] at 2h and [0, 0, 0, 0.5] at 1h

    any_ = _apply(_mask(offsets=["0h", "-1h"], combine="any"), downscaler, y_pred, batch)
    assert any_[0, 1, 0, :, SSR].tolist() == [0.0, 0.0, 0.0, 7.0]

    default = _apply(_mask(), downscaler, y_pred, batch)  # offsets [0h]
    assert default[0, 1, 0, :, SSR].tolist() == [0.0, 0.0, 7.0, 7.0]


def test_other_forcing_with_several_fill_values(downscaler) -> None:
    mask = _mask(mask_variable="lsm", operator="lt", threshold=0.5, fill_values={0.0: ["ssr"], 273.0: ["t2m"]})

    out = _apply(mask, downscaler, torch.full((1, 2, 1, GRID, 2), 7.0), _batch(lsm=[[0.0, 0.2, 0.7, 1.0]] * 4))

    assert out[0, 0, 0, :, SSR].tolist() == [0.0, 0.0, 7.0, 7.0]
    assert out[0, 0, 0, :, T2M].tolist() == [273.0, 273.0, 7.0, 7.0]


def test_threshold_and_fill_value_are_normalised(downscaler) -> None:
    norm_add = torch.tensor([-1.0, 0.0, -0.5, 0.0])
    norm_mul = torch.tensor([2.0, 1.0, 1.0, 1.0])
    pre = SimpleNamespace(processors={"normalizer": SimpleNamespace(_norm_add=norm_add, _norm_mul=norm_mul)})
    batch = _batch(insolation=INSOLATION)
    batch[..., 0] = batch[..., 0] * norm_mul[0] + norm_add[0]
    mask = _mask(threshold=0.2, fill_values={10.0: ["ssr"]})  # physical: insolation <= 0.2, ssr = 10

    out = _apply(mask, downscaler, torch.full((1, 2, 1, GRID, 2), 3.0), batch, pre=pre)

    assert out[0, 0, 0, :, SSR].tolist() == [9.5, 9.5, 9.5, 3.0]  # 10 * 1 - 0.5 in normalised units


def test_missing_offsets_and_runtime_errors(downscaler, caplog) -> None:
    task = TemporalDownscaler(input_timestep="3h", output_timestep="1h", output_left_boundary=True)
    mask = _mask(offsets=["0h", "-1h"])
    with caplog.at_level(logging.WARNING):
        indices = mask.time_indices(task)
        mask.time_indices(task)
    # the first output (0h) has no -1h in the batch: only its own time is used; warned once
    assert [task.get_offsets()[i] for i in indices[0].tolist()] == [datetime.timedelta(0)] * 2
    assert sum("not in the batch" in r.message for r in caplog.records) == 1

    with pytest.raises(ValueError, match="no offset"):
        _mask(offsets=["-10h"]).time_indices(downscaler)
    with pytest.raises(ValueError, match="grid points"):
        _apply(_mask(), downscaler, torch.ones(1, 2, 1, GRID + 1, 2), _batch(insolation=INSOLATION))


def test_from_config_with_the_imputer_layout() -> None:
    config = {"mask_variable": "insolation", "offsets": ["0h", "-1h"], "default": "none", 0: ["ssr"], "273.15": ["t2m"]}
    mask = ForcingMask.from_config(config)

    assert mask.fill_values == {0.0: ["ssr"], 273.15: ["t2m"]}
    assert mask.offsets == [datetime.timedelta(0), datetime.timedelta(hours=-1)]
    assert "default" in config  # the input is not modified


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({0: ["ssr"]}, "is required"),
        ({"mask_variable": "insolation"}, "at least one fill value"),
        ({"mask_variable": "insolation", "oops": ["ssr"], 0: ["ssr"]}, "unknown key 'oops'"),
        ({"mask_variable": "insolation", 0: ["ssr"], "0.0": ["ssrd"]}, "more than once"),
        ({"mask_variable": "insolation", 0: ["ssr"], 1: ["ssr"]}, "more than one fill value"),
        ({"mask_variable": "insolation", "operator": "eq", 0: ["ssr"]}, "unknown operator"),
        ({"mask_variable": "insolation", "offsets": [-3600], 0: ["ssr"]}, "duration strings"),
    ],
)
def test_from_config_errors(config, match) -> None:
    with pytest.raises(ValueError, match=match):
        ForcingMask.from_config(config)


def test_validate_against_data_indices() -> None:
    _mask().validate(_indices())
    with pytest.raises(ValueError, match="'cos_zenith' is not in the data"):
        _mask(mask_variable="cos_zenith").validate(_indices())
    with pytest.raises(ValueError, match="'ssr' is not a forcing"):
        _mask(mask_variable="ssr", fill_values={0.0: ["t2m"]}).validate(_indices())
    with pytest.raises(ValueError, match=r"\['lsm'\] are not model outputs"):
        _mask(fill_values={0.0: ["ssr", "lsm"]}).validate(_indices())


def test_training_module_wiring(downscaler) -> None:
    config = {"mask_variable": "insolation", "offsets": ["0h", "-1h"], 0: ["ssr"]}
    module = SimpleNamespace(supports_forcing_mask=True)
    indices = {"data": _indices()}

    def build(forcing_mask, module=module):
        config = DictConfig({"training": {"forcing_mask": forcing_mask}})
        return BaseTrainingModule._build_forcing_masks(module, config, indices)

    assert build(None) == {}
    assert list(build(config)) == ["data"]
    assert list(build({"datasets": {"data": config}})) == ["data"]
    with pytest.raises(ValueError, match="unknown dataset 'other'"):
        build({"datasets": {"other": config}})
    with pytest.raises(ValueError, match="not supported"):
        build(config, module=SimpleNamespace(supports_forcing_mask=False))

    module = SimpleNamespace(
        forcing_masks=build(config),
        task=downscaler,
        data_indices=indices,
        model=SimpleNamespace(pre_processors={"data": None}),
    )
    y_pred = {"data": torch.full((1, 2, 1, GRID, 2), 7.0)}
    out = BaseTrainingModule._apply_forcing_mask(module, y_pred, {"data": _batch(insolation=INSOLATION)})
    assert out["data"][0, 0, 0, :, SSR].tolist() == [0.0, 7.0, 0.0, 7.0]

    module.forcing_masks = {}
    assert BaseTrainingModule._apply_forcing_mask(module, y_pred, {"data": _batch(insolation=INSOLATION)}) is y_pred
