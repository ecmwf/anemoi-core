# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``training.night_mask``: the mask itself and its wiring into the training module."""

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
from anemoi.training.utils.night_mask import NightMask

# data space: insolation (forcing), ssr, t2m ; model output space: ssr, t2m
NAME_TO_INDEX = {"insolation": 0, "ssr": 1, "t2m": 2}
SSR, T2M = 0, 1  # model output positions
GRID = 4


def _indices() -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": ["insolation"], "diagnostic": [], "target": []}), NAME_TO_INDEX)


def _batch(insolation: list[list[float]]) -> torch.Tensor:
    """Batch ``(1, time, 1, grid, 3)`` with the given insolation per time and grid point."""
    ins = torch.tensor(insolation, dtype=torch.float32)
    batch = torch.zeros(1, ins.shape[0], 1, ins.shape[1], len(NAME_TO_INDEX))
    batch[0, :, 0, :, NAME_TO_INDEX["insolation"]] = ins
    return batch


def _mask(**kwargs) -> NightMask:
    kwargs.setdefault("variables", ["ssr"])
    kwargs.setdefault("accumulation_period", "1h")
    return NightMask(**kwargs)


def _apply(mask: NightMask, task, y_pred: torch.Tensor, batch: torch.Tensor, pre=None, **step) -> torch.Tensor:
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


# points: 0 always dark, 1 sunset (lit at 0h only), 2 sunrise (lit from 2h), 3 always lit
INSOLATION = [[0, 0.3, 0, 0.5], [0, 0, 0, 0.5], [0, 0, 0.1, 0.5], [0, 0, 0.3, 0.5]]


def test_two_ended_mask_on_a_temporal_downscaler(downscaler) -> None:
    y_pred = torch.full((1, 2, 1, GRID, 2), 7.0, requires_grad=True)

    out = _apply(_mask(), downscaler, y_pred, _batch(INSOLATION))

    ssr = out[0, :, 0, :, SSR]
    assert ssr[0].tolist() == [0.0, 7.0, 0.0, 7.0]  # output 1h, window (0h, 1h]: the sunset point is kept
    assert ssr[1].tolist() == [0.0, 0.0, 7.0, 7.0]  # output 2h, window (1h, 2h]: the sunrise point is kept
    assert torch.all(out[..., T2M] == 7.0)  # other variables untouched
    out.sum().backward()  # the gradient flows through the unmasked points
    assert y_pred.grad[0, 0, 0, :, SSR].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_input_is_not_modified(downscaler) -> None:
    y_pred = torch.full((1, 2, 1, GRID, 2), 7.0)
    _apply(_mask(), downscaler, y_pred, _batch(INSOLATION))
    assert torch.all(y_pred == 7.0)


def test_fill_value_and_test_are_in_normalised_space(downscaler) -> None:
    """The mask variable is compared to, and the outputs filled with, the normalised physical zero."""
    norm_add = torch.tensor([-1.0, -0.5, 0.0])  # insolation, ssr, t2m
    norm_mul = torch.tensor([2.0, 1.0, 1.0])
    pre = SimpleNamespace(processors={"normalizer": SimpleNamespace(_norm_add=norm_add, _norm_mul=norm_mul)})
    batch = _batch(INSOLATION)
    ins = NAME_TO_INDEX["insolation"]
    batch[..., ins] = batch[..., ins] * norm_mul[0] + norm_add[0]  # normalise the insolation
    y_pred = torch.full((1, 2, 1, GRID, 2), 3.0)

    out = _apply(_mask(), downscaler, y_pred, batch, pre=pre)

    assert out[0, 0, 0, :, SSR].tolist() == [-0.5, 3.0, -0.5, 3.0]  # filled with the normalised zero of ssr


def test_ensemble_dimension_broadcasts(downscaler) -> None:
    out = _apply(_mask(), downscaler, torch.ones(1, 2, 4, GRID, 2), _batch([[0.0] * GRID] * 4))
    assert torch.all(out[..., SSR] == 0.0)


def test_output_without_an_earlier_time_in_the_batch_uses_its_own_time(caplog) -> None:
    task = TemporalDownscaler(input_timestep="3h", output_timestep="1h", output_left_boundary=True)  # outputs 0h..2h
    mask = _mask()
    with caplog.at_level(logging.WARNING):
        current, previous = mask.time_indices(task)
        mask.time_indices(task)
    assert (current[0], previous[0]) == (0, 0)  # no -1h in the batch: only t is used
    assert (current[1], previous[1]) == (1, 0)
    assert sum("not in the batch" in r.message for r in caplog.records) == 1  # warned once


def test_forecaster_with_6h_accumulations() -> None:
    """Not specific to a task: a 6-hourly forecaster whose accumulations cover 6h."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    mask = _mask(accumulation_period="6h")
    current, previous = mask.time_indices(task, rollout_step=0)
    offsets = task.get_offsets()
    assert [offsets[i] for i in current] == [datetime.timedelta(hours=6)]
    assert [offsets[i] for i in previous] == [datetime.timedelta(0)]  # the input time


def test_grid_mismatch_raises(downscaler) -> None:
    with pytest.raises(ValueError, match="same grid"):
        _apply(_mask(), downscaler, torch.ones(1, 2, 1, GRID + 1, 2), _batch(INSOLATION))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(variables=[]), "at least one"),
        (dict(accumulation_period="0h"), "positive"),
    ],
)
def test_invalid_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _mask(**kwargs)


def test_validate_against_data_indices() -> None:
    _mask().validate(_indices())
    with pytest.raises(ValueError, match="mask variable 'cos_zenith'"):
        _mask(mask_variable="cos_zenith").validate(_indices(), "data")
    with pytest.raises(ValueError, match=r"\['insolation'\] are not model outputs"):
        _mask(variables=["ssr", "insolation"]).validate(_indices())  # a forcing is not an output


# --- wiring into the training module ---------------------------------------------------------------


def _module(**attrs) -> SimpleNamespace:
    return SimpleNamespace(supports_night_mask=True, **attrs)


def _config(night_mask) -> DictConfig:
    return DictConfig({"training": {"night_mask": night_mask}})


def test_build_night_masks_single_and_multi_dataset_forms() -> None:
    indices = {"data": _indices()}
    cfg = {"variables": ["ssr"], "accumulation_period": "1h"}

    assert BaseTrainingModule._build_night_masks(_module(), _config(None), indices) == {}

    masks = BaseTrainingModule._build_night_masks(_module(), _config(cfg), indices)
    assert list(masks) == ["data"]
    assert masks["data"].variables == ["ssr"]
    assert masks["data"].mask_variable == "insolation"

    masks = BaseTrainingModule._build_night_masks(_module(), _config({"datasets": {"data": cfg}}), indices)
    assert list(masks) == ["data"]


def test_build_night_masks_errors() -> None:
    indices = {"data": _indices()}
    cfg = {"variables": ["ssr"], "accumulation_period": "1h"}

    with pytest.raises(ValueError, match="not supported"):
        BaseTrainingModule._build_night_masks(SimpleNamespace(supports_night_mask=False), _config(cfg), indices)
    with pytest.raises(ValueError, match="unknown dataset 'other'"):
        BaseTrainingModule._build_night_masks(_module(), _config({"datasets": {"other": cfg}}), indices)
    with pytest.raises(ValueError, match="not model outputs"):
        BaseTrainingModule._build_night_masks(
            _module(), _config({"variables": ["insolation"], "accumulation_period": "1h"}), indices
        )


def test_apply_night_mask_uses_the_task_step(downscaler) -> None:
    module = _module(
        night_masks={"data": _mask()},
        task=downscaler,
        data_indices={"data": _indices()},
        model=SimpleNamespace(pre_processors={"data": None}),
    )
    y_pred = {"data": torch.full((1, 2, 1, GRID, 2), 7.0)}

    out = BaseTrainingModule._apply_night_mask(module, y_pred, {"data": _batch(INSOLATION)})

    assert out["data"][0, 0, 0, :, SSR].tolist() == [0.0, 7.0, 0.0, 7.0]
    assert torch.all(y_pred["data"] == 7.0)  # the caller's tensors are not modified

    module.night_masks = {}
    assert BaseTrainingModule._apply_night_mask(module, y_pred, {"data": _batch(INSOLATION)}) is y_pred
