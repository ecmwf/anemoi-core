# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DA training is any training method run with a DAForecaster: the rollout loop follows the task's step policy.

``SingleTraining._step`` runs with the real task and a small differentiable stand-in for the model.
"""

import types
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from anemoi.models.data import TensorLayout
from anemoi.models.data.batch import Batch
from anemoi.models.data.sources import Source
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import DAForecaster
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks.base import BaseTask
from anemoi.training.train.corrector import InstrumentCorrectors
from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.utils.masks import NoOutputMask
from tests.batch_builders import build_batch

NAME_TO_INDEX = {"a": 0, "b": 1, "f": 2, "c": 3}  # a, b prognostic; f forcing; c corrector
LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
GRID = 6


def _indices() -> IndexCollection:
    return IndexCollection(OmegaConf.create({"forcing": ["f"], "diagnostic": [], "corrector": ["c"]}), NAME_TO_INDEX)


def _batch(num_times: int) -> Batch:
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(1, num_times, 1, GRID, len(NAME_TO_INDEX), generator=generator)
    data[0, 2:, 0, :2, NAME_TO_INDEX["a"]] = torch.nan  # missing observations after the inputs
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(GRID, 2)},
        layouts={"data": LAYOUT},
        variables={"data": list(NAME_TO_INDEX)},
        statistics={"data": {"mean": np.zeros(4), "stdev": np.ones(4)}},
    )


def _imputing_processor(view: Source, skip_imputation: bool = False, **_kwargs) -> Source:
    return view if skip_imputation else view.map_data(torch.nan_to_num)


def _module(task: BaseTask) -> SingleTraining:
    """A SingleTraining wired with what ``_step`` touches; the model scales the last input's prognostics."""
    module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(module)
    module.data_indices = {"data": _indices()}
    module.dataset_names = ["data"]
    module.target_dataset_names = ["data"]
    module.task = task
    module.model = types.SimpleNamespace(
        pre_processors={"data": _imputing_processor},
        post_processors={"data": _imputing_processor},
    )
    module.output_mask = {"data": NoOutputMask()}
    module.model_comm_group = None
    module.weight = torch.nn.Parameter(torch.tensor(0.5))
    module.forward_kwargs: list[dict[str, Any]] = []
    module.loss_calls: list[dict[str, Any]] = []
    module.advance_kwargs: list[dict[str, Any]] = []
    prog = _indices().model.input.prognostic

    def fake_forward(x: Batch, **kwargs) -> Batch:
        module.forward_kwargs.append(kwargs)
        last = x["data"].select_time([x["data"].time_size - 1]).select_variables(prog)
        return Batch({"data": last.map_data(lambda t: torch.tanh(t * module.weight))})

    def fake_compute_loss_metrics(y_pred: Batch, y: Batch, **kwargs) -> tuple:
        module.loss_calls.append(kwargs)
        target = y["data"].data[..., _indices().data.output.prognostic]
        # mask before the difference: a NaN target would otherwise make the gradient NaN
        valid = ~torch.isnan(target).expand_as(y_pred["data"].data)
        loss = (y_pred["data"].data - target.nan_to_num())[valid].pow(2).mean()
        return loss, {}, y_pred

    advance = task.advance_input

    def spy_advance(*args, **kwargs) -> Batch:
        module.advance_kwargs.append(kwargs)
        return advance(*args, **kwargs)

    module._dataset_loss_metric_name = lambda name: f"{name}_loss"
    module.forward = fake_forward
    module.compute_loss_metrics = fake_compute_loss_metrics
    task.advance_input = spy_advance
    return module


def _da_task(**kwargs) -> DAForecaster:
    return DAForecaster(2, 1, "6H", rollout={"start": 2, "maximum": 2}, **{"da_cycles": 2, **kwargs})


def _num_times(task: BaseTask) -> int:
    return len(task.get_offsets("training"))


def test_unweighted_da_cycles_compute_no_loss_and_the_loss_averages_the_forecast() -> None:
    task = _da_task(da_loss_weight=0.0)
    module = _module(task)
    original = module.compute_loss_metrics
    output = module._step(_batch(_num_times(task)))

    assert [call["is_da"] for call in module.loss_calls] == [False, False]
    assert len(output.predictions) == 4
    # The DA steps still run (their predictions advance the state) but add nothing: the loss is the
    # mean of the two forecast losses.
    forecast_losses = []
    module.compute_loss_metrics = lambda y_pred, y, **kw: (
        forecast_losses.append(original(y_pred, y, **kw)[0]) or (forecast_losses[-1], {}, y_pred)
    )
    output = module._step(_batch(_num_times(task)))
    torch.testing.assert_close(output.loss, sum(forecast_losses) / 2)


def test_exact_validation_rollout_loss_is_the_single_forecast_step() -> None:
    """Validation unrolls the DA cycles and one forecast step; the loss is that step's loss, not half of it."""
    task = _da_task(da_loss_weight=0.0, validation_rollout=1, exact_validation_rollout=True)
    module = _module(task)
    original = module.compute_loss_metrics
    forecast_losses = []
    module.compute_loss_metrics = lambda y_pred, y, **kw: (
        forecast_losses.append(original(y_pred, y, **kw)[0]) or (forecast_losses[-1], {}, y_pred)
    )
    output = module._step(_batch(_num_times(task)), validation_mode=True)
    assert len(output.predictions) == 3
    assert len(forecast_losses) == 1
    torch.testing.assert_close(output.loss, forecast_losses[0])


def test_weighted_da_cycles_add_to_the_forecast_mean() -> None:
    batch = _batch(_num_times(_da_task()))
    weighted = _module(_da_task(da_loss_weight=0.5))
    output = weighted._step(batch)
    # loss = (0.5 * (l_da0 + l_da1) + l_f0 + l_f1) / 2, and the DA-weight-1 total adds the full DA terms
    unweighted = _module(_da_task(da_loss_weight=0.0))._step(batch).loss
    full = _module(_da_task(da_loss_weight=1.0))._step(batch).loss
    torch.testing.assert_close(output.loss, unweighted + 0.5 * (full - unweighted))
    assert [call["is_da"] for call in weighted.loss_calls] == [True, True, False, False]


def test_da_cycles_advance_with_the_observations_and_forecasts_without() -> None:
    task = _da_task()
    module = _module(task)
    module._step(_batch(_num_times(task)))

    assert ["observed" in kwargs for kwargs in module.advance_kwargs] == [True, True, False]
    observed = module.advance_kwargs[0]["observed"]
    # the observations keep their missing values (no imputation)
    assert torch.isnan(observed["data"].data[0, 0, 0, :2, NAME_TO_INDEX["a"]]).all()


def test_flow_dependent_skip_reaches_the_model() -> None:
    task = _da_task(da_flow_dependent_skip=True)
    module = _module(task)
    module._step(_batch(_num_times(task)))

    has_skip = ["skip_input" in kwargs for kwargs in module.forward_kwargs]
    # zero base for the first cycle, background after each blend, none after a forecast step
    assert has_skip == [True, True, True, False]
    assert torch.equal(
        module.forward_kwargs[0]["skip_input"]["data"].data,
        torch.zeros_like(module.forward_kwargs[0]["skip_input"]["data"].data),
    )


def test_checkpointed_steps_give_the_same_loss_and_gradient() -> None:
    batch = _batch(_num_times(_da_task()))
    results = []
    for checkpoint_steps in (0, 3):
        module = _module(_da_task(da_loss_weight=0.3, checkpoint_steps=checkpoint_steps))
        loss = module._step(batch).loss
        (grad,) = torch.autograd.grad(loss, module.weight)
        results.append((loss.detach(), grad))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])


def test_gradient_flows_through_the_da_cycles() -> None:
    """The forecast loss depends on the weight through the DA cycles' predictions too."""
    batch = _batch(_num_times(_da_task()))
    module = _module(_da_task(da_loss_weight=0.0))
    loss = module._step(batch).loss
    (grad,) = torch.autograd.grad(loss, module.weight)
    assert grad.abs() > 0


def test_zero_cycles_is_the_forecaster() -> None:
    forecaster = Forecaster(2, 1, "6H", rollout={"start": 2, "maximum": 2})
    da = _da_task(da_cycles=0)
    batch = _batch(_num_times(forecaster))
    torch.testing.assert_close(_module(da)._step(batch).loss, _module(forecaster)._step(batch).loss)


def test_skip_input_needs_a_supporting_model() -> None:
    task = _da_task(da_flow_dependent_skip=True)
    assert task.requires_skip_input
    assert not _da_task().requires_skip_input


@pytest.mark.parametrize("validation_mode", [False, True])
def test_validation_runs_every_step(validation_mode: bool) -> None:
    task = DAForecaster(2, 1, "6H", rollout={"start": 1, "maximum": 1}, validation_rollout=3, da_cycles=2)
    module = _module(task)
    output = module._step(_batch(len(task.get_offsets())), validation_mode=validation_mode)
    assert len(output.predictions) == (5 if validation_mode else 3)


# -- ensemble ---------------------------------------------------------------------------------


class _EnsembleModel:
    """Model stand-in for EnsembleTraining.forward, which calls ``self.model(x, ...)``."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.pre_processors = {"data": _imputing_processor}
        self.post_processors = {"data": _imputing_processor}
        self.calls: list[dict[str, Any]] = []
        self._module = module

    def __call__(self, x: Batch, **kwargs) -> Batch:
        self.calls.append(kwargs)
        prog = _indices().model.input.prognostic
        last = x["data"].select_time([x["data"].time_size - 1]).select_variables(prog)
        return Batch({"data": last.map_data(lambda t: torch.tanh(t * self._module.weight))})


def _ensemble_module(task: BaseTask, members: int = 3) -> EnsembleTraining:
    single = _module(task)
    module = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(module)
    for name in ("data_indices", "dataset_names", "target_dataset_names", "task", "output_mask", "model_comm_group"):
        setattr(module, name, getattr(single, name))
    module.weight = torch.nn.Parameter(torch.tensor(0.5))
    module.nens_per_device = members
    module.ens_comm_subgroup_size = 1
    module.ens_comm_subgroup = None
    module.model = _EnsembleModel(module)
    module._dataset_loss_metric_name = lambda name: f"{name}_loss"
    module.compute_loss_metrics = single.compute_loss_metrics
    return module


def test_ensemble_da_tiles_members_and_conditions_on_the_continuous_step() -> None:
    task = _da_task(da_flow_dependent_skip=True)
    module = _ensemble_module(task)
    output = module._step(_batch(_num_times(task)), validation_mode=True)

    assert [call["fcstep"] for call in module.model.calls] == [0, 1, 2, 3]
    assert len(output.predictions) == 4
    # every model call sees the three members, and so does the zero residual base of the first cycle
    first_skip = module.model.calls[0]["skip_input"]["data"].data
    assert first_skip.shape[2] == 3
    assert torch.equal(first_skip, torch.zeros_like(first_skip))
    assert all(p["data"].data.shape[2] == 3 for p in output.predictions)


# -- correctors ---------------------------------------------------------------------------------


def test_corrector_changes_the_loss_but_not_the_state() -> None:
    batch = _batch(_num_times(_da_task()))
    plain = _module(_da_task(da_loss_weight=1.0))
    plain_output = plain._step(batch)

    corrected = _module(_da_task(da_loss_weight=1.0))
    corrector = InstrumentCorrectors(
        instrument_groups={"obs": {"corrector_variables": ["c"], "channels": ["a"]}},
        all_corrector_names=["c"],
        output_name_to_index=_indices().model.output.name_to_index,
        hidden_dim=4,
    )
    corrector.correctors["obs"].out.bias.data.fill_(0.3)  # a non-zero correction
    corrected.corrector = torch.nn.ModuleDict({"data": corrector})
    corrected_output = corrected._step(batch)

    assert not torch.allclose(corrected_output.loss, plain_output.loss)
    # the predictions returned and advanced are the raw ones: identical to the run without corrector
    for raw, plain_pred in zip(corrected_output.predictions, plain_output.predictions, strict=True):
        torch.testing.assert_close(raw["data"].data, plain_pred["data"].data)
    # and the corrector trains
    (grad,) = torch.autograd.grad(corrected_output.loss, corrector.correctors["obs"].out.bias)
    assert grad.abs().sum() > 0
