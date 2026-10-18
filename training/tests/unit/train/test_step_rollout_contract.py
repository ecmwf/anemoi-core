# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contract tests for the rollout loop in ``SingleTraining._step`` and ``compute_loss_metrics``.

Two behaviours are pinned here:

* **Per-dataset losses are averaged over the training rollout.** 
  ``_combine_loss_and_metrics`` averages over the first ``task.num_steps`` steps, so 
  ``val_<dataset>_..._loss`` is comparable to ``train_*_loss`` even when validation unrolls further.

* **The input state must not be advanced after the final step.** Its result is never read, and
  producing it also normalises the whole target slice for nothing.

These use a real :class:`Batch` with small stubs, rather than the helpers in ``test_methods.py``,
so they exercise the actual view/layout plumbing that ``_step`` walks.
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
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.single import SingleTraining
from tests.batch_builders import build_batch

NAME_TO_INDEX = {"prog": 0, "force": 1}
FORCING = ["force"]
LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
BATCH, TIMES, ENSEMBLE, GRID, NVARS = 1, 2, 1, 4, 2


def make_batch() -> Batch:
    """A minimal single-dataset gridded batch with two time steps."""
    return build_batch(
        data={"data": torch.randn(BATCH, TIMES, ENSEMBLE, GRID, NVARS)},
        coordinates={"data": torch.zeros(GRID, 2)},
        layouts={"data": LAYOUT},
        variables={"data": list(NAME_TO_INDEX)},
        statistics={"data": {"mean": np.zeros(NVARS), "stdev": np.ones(NVARS)}},
    )


class StubTask:
    """Task exposing only what ``_step`` calls, recording its ``advance_input`` invocations."""

    def __init__(self, num_steps: int, num_validation_steps: int | None = None) -> None:
        self.num_steps = num_steps
        self.num_validation_steps = num_steps if num_validation_steps is None else num_validation_steps
        self.advance_calls: list[dict[str, Any]] = []

    def steps(self, mode: str = "training") -> tuple[dict[str, int], ...]:
        n = self.num_validation_steps if mode == "validation" else self.num_steps
        return tuple({"rollout_step": i} for i in range(n))

    def get_inputs(self, batch: Batch, **_kwargs) -> Batch:
        return batch.select(time=[0])

    def get_targets(self, batch: Batch, **_kwargs) -> tuple[Batch, Batch, Batch]:
        """Mirror the real contract: (targets, target_template, target_forcings)."""
        targets = batch.select(time=[1])
        return targets, targets, targets.select(variables={"data": [NAME_TO_INDEX["force"]]})

    def advance_input(self, x: Batch, *_args, **kwargs) -> Batch:
        self.advance_calls.append(kwargs)
        return x


def make_module(num_steps: int) -> SingleTraining:
    """Build a SingleTraining wired with the minimum ``_step`` touches."""
    module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(module)
    data_indices = IndexCollection(
        OmegaConf.create({"forcing": FORCING, "diagnostic": [], "target": []}),
        NAME_TO_INDEX,
    )
    module.data_indices = {"data": data_indices}
    module.dataset_names = ["data"]
    module.target_dataset_names = []
    module.task = StubTask(num_steps)
    passthrough = {"data": lambda view, **_kwargs: view}
    module.model = types.SimpleNamespace(pre_processors=passthrough, post_processors=passthrough)
    module.output_mask = {"data": None}
    module.model_comm_group = None

    module.forward_kwargs: list[dict[str, Any]] = []
    module.loss_metric_kwargs: list[dict[str, Any]] = []

    def fake_forward(x: Batch, **kwargs) -> Batch:
        module.forward_kwargs.append(kwargs)
        return x

    def fake_compute_loss_metrics(y_pred: Batch, _y: Batch, **kwargs) -> tuple:
        module.loss_metric_kwargs.append(kwargs)
        return torch.tensor(1.0), {}, y_pred

    module.forward = fake_forward
    module.compute_loss_metrics = fake_compute_loss_metrics
    return module


# --- the rollout loop ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 2, 5])
def test_advance_input_is_skipped_on_the_final_step(num_steps: int) -> None:
    """The final step's advanced input is never read, so it must not be computed."""
    module = make_module(num_steps)
    module._step(make_batch(), validation_mode=False)
    assert len(module.task.advance_calls) == num_steps - 1


@pytest.mark.parametrize("num_steps", [1, 3])
def test_step_still_returns_one_prediction_per_step(num_steps: int) -> None:
    """Skipping the last advance_input must not change the step's outputs."""
    module = make_module(num_steps)
    output = module._step(make_batch(), validation_mode=False)
    assert len(output.predictions) == num_steps
    # Each step contributes 1.0 and the total is averaged over steps.
    assert float(output.loss) == pytest.approx(1.0)


def test_target_forcings_reach_the_model_as_the_target_kwarg() -> None:
    """The decoder is conditioned on the *forcing* half of get_targets, not the loss target."""
    module = make_module(1)
    module._step(make_batch(), validation_mode=False)
    (kwargs,) = module.forward_kwargs
    forcings = kwargs["target_forcings"]
    # Narrowed to the forcing variables, and normalised in model-input space.
    assert forcings["data"].variables == FORCING


# --- per-dataset loss metric keys ---------------------------------------------------------------


def _module_for_metric_keys(monkeypatch: pytest.MonkeyPatch) -> BaseTrainingModule:
    """A module whose per-dataset loss computation is stubbed to a constant."""
    module = SingleTraining.__new__(SingleTraining)
    module.target_dataset_names = ["surface", "era5"]
    module.loss = {
        "surface": types.SimpleNamespace(name="lossvariablemapper"),
        "era5": types.SimpleNamespace(name="lossvariablemapper"),
    }
    monkeypatch.setattr(
        BaseTrainingModule,
        "compute_dataset_loss_metrics",
        lambda _self, y_pred, _y, **_kwargs: (torch.tensor(1.0), {}, y_pred),
    )
    return module


def _dummy_pair() -> tuple[Batch, Batch]:
    """Two-dataset batch; contents are irrelevant since the per-dataset loss is stubbed."""
    names = ("surface", "era5")
    batch = build_batch(
        data={name: torch.zeros(BATCH, 1, ENSEMBLE, GRID, NVARS) for name in names},
        coordinates={name: torch.zeros(GRID, 2) for name in names},
        layouts=dict.fromkeys(names, LAYOUT),
        variables={name: list(NAME_TO_INDEX) for name in names},
        statistics={name: {"mean": np.zeros(NVARS), "stdev": np.ones(NVARS)} for name in names},
    )
    return batch, batch


def test_loss_key_is_unsuffixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-dataset loss key carries no step suffix at any rollout step."""
    module = _module_for_metric_keys(monkeypatch)
    y_pred, y = _dummy_pair()
    for step in range(2):
        _, metrics, _ = module.compute_loss_metrics(y_pred, y, validation_mode=True, rollout_step=step)
        assert set(metrics) == {"surface_lossvariablemapper_loss", "era5_lossvariablemapper_loss"}


def test_dataset_loss_is_averaged_over_training_rollout_only() -> None:
    """Validation unrolls 3 steps, training 2: the per-dataset loss averages steps 1-2, metrics keep all 3."""
    module = SingleTraining.__new__(SingleTraining)
    module.target_dataset_names = ["era5"]
    module.loss = {"era5": types.SimpleNamespace(name="mse")}
    module.task = StubTask(num_steps=2, num_validation_steps=3)
    step_metrics = [
        {"era5_mse_loss": torch.tensor(float(k)), f"era5_mse_metric/era5/all/{k}": torch.tensor(float(k))}
        for k in (1, 2, 3)
    ]
    step_losses = [torch.tensor(float(k)) for k in (1, 2, 3)]
    output = module._combine_loss_and_metrics(step_losses, step_metrics, predictions=[])
    assert float(output.loss) == pytest.approx(1.5)
    assert float(output.metrics["era5_mse_loss"]) == pytest.approx(1.5)
    assert {k for k in output.metrics if "metric" in k} == {f"era5_mse_metric/era5/all/{k}" for k in (1, 2, 3)}


def test_loss_keys_are_absent_outside_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-dataset loss keys are a validation-only diagnostic."""
    module = _module_for_metric_keys(monkeypatch)
    y_pred, y = _dummy_pair()
    _, metrics, _ = module.compute_loss_metrics(y_pred, y, validation_mode=False, rollout_step=1)
    assert metrics == {}
